#![feature(proc_macro)]
#![allow(dead_code)]

extern crate accel;
extern crate accel_derive;
extern crate common;
extern crate image;
extern crate kernel;
extern crate obj;

use accel::*;
use accel_derive::kernel;
use common::{Color, Material, Polygon, Vector3};
use image::ImageBuffer;
use obj::Obj;
use std::path::Path;
use std::time::Instant;

mod matrix;

use matrix::Matrix44;

#[kernel]
#[crate("accel-core" = "0.2.0-alpha")]
#[crate_path("kernel" = "../kernel")]
#[crate_path("common" = "../common")]
#[build_path(".kernel")]
pub unsafe fn trace(
    base_x: u32,
    base_y: u32,
    width: u32,
    height: u32,
    fov_adjustment: f32,
    image: *mut common::Color,
    polygons: *const common::Polygon,
    polygon_count: usize,
    materials: *const common::Material,
    material_count: usize,
) {
    use accel_core::*;

    let x = base_x + (nvptx_block_idx_x() * nvptx_block_dim_x() + nvptx_thread_idx_x()) as u32;
    let y = base_y + (nvptx_block_idx_y() * nvptx_block_dim_y() + nvptx_thread_idx_y()) as u32;

    kernel::trace_inner(
        x,
        y,
        width,
        height,
        fov_adjustment,
        image,
        polygons,
        polygon_count,
        materials,
        material_count,
    );
}

fn convert_objects_to_polygons(
    obj: &Obj<obj::SimplePolygon>,
    material_idx: usize,
    object_to_world: Matrix44,
) -> Vec<Polygon> {
    let mut polygons = vec![];

    let make_vector = |floats: &[f32; 3]| {
        let v = Vector3 {
            x: floats[0],
            y: floats[1],
            z: floats[2],
        };

        let t = object_to_world.clone() * v;
        t
    };

    let make_polygon = |index1, index2, index3| {
        let obj::IndexTuple(index1, _, _) = index1;
        let obj::IndexTuple(index2, _, _) = index2;
        let obj::IndexTuple(index3, _, _) = index3;

        let vertex1 = make_vector(&obj.position[index1]);
        let vertex2 = make_vector(&obj.position[index2]);
        let vertex3 = make_vector(&obj.position[index3]);

        let a = vertex2.sub(vertex1);
        let b = vertex3.sub(vertex1);

        let normal = a.cross(b).normalize();

        Polygon {
            vertices: [vertex1, vertex2, vertex3],
            normal,
            material_idx,
        }
    };

    for object in &obj.objects {
        for group in &object.groups {
            for poly in &group.polys {
                let index1 = poly[0];
                for others in poly[1..].windows(2) {
                    let polygon = make_polygon(index1, others[0], others[1]);
                    polygons.push(polygon);
                }
            }
        }
    }

    return polygons;
}

// This has to go here because the powf function doesn't exist in no_std and the intrisic breaks
// the linker. *sigh*
mod color_ext {
    use common::Color;
    use image::Rgba;

    const GAMMA: f32 = 2.2;

    fn gamma_encode(linear: f32) -> f32 {
        linear.powf(1.0 / GAMMA)
    }

    pub trait ColorExt {
        fn to_rgba(&self) -> Rgba<u8>;
    }
    impl ColorExt for Color {
        fn to_rgba(&self) -> Rgba<u8> {
            Rgba {
                data: [
                    (gamma_encode(self.red) * 255.0) as u8,
                    (gamma_encode(self.green) * 255.0) as u8,
                    (gamma_encode(self.blue) * 255.0) as u8,
                    255,
                ],
            }
        }
    }
}

fn main() {
    use color_ext::ColorExt;

    let load_start = Instant::now();
    let mesh_path = Path::new("resources/utah-teapot.obj");
    let mesh: Obj<obj::SimplePolygon> = Obj::load(mesh_path).expect("Failed to load mesh");

    let object_to_world_matrix = Matrix44::translate(0.0, 0.0, -5.0) * Matrix44::scale_linear(1.0)
        * Matrix44::translate(0.0, -(3.15 / 2.0), 0.0);
    let polygons = convert_objects_to_polygons(&mesh, 0, object_to_world_matrix);

    let load_time = load_start.elapsed();
    println!("Load/Convert Time: {:?}", load_time);

    // TODO: Allow arbitrary image sizes, not just multiples of 32.
    let width = 1024u32 * 2;
    let height = 736u32 * 2;
    let fov = 90.0f32;
    let fov_adjustment = (fov.to_radians() / 2.0).tan();
    let mut image_device: UVec<Color> = UVec::new((width * height) as usize).unwrap();
    let material_count = 1;
    let mut materials_device: UVec<Material> = UVec::new(material_count).unwrap();
    materials_device[0] = Material {
        color: Color {
            red: 0.0,
            green: 1.0,
            blue: 0.0,
        },
        albedo: 0.18,
    };
    let polygon_count = polygons.len();
    println!("{} polygons in scene", polygon_count);
    let mut polygons_device: UVec<Polygon> = UVec::new(polygon_count).unwrap();
    for (i, poly) in polygons.into_iter().enumerate() {
        polygons_device[i] = poly;
    }

    let chunk_size_x = 256;
    let chunk_size_y = 128;

    let grid = Grid::xy(8, 4);
    let block = Block::xy(chunk_size_x / grid.x, chunk_size_y / grid.y);

    let trace_start = Instant::now();

    for chunk_y in 0..(height / chunk_size_y) {
        let base_y = chunk_y * chunk_size_y;
        for chunk_x in 0..(width / chunk_size_x) {
            let base_x = chunk_x * chunk_size_x;
            trace(
                grid,
                block,
                base_x,
                base_y,
                width,
                height,
                fov_adjustment,
                image_device.as_mut_ptr(),
                polygons_device.as_ptr(),
                polygon_count,
                materials_device.as_ptr(),
                material_count,
            );
            let err = device::sync();
            match err {
                Err(e) => println!("{:?}", e),
                Ok(_) => {}
            }
        }
    }

    let trace_time = trace_start.elapsed();
    println!("Trace time: {:?}", trace_time);

    let transfer_start = Instant::now();
    let mut image_host = ImageBuffer::new(width, height);
    for y in 0..height {
        let line_start = y * width;
        for x in 0..width {
            let color = &image_device[(line_start + x) as usize];
            image_host.put_pixel(x, y, color.to_rgba());
        }
    }
    let transfer_time = transfer_start.elapsed();
    println!("Transfer time: {:?}", transfer_time);

    image_host.save("image_out.png").unwrap();
}
