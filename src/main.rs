#![allow(dead_code)]

extern crate common;
extern crate image;
extern crate kernel;
extern crate obj;

#[macro_use]
extern crate rustacuda;

#[macro_use]
extern crate lazy_static;

use common::{
    BoundingBox, Color, Material, Object, Polygon, Ray, ScratchSpace, Vector3, BLACK, WHITE,
};
use image::ImageBuffer;
use kernel::ROUND_COUNT;
use obj::Obj;
use std::path::Path;
use std::time::{Duration, Instant};
use rustacuda::prelude::*;
use rustacuda::error::CudaError;
use rustacuda::function::{BlockSize, GridSize};

mod grid;
mod matrix;
use matrix::Matrix44;

struct GpuPathTracer {
    module: Module,
    stream: Stream,
}

impl GpuPathTracer {
    fn trace(
        &self,
        grid_size: &GridSize,
        block_size: &BlockSize,
        base_x: u32,
        base_y: u32,
        width: u32,
        height: u32,
        round: u32,
        fov_adjustment: f32,
        image: &mut DeviceBuffer<common::Color>,
        polygons: &mut DeviceBuffer<common::Polygon>,
        objects: &mut DeviceBuffer<common::Object>,
        scratch_space: &mut DeviceBuffer<common::ScratchSpace>,
    ) -> Result<(), CudaError> {

        let module = &self.module;
        let stream = &self.stream;

        unsafe {
            launch!(module.trace<<<grid_size, block_size, 0, stream>>>(
                base_x, base_y, width, height, round, fov_adjustment, 
                image.as_device_ptr(), polygons.as_device_ptr(), objects.as_device_ptr(),
                objects.len(), scratch_space.as_device_ptr()
            ))?;
        }

        stream.synchronize()?;

        Ok(())   
    }
}


fn make_object(
    all_polygons: &mut Vec<Polygon>,
    all_grids: &mut Vec<grid::Grid>,
    obj: &Obj<obj::SimplePolygon>,
    material: Material,
    object_to_world: Matrix44,
) -> Object {
    let polygon_start = all_polygons.len();

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

        Polygon {
            vertices: [vertex1, vertex2, vertex3],
        }
    };

    for object in &obj.objects {
        for group in &object.groups {
            for poly in &group.polys {
                let index1 = poly[0];
                for others in poly[1..].windows(2) {
                    let polygon = make_polygon(index1, others[0], others[1]);
                    all_polygons.push(polygon);
                }
            }
        }
    }

    let polygon_end = all_polygons.len();

    let polygon_slice = &all_polygons[polygon_start..polygon_end];

    let bounding_box = make_bounding_box(polygon_slice);

    let mut grid = grid::Grid::new(&bounding_box, polygon_slice, polygon_start);
    let grid_device = grid.to_device();
    all_grids.push(grid);

    Object {
        polygon_start,
        polygon_end,
        material,
        bounding_box,
        grid: grid_device,
    }
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

fn to_millis(duration: Duration) -> f32 {
    ((duration.as_secs() as f32) * 1000.0) + ((duration.subsec_nanos() as f32) / 1000000.0)
}

fn make_bounding_box(polygons: &[Polygon]) -> BoundingBox {
    let mut bounding_box = BoundingBox {
        min_x: 100000.0,
        max_x: -100000.0,
        min_y: 100000.0,
        max_y: -100000.0,
        min_z: 100000.0,
        max_z: -100000.0,
    };
    for p in polygons {
        for v in &p.vertices {
            bounding_box.min_x = v.x.min(bounding_box.min_x);
            bounding_box.max_x = v.x.max(bounding_box.max_x);
            bounding_box.min_y = v.y.min(bounding_box.min_y);
            bounding_box.max_y = v.y.max(bounding_box.max_y);
            bounding_box.min_z = v.z.min(bounding_box.min_z);
            bounding_box.max_z = v.z.max(bounding_box.max_z);
        }
    }
    bounding_box
}

fn trace_gpu(
    height: u32,
    width: u32,
    fov_adjustment: f32,
    mut polygons: DeviceBuffer<Polygon>,
    mut objects: DeviceBuffer<Object>,
) {
    use std::ffi::CString;
    println!(env!("KERNEL_PTX_PATH"));
    let ptx_file = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
    let tracer = GpuPathTracer {
        module: Module::load_from_string(&ptx_file).unwrap(),
        stream: Stream::new(StreamFlags::NON_BLOCKING, None).unwrap(),
    };

    use color_ext::ColorExt;
    let mut image_device: DeviceBuffer<Color> = unsafe { DeviceBuffer::zeroed((width * height) as usize).unwrap() };

    let chunk_size_x = 32;
    let chunk_size_y = 64;
    let thread_count = (chunk_size_x * chunk_size_y) as usize;

    let mut scratch_space: DeviceBuffer<ScratchSpace> = unsafe { DeviceBuffer::zeroed(thread_count).unwrap() };

    let block_y = 16;
    let block_x = 32;

    let trace_start = Instant::now();

    let num_chunks_y = (height / chunk_size_y) + 1;
    let num_chunks_x = (width / chunk_size_x) + 1;
    let num_chunks = num_chunks_x * num_chunks_y;

    let block = BlockSize::xy(block_x, block_y);
    let grid = GridSize::xy(chunk_size_x / block_x, chunk_size_y / block_y);

    for chunk_y in 0..num_chunks_y {
        let base_y = chunk_y * chunk_size_y;
        for chunk_x in 0..num_chunks_x {
            let base_x = chunk_x * chunk_size_x;
            let block_start = Instant::now();

            for round in 0..ROUND_COUNT {
                let round_start = Instant::now();

                let err = tracer.trace(
                    &grid,
                    &block,
                    base_x,
                    base_y,
                    width,
                    height,
                    round,
                    fov_adjustment,
                    &mut image_device,
                    &mut polygons,
                    &mut objects,
                    &mut scratch_space,
                );
                match err {
                    Err(e) => println!("{:?}", e),
                    Ok(_) => {}
                }
                let round_time = round_start.elapsed();
                if ROUND_COUNT > 1 {
                    println!(
                        "Round {}/{} time: {:0.6}ms",
                        round,
                        ROUND_COUNT,
                        to_millis(round_time)
                    );
                }
            }
            let block_time = block_start.elapsed();
            println!(
                "Block time: {}/{} {:0.6}ms",
                (chunk_y * num_chunks_y + chunk_x),
                num_chunks,
                to_millis(block_time)
            );
        }
    }

    let trace_time = trace_start.elapsed();

    println!("{} polygons in scene", polygons.len());
    println!("Trace time: {:0.6}ms", to_millis(trace_time));

    let mut scratch_space_host : Vec<ScratchSpace> = Vec::with_capacity(thread_count);
    unsafe { scratch_space_host.set_len(thread_count); }
    scratch_space.copy_to(&mut scratch_space_host).unwrap();

    let mut rays_traced = 0u64;
    let mut triangle_intersections = 0u64;
    let mut bounding_box_intersections = 0u64;
    for i in 0..thread_count {
        rays_traced += scratch_space_host[i].rays_traced;
        triangle_intersections += scratch_space_host[i].triangle_intersections;
        bounding_box_intersections += scratch_space_host[i].bounding_box_intersections;
    }
    println!("Traced {} rays", rays_traced);
    println!(
        "Performed {} bounding-box intersections",
        bounding_box_intersections
    );
    println!(
        "Performed {} triangle intersections",
        triangle_intersections
    );

    let mut image_host : Vec<Color> = Vec::with_capacity((width * height) as usize);
    unsafe { image_host.set_len((width * height) as usize); }
    let transfer_start = Instant::now();
    image_device.copy_to(&mut image_host).unwrap();
    let transfer_time = transfer_start.elapsed();
    println!("Transfer time: {:0.6}ms", to_millis(transfer_time));

    let mut image_buffer = ImageBuffer::new(width, height);
    for y in 0..height {
        let line_start = y * width;
        for x in 0..width {
            let color = &image_host[(line_start + x) as usize];
            image_buffer.put_pixel(x, y, color.clamp().to_rgba());
        }
    }

    image_buffer.save("image_out.png").unwrap();
}

fn trace_cpu(
    height: u32,
    width: u32,
    fov_adjustment: f32,
    polygons: Vec<Polygon>,
    objects: Vec<Object>,
) {
    use color_ext::ColorExt;
    let mut image_device: Vec<Color> = vec![BLACK; (width * height) as usize];

    let zero_ray = Ray {
        direction: Vector3::zero(),
        origin: Vector3::zero(),
    };

    let mut scratch_space = ScratchSpace {
        rays: [
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
            zero_ray.clone(),
        ],
        masks: [BLACK; 16],
        num_rays: 0,
        rays_traced: 0,
        triangle_intersections: 0,
        bounding_box_intersections: 0,
    };

    let trace_start = Instant::now();

    for y in 0..height {
        for x in 0..width {
            for round in 0..ROUND_COUNT {
                unsafe {
                    kernel::trace_inner(
                        x,
                        y,
                        width,
                        height,
                        round,
                        fov_adjustment,
                        image_device.as_mut_ptr(),
                        polygons.as_ptr(),
                        objects.as_ptr(),
                        objects.len(),
                        &mut scratch_space,
                    );
                }
            }
        }
    }

    let trace_time = trace_start.elapsed();
    println!("Trace time: {:0.6}ms", to_millis(trace_time));

    let transfer_start = Instant::now();
    let mut image_host = ImageBuffer::new(width, height);
    for y in 0..height {
        let line_start = y * width;
        for x in 0..width {
            let color = &image_device[(line_start + x) as usize];
            image_host.put_pixel(x, y, color.clamp().to_rgba());
        }
    }
    let transfer_time = transfer_start.elapsed();
    println!("Transfer time: {:0.6}ms", to_millis(transfer_time));

    image_host.save("image_out.png").unwrap();
}

fn main() {
    let _ctx = rustacuda::quick_init().unwrap();

    let load_start = Instant::now();

    let mut all_polygons = vec![];
    let mut all_objects = vec![];
    let mut all_grids = vec![];

    let mesh_path = Path::new("resources/dragon.obj");
    let mesh: Obj<obj::SimplePolygon> = Obj::load(mesh_path).expect("Failed to load mesh");
    let object_to_world_matrix = Matrix44::translate(0.0, -3.0, -5.0) * Matrix44::scale_linear(0.5);
    let teapot_1_material = Material::Refractive { index: 1.5 };
    all_objects.push(make_object(
        &mut all_polygons,
        &mut all_grids,
        &mesh,
        teapot_1_material,
        object_to_world_matrix,
    ));

    let object_to_world_matrix =
        Matrix44::translate(-4.0, -3.0, -6.0) * Matrix44::scale_linear(0.3);
    let teapot_2_material = Material::Reflective;
    all_objects.push(make_object(
        &mut all_polygons,
        &mut all_grids,
        &mesh,
        teapot_2_material,
        object_to_world_matrix,
    ));

    let object_to_world_matrix = Matrix44::translate(4.0, -3.0, -6.0) * Matrix44::scale_linear(0.3);
    let teapot_3_material = Material::Diffuse {
        color: Color {
            red: 0.0,
            green: 1.0,
            blue: 0.0,
        },
        albedo: 0.36,
    };
    all_objects.push(make_object(
        &mut all_polygons,
        &mut all_grids,
        &mesh,
        teapot_3_material,
        object_to_world_matrix,
    ));

    let mesh_path = Path::new("resources/box2.obj");
    let mesh: Obj<obj::SimplePolygon> = Obj::load(mesh_path).expect("Failed to load mesh");
    let object_to_world_matrix = Matrix44::translate(0.0, 7.0, -5.0) * Matrix44::scale_linear(0.25)
        * Matrix44::translate(0.0, -(3.15 / 2.0), 0.0);
    let light_material = Material::Emissive {
        emission: WHITE.mul_s(2.0),
    };
    all_objects.push(make_object(
        &mut all_polygons,
        &mut all_grids,
        &mesh,
        light_material,
        object_to_world_matrix,
    ));

    let box_path = Path::new("resources/box.obj");
    let box_mesh: Obj<obj::SimplePolygon> = Obj::load(box_path).expect("Failed to load mesh");
    let box_material = Material::Diffuse {
        color: Color {
            red: 0.75,
            green: 0.75,
            blue: 0.75,
        },
        albedo: 0.36,
    };
    all_objects.push(make_object(
        &mut all_polygons,
        &mut all_grids,
        &box_mesh,
        box_material,
        Matrix44::identity(),
    ));

    let load_time = load_start.elapsed();
    println!("Load/Convert Time: {:0.6}ms", to_millis(load_time));

    let width = 1024u32 / 4;
    let height = 736u32 / 4;
    let fov = 90.0f32;
    let fov_adjustment = (fov.to_radians() / 2.0).tan();
    println!("{} polygons in scene", all_polygons.len());

    let polygons_device: DeviceBuffer<Polygon> = DeviceBuffer::from_slice(&all_polygons).unwrap();
    let objects_device: DeviceBuffer<Object> = DeviceBuffer::from_slice(&all_objects).unwrap();

    trace_gpu(
        height,
        width,
        fov_adjustment,
        polygons_device,
        objects_device,
    );
}
