#![no_std]
extern crate common;

use common::{Color, Polygon, Material, Vector3};

#[no_mangle]
pub unsafe fn trace_inner(x: u32, y: u32, width: u32, height: u32, image: *mut Color,
    polygons: *const Polygon, polygon_count: usize,
    materials: *const Material, material_count: usize) {
    
    let i = (y * width + x) as isize;

    let polygon = 10000;
    let vertex = 1;

    if x < width && y < height {
        let color = common::Color {
            red: (*polygons.offset(polygon)).vertices[vertex].x,
            green: (*polygons.offset(polygon)).vertices[vertex].y,
            blue: (*polygons.offset(polygon)).vertices[vertex].z,
        };
        *image.offset(i) = color;
    }
}