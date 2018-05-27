#![no_std]
#![feature(core_intrinsics)]
#![allow(dead_code)]

mod color;
mod material;
mod math;
mod polygon;
mod vector;

pub use color::Color;
pub use material::Material;
pub use polygon::Polygon;
pub use vector::Ray;
pub use vector::Vector3;
