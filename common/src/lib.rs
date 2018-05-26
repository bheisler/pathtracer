#![no_std]
#![feature(core_intrinsics)]

mod color;
mod vector;
mod material;
mod polygon;

pub use color::Color;
pub use vector::Vector3;
pub use material::Material;
pub use polygon::Polygon;
pub use vector::Ray;