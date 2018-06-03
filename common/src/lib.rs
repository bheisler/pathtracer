#![no_std]
#![feature(core_intrinsics)]
#![allow(dead_code)]

mod color;
mod material;
pub mod math;
mod polygon;
mod scratch;
mod vector;

pub use color::{Color, BLACK, WHITE};
pub use material::Material;
pub use polygon::Polygon;
pub use scratch::ScratchSpace;
pub use vector::Ray;
pub use vector::Vector3;
