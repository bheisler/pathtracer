#![no_std]
#![feature(core_intrinsics)]
#![allow(dead_code)]

#[macro_use]
extern crate rustacuda_derive;
extern crate rustacuda_core;

mod color;
mod grid;
mod material;
pub mod math;
mod object;
mod polygon;
mod scratch;
mod vector;

pub use color::{Color, BLACK, WHITE};
pub use grid::{GridDevice, IndexRange};
pub use material::Material;
pub use object::{BoundingBox, Object};
pub use polygon::Polygon;
pub use scratch::ScratchSpace;
pub use vector::Ray;
pub use vector::Vector3;
