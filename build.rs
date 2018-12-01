extern crate ptx_builder;

use ptx_builder::error::Result;
use ptx_builder::prelude::*;

fn main() -> Result<()> {
    CargoAdapter::with_env_var("KERNEL_PTX_PATH").build(Builder::new("kernel")?);
}