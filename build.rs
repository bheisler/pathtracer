extern crate ptx_builder;

use std::process::exit;
use ptx_builder::prelude::*;

fn main() {
    if let Err(error) = build() {
        eprintln!("{}", BuildReporter::report(error));
        exit(1);
    }
}

fn build() -> Result<()> {
    let status = Builder::new("kernel")?.build()?;

    match status {
        BuildStatus::Success(output) => {
            // Provide the PTX Assembly location via env variable
            println!(
                "cargo:rustc-env=KERNEL_PTX_PATH={}",
                output.get_assembly_path().to_str().unwrap()
            );

            // Observe changes in kernel sources
            for path in output.source_files()? {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
            }
        }

        BuildStatus::NotNeeded => {
            println!("cargo:rustc-env=KERNEL_PTX_PATH=/dev/null");
        }
    };

    Ok(())
}