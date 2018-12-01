use grid::GridDevice;
use material::Material;

#[derive(DeviceCopy)]
pub struct BoundingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
}

#[derive(DeviceCopy)]
pub struct Object {
    pub polygon_start: usize,
    pub polygon_end: usize,
    pub material: Material,
    pub bounding_box: BoundingBox,
    pub grid: GridDevice,
}
