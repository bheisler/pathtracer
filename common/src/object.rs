use material::Material;

pub struct BoundingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
}

pub struct Object {
    pub polygon_start: usize,
    pub polygon_end: usize,
    pub material: Material,
    pub bounding_box: BoundingBox,
}
