use material::Material;

pub struct Object {
    pub polygon_start: usize,
    pub polygon_end: usize,
    pub material: Material,
}
