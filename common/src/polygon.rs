use vector::Vector3;

pub struct Polygon {
    pub vertices: [Vector3; 3],
    pub normal: Vector3,
    pub material_idx: usize,
}