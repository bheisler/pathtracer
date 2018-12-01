use vector::Vector3;

#[derive(DeviceCopy)]
pub struct Polygon {
    pub vertices: [Vector3; 3],
}
