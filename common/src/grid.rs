use rustacuda_core::memory::DevicePointer;

#[derive(Debug, Clone, DeviceCopy)]
pub struct IndexRange {
    pub start: isize,
    pub stop: isize,
}

impl Default for IndexRange {
    fn default() -> IndexRange {
        IndexRange {
            start: 0,
            stop: 0,
        }
    }
}

#[derive(DeviceCopy)]
pub struct GridDevice {
    pub cell_x: f32,
    pub cell_y: f32,
    pub cell_z: f32,

    pub n_x: i32,
    pub n_y: i32,
    pub n_z: i32,

    // Array of indexes into the shared polygon array
    pub polygon_indexes: DevicePointer<isize>,

    // An index range for each grid cell.
    pub index_ranges: DevicePointer<IndexRange>,
}
