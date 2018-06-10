use accel::UVec;
use common::{BoundingBox, GridDevice, IndexRange, Polygon};

const LAMBDA: f32 = 4.0;

pub struct Grid {
    cell_x: f32,
    cell_y: f32,
    cell_z: f32,

    n_x: u32,
    n_y: u32,
    n_z: u32,
    // Indexes into the shared polygon array
    polygon_indexes: UVec<isize>,

    // An index range for each grid cell.
    index_ranges: UVec<IndexRange>,
}

fn clamp(min: u32, value: u32, max: u32) -> u32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

impl Grid {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    pub fn new(bbox: &BoundingBox, polygons: &[Polygon], polygon_offset: usize) -> Grid {
        let d_x = bbox.max_x - bbox.min_x;
        let d_y = bbox.max_y - bbox.min_y;
        let d_z = bbox.max_z - bbox.min_z;

        let volume = d_x * d_y * d_z;

        let polygon_count = polygons.len() as f32;
        let cube_root = ((LAMBDA * polygon_count) / volume).cbrt();

        // Numbers of cells in each direction
        let n_x = clamp(1, (d_x * cube_root) as u32, 128);
        let n_y = clamp(1, (d_y * cube_root) as u32, 128);
        let n_z = clamp(1, (d_z * cube_root) as u32, 128);

        // Size of cell in each direction.
        let cell_x = d_x / (n_x as f32);
        let cell_y = d_y / (n_y as f32);
        let cell_z = d_z / (n_z as f32);

        let num_cells = n_x * n_y * n_z;

        // Build vectors of the polygon IDs in each cell
        let mut cell_polygons: Vec<Vec<usize>> = vec![vec![]; num_cells as usize];
        for i in 0..polygons.len() {
            let polygon_num = i + polygon_offset;

            let polygon_box = super::make_bounding_box(&polygons[i..i + 1]);

            // Convert to min/max cell coordinates
            let min_x = clamp(0, ((polygon_box.min_x - bbox.min_x) / cell_x) as u32, n_x - 1);
            let max_x = clamp(0, ((polygon_box.max_x - bbox.min_x) / cell_x) as u32, n_x - 1);

            let min_y = clamp(0, ((polygon_box.min_y - bbox.min_y) / cell_y) as u32, n_y - 1);
            let max_y = clamp(0, ((polygon_box.max_y - bbox.min_y) / cell_y) as u32, n_y - 1);

            let min_z = clamp(0, ((polygon_box.min_z - bbox.min_z) / cell_z) as u32, n_z - 1);
            let max_z = clamp(0, ((polygon_box.max_z - bbox.min_z) / cell_z) as u32, n_z - 1);

            for x in min_x..=max_x {
                for y in min_y..=max_y {
                    for z in min_z..=max_z {
                        let i = (z * n_y * n_x) + (y * n_x) + x;
                        cell_polygons[i as usize].push(polygon_num);
                    }
                }
            }
        }

        // Concatenate those vectors together into two buffers that we can send to the device easily
        let num_indexes = cell_polygons.iter().map(|vec| vec.len()).sum();

        let mut index_i = 0;

        let mut polygon_indexes: UVec<isize> = UVec::new(num_indexes).unwrap();
        let mut index_ranges: UVec<IndexRange> = UVec::new(num_cells as usize).unwrap();

        for z in 0..n_z {
            for y in 0..n_y {
                for x in 0..n_x {
                    let i = (z * n_y * n_x) + (y * n_x) + x;
                    let indexes_for_current_cell = &cell_polygons[i as usize];
                    
                    let start = index_i;
                    for index in indexes_for_current_cell {
                        polygon_indexes[index_i] = (*index) as isize;
                        index_i += 1;
                    }
                    let end = index_i;
                    index_ranges[i as usize] = IndexRange {
                        start: start as isize, 
                        stop: end as isize,
                    }
                }
            }
        }

        let grid = Grid {
            cell_x,
            cell_y,
            cell_z,
            n_x,
            n_y,
            n_z,
            polygon_indexes,
            index_ranges
        };
        grid
    }

    pub fn to_device(&self) -> GridDevice {
        GridDevice {
            cell_x: self.cell_x,
            cell_y: self.cell_y,
            cell_z: self.cell_z,

            n_x: self.n_x as i32,
            n_y: self.n_y as i32,
            n_z: self.n_z as i32,

            polygon_indexes: self.polygon_indexes.as_ptr(),

            index_ranges: self.index_ranges.as_ptr(),
        }
    }
}
