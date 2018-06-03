use color::Color;
use vector::Ray;

const NUM_RAYS: usize = 8;

pub struct ScratchSpace {
    pub rays: [Ray; NUM_RAYS],
    pub masks: [Color; NUM_RAYS],
    pub num_rays: i32,
}
