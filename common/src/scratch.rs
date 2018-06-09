use color::Color;
use vector::Ray;

const NUM_RAYS: usize = 8;

pub struct ScratchSpace {
    pub rays: [Ray; NUM_RAYS],
    pub masks: [Color; NUM_RAYS],
    pub num_rays: i32,
    pub rays_traced: u64,
    pub triangle_intersections: u64,
}
impl ScratchSpace {
    #[inline]
    pub fn add_ray(&mut self) -> usize {
        if self.num_rays <= (NUM_RAYS - 1) as i32 {
            self.num_rays += 1;
            self.num_rays as usize
        } else {
            0
        }
    }
}
