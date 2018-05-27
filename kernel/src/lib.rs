#![no_std]
#![feature(core_intrinsics)]

extern crate common;

use common::{Color, Material, Polygon, Ray};
use core::intrinsics;

const BLACK: Color = Color {
    red: 0.0,
    green: 0.0,
    blue: 0.0,
};

#[no_mangle]
pub unsafe fn trace_inner(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    fov_adjustment: f32,
    image: *mut Color,
    polygons: *const Polygon,
    polygon_count: usize,
    materials: *const Material,
    material_count: usize,
) {
    let i = (y * width + x) as isize;
    if x < width && y < height {
        let ray = Ray::create_prime(
            x as f32,
            y as f32,
            width as f32,
            height as f32,
            fov_adjustment,
        );

        let mut polygon_i = 0;
        let mut closest_distance = 10000000.0;
        let mut closest_i: isize = -1;
        while (polygon_i < polygon_count) {
            let polygon: &Polygon = &*polygons.offset(polygon_i as isize);
            let maybe_hit = intersection_test(polygon, &ray);

            if let Some(distance) = maybe_hit {
                if (distance < closest_distance) {
                    closest_distance = distance;
                    closest_i = polygon_i as isize;
                }
            }

            polygon_i += 1;
        }

        if closest_i != -1 {
            let closest_polygon: &Polygon = &*polygons.offset(closest_i);
            let material_idx = closest_polygon.material_idx as isize;
            let color = (&*materials.offset(material_idx)).color;
            *image.offset(i) = color;
        } else {
            *image.offset(i) = BLACK;
        }
    }
}

const EPSILON: f32 = 0.00001;

fn intersection_test(polygon: &Polygon, ray: &Ray) -> Option<f32> {
    // Step 1: Find P (intersection between triangle plane and ray)

    let n = polygon.normal;

    let n_dot_r = n.dot(ray.direction);
    if unsafe { intrinsics::fabsf32(n_dot_r) } < EPSILON {
        // The ray is parallel to the triangle. No intersection.
        return None;
    }

    // Compute -D
    let neg_d = n.dot(polygon.vertices[0]);

    // Compute T
    let t = (neg_d - ray.origin.dot(n)) / n_dot_r;
    if t < 0.0 {
        // Triangle is behind the origin of the ray. No intersection.
        return None;
    }

    // Calculate P
    let p = ray.origin.add(ray.direction.mul_s(t));

    // Step 2: is P in the triangle?

    // Is P left of the first edge?
    let edge = polygon.vertices[1].sub(polygon.vertices[0]);
    let vp = p.sub(polygon.vertices[0]);
    let c = edge.cross(vp);
    if n.dot(c) < 0.0 {
        return None;
    } // P is right of the edge. No intersection.

    // Repeat for edges 2 and 3

    let edge = polygon.vertices[2].sub(polygon.vertices[1]);
    let vp = p.sub(polygon.vertices[1]);
    let c = edge.cross(vp);
    if n.dot(c) < 0.0 {
        return None;
    }

    let edge = polygon.vertices[0].sub(polygon.vertices[2]);
    let vp = p.sub(polygon.vertices[2]);
    let c = edge.cross(vp);
    if n.dot(c) < 0.0 {
        return None;
    }

    // Finally, we've confirmed an intersection.
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{Polygon, Ray, Vector3};

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_polygon_intersection() {
        let polygon = Polygon {
            vertices: [
                Vector3 { x: -1.0, y: 1.0, z: -1.0 },
                Vector3 { x: -1.0, y: -1.0, z: -1.0 },
                Vector3 { x: 1.0, y: -1.0, z: -1.0 },
            ],
            normal: Vector3 { x: 0.0, y: 0.0, z: 1.0 },
            material_idx: 0,
        };

        let ray = Ray {
            origin: Vector3 { x: -0.5, y: 0.0, z: 0.0 },
            direction: Vector3 { x: 0.0, y: 0.0, z: -1.0 },
        };

        let intersection = intersection_test(&polygon, &ray).unwrap();
        assert_eq!(intersection, 1.0);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_polygon_intersection_offset() {
        let polygon = Polygon {
            vertices: [
                Vector3 { x: -1.0, y: 1.0, z: -2.0 },
                Vector3 { x: -1.0, y: -1.0, z: -2.0 },
                Vector3 { x: 1.0, y: -1.0, z: -2.0 },
            ],
            normal: Vector3 { x: 0.0, y: 0.0, z: 1.0 },
            material_idx: 0,
        };

        let ray = Ray {
            origin: Vector3 { x: -0.5, y: 0.0, z: -1.0 },
            direction: Vector3 { x: 0.0, y: 0.0, z: -1.0 },
        };

        let intersection = intersection_test(&polygon, &ray).unwrap();
        assert_eq!(intersection, 1.0);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_polygon_intersection_reverse() {
        let polygon = Polygon {
            vertices: [
                Vector3 { x: -1.0, y: 1.0, z: -1.0 },
                Vector3 { x: 1.0, y: -1.0, z: -1.0 },
                Vector3 { x: -1.0, y: -1.0, z: -1.0 },
            ],
            normal: Vector3 { x: 0.0, y: 0.0, z: -1.0 },
            material_idx: 0,
        };

        let ray = Ray {
            origin: Vector3 { x: -0.5, y: 0.0, z: -2.0 },
            direction: Vector3 { x: 0.0, y: 0.0, z: 1.0 },
        };

        let intersection = intersection_test(&polygon, &ray).unwrap();
        assert_eq!(intersection, 1.0);
    }

}
