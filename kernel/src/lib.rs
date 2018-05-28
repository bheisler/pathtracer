#![no_std]
#![feature(core_intrinsics)]

extern crate common;

use common::math::*;
use common::{Color, Material, Polygon, Ray, Vector3};
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
        let mut random_seed: u64 = ((x as u64) << 32) + y as u64;

        let ray = Ray::create_prime(
            x as f32,
            y as f32,
            width as f32,
            height as f32,
            fov_adjustment,
        );

        if let Some((distance, hit_poly)) = intersect_scene(&ray, polygon_count, polygons) {
            let closest_polygon: &Polygon = &*polygons.offset(hit_poly);

            let hit_normal = closest_polygon.normal;

            let material_idx = closest_polygon.material_idx as isize;
            let color = (&*materials.offset(material_idx)).color;
            *image.offset(i) = color;
        } else {
            *image.offset(i) = BLACK;
        }
    }
}

fn random_float(seed: &mut u64) -> f32 {
    // TODO: 1.0
}

fn create_scatter_direction(normal: &Normal) -> Vector3 {
    let r1 = random_float(&mut random_seed);
    let r2 = random_float(&mut random_seed);

    let y = r1;
    let azimuth = r2 * 2 * ::core::f32::consts::PI;
    let sin_elevation = sqrt(1.0 - y * y);
    let x = sin_elevation * cos(azimuth);
    let z = sin_elevation * sin(azimuth);

    let hemisphere_vec = Vector3 { x, y, z };

    let (n_t, n_b) = create_coordinate_system(&hit_normal);

    Vector3 {
        x: hemisphere_vec.x * n_b.x + hemisphere_vec.y * normal.x + hemisphere_vec.z * n_t.x,
        y: hemisphere_vec.x * n_b.y + hemisphere_vec.y * normal.y + hemisphere_vec.z * n_t.y,
        z: hemisphere_vec.x * n_b.z + hemisphere_vec.y * normal.z + hemisphere_vec.z * n_t.z,
    }
}

fn create_coordinate_system(normal: &Vector3) -> (Vector3, Vector3) {
    let n_t = if fabs(normal.x) > fabs(normal.y) {
        Vector3 {
            x: normal.z,
            y: 0.0,
            z: -normal.x,
        }.normalize()
    } else {
        Vector3 {
            x: 0.0,
            y: -normal.z,
            z: normal.y,
        }.normalize()
    };
    let n_b = normal.cross(n_t);
    (n_t, n_b)
}

unsafe fn intersect_scene(
    ray: &Ray,
    polygon_count: usize,
    polygons: *const Polygon,
) -> Option<(f32, isize)> {
    let mut polygon_i = 0;
    let mut closest_distance = 10000000.0;
    let mut closest_i: isize = -1;
    while polygon_i < polygon_count {
        let polygon: &Polygon = &*polygons.offset(polygon_i as isize);
        let maybe_hit = intersection_test(polygon, ray);

        if let Some(distance) = maybe_hit {
            if distance < closest_distance {
                closest_distance = distance;
                closest_i = polygon_i as isize;
            }
        }

        polygon_i += 1;
    }
    if closest_i != -1 {
        Some((closest_distance, closest_i))
    } else {
        None
    }
}

const EPSILON: f32 = 0.00001;

fn intersection_test(polygon: &Polygon, ray: &Ray) -> Option<f32> {
    // Step 1: Find P (intersection between triangle plane and ray)

    let n = polygon.normal;

    let n_dot_r = n.dot(ray.direction);
    if fabs(n_dot_r) < EPSILON {
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
