#![no_std]

extern crate common;

use common::math::*;
use common::{
    BoundingBox, Color, Material, Object, Polygon, Ray, ScratchSpace, Vector3, BLACK, WHITE,
};
use core::mem::swap;

const BOUNCE_CAP: u32 = 8;
const FLOATING_POINT_BACKOFF: f32 = 0.01;
const LIGHT_POWER_FUDGE_FACTOR: f32 = 1.25;

// For each block of the image, we trace RAY_COUNT rays, and we trace over each block ROUND_COUNT times.
// This makes it possible to take many more samples than we can fit into the 3-second window.
// This has to be tuned based on the complexity of the scene.
pub const ROUND_COUNT: u32 = 64;
const RAY_COUNT: u32 = 32;

const RANDOM_SEED: u32 = 0x8802dfb5;

struct Stats {
    rays_traced: u64,
    triangle_intersections: u64,
    bounding_box_intersections: u64,
}

#[no_mangle]
pub unsafe fn trace_inner(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    round: u32,
    fov_adjustment: f32,
    image: *mut Color,
    polygons: *const Polygon,
    objects: *const Object,
    object_count: usize,
    scratch_space: &mut ScratchSpace,
) {
    let mut stats = Stats {
        rays_traced: 0,
        triangle_intersections: 0,
        bounding_box_intersections: 0,
    };

    let i = (y * width + x) as isize;
    if x < width && y < height {
        let mut random_seed: u32 = RANDOM_SEED
            ^ ((x << 16)
                .wrapping_add(((*objects.offset(object_count as isize)).polygon_end as u32) << 12)
                .wrapping_add(width << 23)
                .wrapping_add(height << 28)
                .wrapping_add(round << 5)
                .wrapping_add(y));

        let mut skip_i = 0;
        while skip_i < 10 {
            random_float(&mut random_seed);
            skip_i += 1;
        }

        let mut color_accumulator = *image.offset(i);
        let mut ray_num = 0;
        while ray_num < RAY_COUNT {
            color_accumulator = color_accumulator.add(
                get_radiance(
                    x,
                    y,
                    width,
                    height,
                    fov_adjustment,
                    polygons,
                    objects,
                    object_count,
                    &mut random_seed,
                    scratch_space,
                    &mut stats,
                ).mul_s(1.0 / (RAY_COUNT * ROUND_COUNT) as f32),
            );
            ray_num += 1;
        }

        *image.offset(i) = color_accumulator;
        scratch_space.rays_traced += stats.rays_traced;
        scratch_space.bounding_box_intersections += stats.bounding_box_intersections;
        scratch_space.triangle_intersections += stats.triangle_intersections;
    }
}

unsafe fn get_radiance(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    fov_adjustment: f32,
    polygons: *const Polygon,
    objects: *const Object,
    object_count: usize,
    random_seed: &mut u32,
    scratch_space: &mut ScratchSpace,
    stats: &mut Stats,
) -> Color {
    let mut color_accumulator = BLACK;

    scratch_space.num_rays = 0;
    scratch_space.rays[0] = create_prime_ray(
        x as f32,
        y as f32,
        width as f32,
        height as f32,
        fov_adjustment,
        random_seed,
    );
    scratch_space.masks[0] = WHITE;

    let mut bounce_i = 0;
    while bounce_i < BOUNCE_CAP {
        let mut ray_i = scratch_space.num_rays;
        while ray_i >= 0 {
            let ray_u = ray_i as usize;
            let mut current_ray = scratch_space.rays.get_unchecked(ray_u).clone();
            let mut color_mask = scratch_space.masks.get_unchecked(ray_u).clone();

            if let Some((distance, hit_object, hit_normal)) =
                intersect_scene(&current_ray, polygons, objects, object_count, stats)
            {
                let hit_point = current_ray
                    .origin
                    .add(current_ray.direction.mul_s(distance));
                // Back off along the hit normal a bit to avoid floating-point problems.
                current_ray.origin = hit_point.add(hit_normal.mul_s(FLOATING_POINT_BACKOFF));

                let material: Material = (&*objects.offset(hit_object)).material.clone();

                match material {
                    Material::Diffuse { color, albedo } => {
                        let (direction, weight) =
                            create_scatter_direction(&hit_normal, random_seed);
                        current_ray.direction = direction;
                        // Lighting = emission + (incident_light * color * incident_direction dot normal * albedo * PI * weight)
                        let cosine_angle = current_ray.direction.dot(hit_normal);
                        let reflected_power = albedo * ::core::f32::consts::PI;
                        let reflected_color = color
                            .mul_s(cosine_angle)
                            .mul_s(reflected_power)
                            .mul_s(weight);

                        color_mask = color_mask
                            .mul(reflected_color)
                            .mul_s(LIGHT_POWER_FUDGE_FACTOR);
                    }
                    Material::Emissive { emission } => {
                        let (direction, _) = create_scatter_direction(&hit_normal, random_seed);
                        current_ray.direction = direction;
                        color_accumulator = color_accumulator.add(emission.mul(color_mask));
                        // Leave the color mask as-is, I guess?
                    }
                    Material::Reflective => {
                        current_ray.direction = make_reflection(current_ray.direction, hit_normal);
                    }
                    Material::Refractive { index } => {
                        let kr = fresnel(current_ray.direction, hit_normal, index);

                        if kr < 1.0 {
                            // Add the transmission ray
                            if let Some(new_ray) = scratch_space.add_ray() {
                                *scratch_space.rays.get_unchecked_mut(new_ray) = Ray {
                                    direction: make_transmission(
                                        hit_normal,
                                        current_ray.direction,
                                        index,
                                    ),
                                    origin: hit_point
                                        .add(hit_normal.mul_s(-FLOATING_POINT_BACKOFF)),
                                };
                                *scratch_space.masks.get_unchecked_mut(new_ray) =
                                    color_mask.mul_s((1.0 - kr) * LIGHT_POWER_FUDGE_FACTOR);
                            }
                        }

                        // The current ray becomes the reflection ray
                        current_ray.direction = make_reflection(current_ray.direction, hit_normal);
                        color_mask = color_mask.mul_s(kr * LIGHT_POWER_FUDGE_FACTOR);
                    }
                }
            } else {
                color_mask = BLACK;
            }

            *scratch_space.rays.get_unchecked_mut(ray_u) = current_ray;
            *scratch_space.masks.get_unchecked_mut(ray_u) = color_mask;

            ray_i = ray_i - 1;
        }
        bounce_i += 1;
    }
    return color_accumulator;
}

fn create_prime_ray(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    fov_adjustment: f32,
    seed: &mut u32,
) -> Ray {
    let aspect_ratio = width / height;
    let sensor_x =
        ((((x + random_float(seed)) / width) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
    let sensor_y = (1.0 - ((y + random_float(seed)) / height) * 2.0) * fov_adjustment;

    Ray {
        origin: Vector3::zero(),
        direction: Vector3 {
            x: sensor_x,
            y: sensor_y,
            z: -1.0,
        }.normalize(),
    }
}

fn make_reflection(incident: Vector3, normal: Vector3) -> Vector3 {
    incident.sub(normal.mul_s(2.0 * incident.dot(normal)))
}

pub fn make_transmission(normal: Vector3, incident: Vector3, index: f32) -> Vector3 {
    let mut ref_n = normal;
    let mut eta_t = index;
    let mut eta_i = 1.0;
    let mut i_dot_n = incident.dot(normal);
    if i_dot_n < 0.0 {
        //Outside the surface
        i_dot_n = -i_dot_n;
    } else {
        //Inside the surface; invert the normal and swap the indices of refraction
        ref_n = normal.neg();
        eta_i = eta_t;
        eta_t = 1.0;
    }

    let eta = eta_i / eta_t;
    let k = 1.0 - (eta * eta) * (1.0 - i_dot_n * i_dot_n);
    incident
        .add(ref_n.mul_s(i_dot_n))
        .mul_s(eta)
        .sub(ref_n.mul_s(sqrt(k)))
}

fn fresnel(incident: Vector3, normal: Vector3, index: f32) -> f32 {
    let i_dot_n = incident.dot(normal);
    let mut eta_i = 1.0;
    let mut eta_t = index as f32;
    if i_dot_n > 0.0 {
        eta_i = eta_t;
        eta_t = 1.0;
    }

    let sin_t = eta_i / eta_t * sqrt(max(1.0 - i_dot_n * i_dot_n, 0.0));
    if sin_t > 1.0 {
        //Total internal reflection
        return 1.0;
    } else {
        let cos_t = sqrt(max(1.0 - sin_t * sin_t, 0.0));
        let cos_i = fabs(cos_t);
        let r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
        let r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
        return (r_s * r_s + r_p * r_p) / 2.0;
    }
}

/// Generates a random floating-point number in the range [0.0, 1.0] using an xorshift32 pRNG and
/// some evil floating-point bit-level hacking. We take the high 32 bits of the output from the
/// pRNG, mask off the low 23 bits to use as a random mantissa and set the appropriate sign and
/// exponent bits to turn that mantissa into a floating point value in the [1.0, 2.0] range, then
/// subtract 1.0 (thus avoiding having to deal with denormals and similar things).
///
/// Non-rigorous eyeball checking suggests that the output is at least approximately uniform.
fn random_float(seed: &mut u32) -> f32 {
    let mut x = *seed;
    x ^= x >> 13;
    x ^= x << 17;
    x ^= x >> 5;
    *seed = x;
    let float_bits = (x & 0x007FFFFF) | 0x3F800000;
    let float: f32 = unsafe { ::core::mem::transmute(float_bits) };
    return float - 1.0;
}

fn create_scatter_direction(normal: &Vector3, random_seed: &mut u32) -> (Vector3, f32) {
    let r1 = random_float(random_seed);
    let r2 = random_float(random_seed);

    let y = r1;
    let azimuth = r2 * 2.0 * ::core::f32::consts::PI;
    let sin_elevation = sqrt(1.0 - y * y);
    let x = sin_elevation * cos(azimuth);
    let z = sin_elevation * sin(azimuth);

    let hemisphere_vec = Vector3 { x, y, z };

    let (n_t, n_b) = create_coordinate_system(normal);

    let scatter = Vector3 {
        x: hemisphere_vec.x * n_b.x + hemisphere_vec.y * normal.x + hemisphere_vec.z * n_t.x,
        y: hemisphere_vec.x * n_b.y + hemisphere_vec.y * normal.y + hemisphere_vec.z * n_t.y,
        z: hemisphere_vec.x * n_b.z + hemisphere_vec.y * normal.z + hemisphere_vec.z * n_t.z,
    };
    let weight = 1.0 / scatter.dot(*normal);
    (scatter, weight)
}

fn create_coordinate_system(normal: &Vector3) -> (Vector3, Vector3) {
    let n_t = if fabs(normal.x) > fabs(normal.y) {
        Vector3 {
            x: normal.z,
            y: 0.0,
            z: -normal.x,
        }
    } else {
        Vector3 {
            x: 0.0,
            y: -normal.z,
            z: normal.y,
        }
    };
    let n_t = n_t.normalize();
    let n_b = normal.cross(n_t);
    (n_t, n_b)
}

unsafe fn intersect_scene(
    ray: &Ray,
    polygons: *const Polygon,
    objects: *const Object,
    object_count: usize,
    stats: &mut Stats,
) -> Option<(f32, isize, Vector3)> {
    stats.rays_traced += 1;

    let mut closest_distance = 10000000.0;
    let mut closest_obj_i: isize = -1;
    let mut closest_normal = Vector3::zero();

    let mut object_i = 0;
    while object_i < object_count {
        let object = &*objects.offset(object_i as isize);
        stats.bounding_box_intersections += 1;
        if let Some(t) = test_bounding_box(&object.bounding_box, ray) {
            let hit_point = ray.origin
                .add(ray.direction.mul_s(t + FLOATING_POINT_BACKOFF));

            if let Some((distance, normal)) = grid_march(&object, polygons, &hit_point, &ray, stats)
            {
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_normal = normal;
                    closest_obj_i = object_i as isize;
                }
            }
        }

        object_i += 1;
    }

    if closest_obj_i != -1 {
        Some((closest_distance, closest_obj_i, closest_normal))
    } else {
        None
    }
}

unsafe fn grid_march(
    object: &Object,
    polygons: *const Polygon,
    start: &Vector3,
    ray: &Ray,
    stats: &mut Stats,
) -> Option<(f32, Vector3)> {
    let start2 = Vector3 {
        x: start.x - object.bounding_box.min_x,
        y: start.y - object.bounding_box.min_y,
        z: start.z - object.bounding_box.min_z,
    };

    let mut cur_x = floor(start2.x / object.grid.cell_x) as i32;
    let mut cur_y = floor(start2.y / object.grid.cell_y) as i32;
    let mut cur_z = floor(start2.z / object.grid.cell_z) as i32;

    let step_x = if ray.direction.x > 0.0 { 1 } else { -1 };
    let step_y = if ray.direction.y > 0.0 { 1 } else { -1 };
    let step_z = if ray.direction.z > 0.0 { 1 } else { -1 };

    fn voxel_side(k: i32, step_k: i32) -> f32 {
        if step_k < 0 {
            k as f32
        } else {
            (k + step_k) as f32
        }
    }

    let mut t_max_x = (voxel_side(cur_x, step_x) * object.grid.cell_x - start2.x) / ray.direction.x;
    let mut t_max_y = (voxel_side(cur_y, step_y) * object.grid.cell_y - start2.y) / ray.direction.y;
    let mut t_max_z = (voxel_side(cur_z, step_z) * object.grid.cell_z - start2.z) / ray.direction.z;

    let t_delta_x = (step_x as f32) * (object.grid.cell_x / ray.direction.x);
    let t_delta_y = (step_y as f32) * (object.grid.cell_y / ray.direction.y);
    let t_delta_z = (step_z as f32) * (object.grid.cell_z / ray.direction.z);

    let mut closest_t = 10000000.0;
    let mut closest_normal = Vector3::zero();
    while cur_x >= 0 && cur_x < object.grid.n_x && cur_y >= 0 && cur_y < object.grid.n_y
        && cur_z >= 0 && cur_z < object.grid.n_z
    {
        let i = (cur_z * object.grid.n_y * object.grid.n_x) + (cur_y * object.grid.n_x) + cur_x;
        let mut index_i = (*object.grid.index_ranges.offset(i as isize)).start;
        let index_stop = (*object.grid.index_ranges.offset(i as isize)).stop;

        let mut t_min = min(t_max_x, min(t_max_y, t_max_z));

        while index_i < index_stop {
            let polygon_i = *object.grid.polygon_indexes.offset(index_i);
            let polygon = &*polygons.offset(polygon_i);
            stats.triangle_intersections += 1;
            if let Some((distance, normal)) = intersection_test(polygon, ray) {
                if distance < closest_t && distance < t_min {
                    closest_t = distance;
                    closest_normal = normal;
                }
            }
            index_i += 1;
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                cur_x += step_x;
                t_max_x += t_delta_x;
            } else {
                cur_z += step_z;
                t_max_z += t_delta_z;
            }
        } else {
            if t_max_y < t_max_z {
                cur_y += step_y;
                t_max_y += t_delta_y;
            } else {
                cur_z += step_z;
                t_max_z += t_delta_z;
            }
        }
    }

    if closest_t != 10000000.0 {
        Some((closest_t, closest_normal))
    } else {
        None
    }
}

const EPSILON: f32 = 0.00001;

fn test_bounding_box(bounding_box: &BoundingBox, ray: &Ray) -> Option<f32> {
    let dir_inv = ray.direction.inverse();

    let mut tmin = (bounding_box.min_x - ray.origin.x) * dir_inv.x;
    let mut tmax = (bounding_box.max_x - ray.origin.x) * dir_inv.x;

    if tmin > tmax {
        swap(&mut tmin, &mut tmax);
    }

    let mut tymin = (bounding_box.min_y - ray.origin.y) * dir_inv.y;
    let mut tymax = (bounding_box.max_y - ray.origin.y) * dir_inv.y;

    if tymin > tymax {
        swap(&mut tymin, &mut tymax);
    }

    if tmin > tymax || tymin > tmax {
        return None;
    }

    tmin = max(tmin, tymin);
    tmax = min(tymax, tmax);

    let mut tzmin = (bounding_box.min_z - ray.origin.z) * dir_inv.z;
    let mut tzmax = (bounding_box.max_z - ray.origin.z) * dir_inv.z;

    if tzmin > tzmax {
        swap(&mut tzmin, &mut tzmax);
    }

    if tmin > tzmax || tzmin > tmax {
        return None;
    }

    tmin = max(tmin, tzmin);
    tmax = min(tzmax, tmax);

    let t = if tmin < 0.0 { 0.0 } else { tmin };
    if t < 0.0 {
        return None;
    }

    Some(t)
}

fn intersection_test(polygon: &Polygon, ray: &Ray) -> Option<(f32, Vector3)> {
    let v0 = polygon.vertices[0];
    let edge1 = polygon.vertices[1].sub(v0);
    let edge2 = polygon.vertices[2].sub(v0);
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);
    if fabs(a) < EPSILON {
        return None;
    }
    let f = 1.0 / a;
    let s = ray.origin.sub(v0);
    let u = f * (s.dot(h));
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t > EPSILON {
        return Some((t, edge1.cross(edge2).normalize()));
    }
    return None;
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

    #[test]
    fn test_rng() {
        let mut seed = 0xDEADBEEFu64;
        for x in (0..1000000) {
            let rand = random_float(&mut seed);
            assert!(rand >= 0.0);
            assert!(rand <= 1.0);
        }
    }
}
