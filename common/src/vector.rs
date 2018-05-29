use math;

#[derive(Clone, Copy)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vector3 {
    #[inline]
    pub fn zero() -> Vector3 {
        Vector3::from_one(0.0)
    }
    #[inline]
    pub fn from_one(v: f32) -> Vector3 {
        Vector3 { x: v, y: v, z: v }
    }

    #[inline]
    pub fn length(self) -> f32 {
        math::sqrt(self.norm())
    }

    #[inline]
    pub fn norm(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z)
    }

    #[inline]
    pub fn normalize(self) -> Vector3 {
        let inv_len = self.length().recip();
        Vector3 {
            x: self.x * inv_len,
            y: self.y * inv_len,
            z: self.z * inv_len,
        }
    }

    #[inline]
    pub fn dot(self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn add(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn sub(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn mul_v(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    #[inline]
    pub fn mul_s(self, other: f32) -> Vector3 {
        Vector3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }

    #[inline]
    pub fn neg(self) -> Vector3 {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[derive(Clone)]
pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}
impl Ray {
    #[inline]
    pub fn create_prime(x: f32, y: f32, width: f32, height: f32, fov_adjustment: f32) -> Ray {
        let aspect_ratio = width / height;
        let sensor_x = ((((x + 0.5) / width) * 2.0 - 1.0) * aspect_ratio) * fov_adjustment;
        let sensor_y = (1.0 - ((y + 0.5) / height) * 2.0) * fov_adjustment;

        Ray {
            origin: Vector3::zero(),
            direction: Vector3 {
                x: sensor_x,
                y: sensor_y,
                z: -1.0,
            }.normalize(),
        }
    }
}
