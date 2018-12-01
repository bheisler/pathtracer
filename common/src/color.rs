use math;

pub const BLACK: Color = Color {
    red: 0.0,
    green: 0.0,
    blue: 0.0,
};

pub const WHITE: Color = Color {
    red: 1.0,
    green: 1.0,
    blue: 1.0,
};

#[derive(Copy, Clone, DeviceCopy)]
pub struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}
impl Color {
    #[inline]
    pub fn add(self, other: Color) -> Color {
        Color {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }

    #[inline]
    pub fn mul(self, other: Color) -> Color {
        Color {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
        }
    }

    #[inline]
    pub fn mul_s(self, other: f32) -> Color {
        Color {
            red: self.red * other,
            green: self.green * other,
            blue: self.blue * other,
        }
    }

    #[inline]
    pub fn clamp(self) -> Color {
        Color {
            red: math::clamp(0.0, self.red, 1.0),
            green: math::clamp(0.0, self.green, 1.0),
            blue: math::clamp(0.0, self.blue, 1.0),
        }
    }
}
