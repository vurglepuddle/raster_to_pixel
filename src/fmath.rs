//! `f32` math that gives the same bits on every platform.
//!
//! `f32::powf` and friends call the platform's libm, and those are not correctly
//! rounded: the Windows and WebAssembly libms disagree on up to a third of inputs for
//! some functions, which is enough to send k-means down a different path and change a
//! palette. Computing in `f64` and rounding back absorbs those last-bit errors, so the
//! CLI, native GUI, and browser build all produce the same output for the same pixels.
//! The results were checked bit-identical across Windows and wasm32 over 2M inputs per
//! function (see web-wasm/README.md). All pipeline math must go through here.

pub fn powf(x: f32, exponent: f32) -> f32 {
    (x as f64).powf(exponent as f64) as f32
}

pub fn cbrt(x: f32) -> f32 {
    (x as f64).cbrt() as f32
}

pub fn exp(x: f32) -> f32 {
    (x as f64).exp() as f32
}

pub fn atan2(y: f32, x: f32) -> f32 {
    (y as f64).atan2(x as f64) as f32
}

pub fn sin_cos(x: f32) -> (f32, f32) {
    let (sin, cos) = (x as f64).sin_cos();
    (sin as f32, cos as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Within two ULPs of std: the same functions, only rounded consistently.
    fn close(a: f32, b: f32) -> bool {
        a == b || (a.to_bits() as i64 - b.to_bits() as i64).abs() <= 2
    }

    #[test]
    fn matches_std_functions_apart_from_last_bit_rounding() {
        for i in 1..20_000 {
            let x = i as f32 / 20_000.0;
            assert!(close(powf(x, 2.4), x.powf(2.4)), "powf({x})");
            assert!(close(cbrt(x), x.cbrt()), "cbrt({x})");
            assert!(close(exp(-x * 11.0), (-x * 11.0).exp()), "exp({x})");
            assert!(close(atan2(x, 0.37), x.atan2(0.37)), "atan2({x})");
            let (s, c) = sin_cos(x);
            assert!(
                (s - x.sin()).abs() < 1e-6 && (c - x.cos()).abs() < 1e-6,
                "sin_cos({x})"
            );
        }
    }

    #[test]
    fn negative_cube_roots_stay_real() {
        // Slightly out-of-gamut linear values reach cbrt with a negative sign.
        assert_eq!(cbrt(-8.0), -2.0);
    }
}
