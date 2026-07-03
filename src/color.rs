//! sRGB <-> linear <-> Oklab. All pipeline math happens in linear RGB or
//! Oklab (see PLAN.md bug risk #1). Oklab matrices from Björn Ottosson
//! (https://bottosson.github.io/posts/oklab/, public domain).

#![allow(clippy::excessive_precision)]

/// sRGB 8-bit channel -> linear [0,1].
pub fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear [0,1] -> sRGB 8-bit channel (clamped).
pub fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Linear RGB -> Oklab [L, a, b].
pub fn linear_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = 0.412_221_47 * r + 0.536_332_54 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_55 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_84 * g + 0.629_978_7 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Oklab [L, a, b] -> linear RGB (may be slightly out of gamut; clamp at encode).
pub fn oklab_to_linear(lab: [f32; 3]) -> [f32; 3] {
    let l_ = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
    let m_ = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
    let s_ = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// Squared distance in Oklab — the metric used for all palette matching.
pub fn oklab_dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Convenience: sRGB 8-bit triplet -> Oklab.
pub fn srgb8_to_oklab(r: u8, g: u8, b: u8) -> [f32; 3] {
    linear_to_oklab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
}

/// Convenience: Oklab -> sRGB 8-bit triplet.
pub fn oklab_to_srgb8(lab: [f32; 3]) -> [u8; 3] {
    let [r, g, b] = oklab_to_linear(lab);
    [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip_exact_for_all_bytes() {
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb(srgb_to_linear(v)), v, "byte {v}");
        }
    }

    #[test]
    fn oklab_white_black_reference_values() {
        // White (1,1,1) linear -> L≈1, a≈0, b≈0. Black -> all 0.
        let w = linear_to_oklab(1.0, 1.0, 1.0);
        assert!((w[0] - 1.0).abs() < 1e-3 && w[1].abs() < 1e-3 && w[2].abs() < 1e-3);
        let k = linear_to_oklab(0.0, 0.0, 0.0);
        assert!(k[0].abs() < 1e-6 && k[1].abs() < 1e-6 && k[2].abs() < 1e-6);
    }

    #[test]
    fn oklab_roundtrip_all_gray_and_sample_colors() {
        for &(r, g, b) in &[
            (255u8, 0u8, 0u8),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (255, 204, 170),
            (29, 43, 83),
        ] {
            let lab = srgb8_to_oklab(r, g, b);
            let back = oklab_to_srgb8(lab);
            // Allow 1 LSB of rounding noise per channel.
            assert!(
                (back[0] as i32 - r as i32).abs() <= 1
                    && (back[1] as i32 - g as i32).abs() <= 1
                    && (back[2] as i32 - b as i32).abs() <= 1,
                "({r},{g},{b}) -> {back:?}"
            );
        }
    }
}
