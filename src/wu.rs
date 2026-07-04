//! Wu color quantization (Xiaolin Wu, "Color Quantization by Dynamic
//! Programming and Principal Analysis", 1992), adapted to run on an Oklab
//! lattice instead of the classical RGB cube so its variance-maximizing
//! splits happen in the same perceptual space every other palette decision
//! in this project uses.
//!
//! The algorithm: accumulate a 32³ histogram of moments (weight, per-channel
//! sums, sum of squares) over normalized Oklab coordinates, turn the arrays
//! into 3D summed-area tables, then greedily split the box with the largest
//! variance at the cut that maximizes between-class variance. Every box's
//! representative is the true mean of its samples (moments store real
//! coordinates, not bin centers), so quality does not degrade to lattice
//! resolution. Fully deterministic: no RNG, fixed tie-breaks.

use crate::color::oklab_dist2;

const NBINS: usize = 32;
/// Histogram side: bins 1..=NBINS, index 0 is zero padding for the SAT.
const HS: usize = NBINS + 1;

#[inline]
fn at(r: usize, g: usize, b: usize) -> usize {
    (r * HS + g) * HS + b
}

/// Normalize Oklab to the unit cube. L is already 0..1 for sRGB inputs;
/// a/b sit within ±0.32 for the sRGB gamut, so +0.5 centers them safely.
fn normalize(lab: [f32; 3]) -> [f64; 3] {
    [
        (lab[0] as f64).clamp(0.0, 1.0),
        (lab[1] as f64 + 0.5).clamp(0.0, 1.0),
        (lab[2] as f64 + 0.5).clamp(0.0, 1.0),
    ]
}

fn bin(v: f64) -> usize {
    ((v * NBINS as f64) as usize).min(NBINS - 1) + 1
}

struct Moments {
    w: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    m2: Vec<f64>,
}

impl Moments {
    fn build(samples: &[[f32; 3]]) -> Self {
        let n = HS * HS * HS;
        let mut m = Moments {
            w: vec![0.0; n],
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            m2: vec![0.0; n],
        };
        for s in samples {
            let p = normalize(*s);
            let i = at(bin(p[0]), bin(p[1]), bin(p[2]));
            m.w[i] += 1.0;
            m.x[i] += p[0];
            m.y[i] += p[1];
            m.z[i] += p[2];
            m.m2[i] += p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        }
        for arr in [&mut m.w, &mut m.x, &mut m.y, &mut m.z, &mut m.m2] {
            cumulate(arr);
        }
        m
    }
}

/// Turn a histogram into a 3D summed-area table (prefix sums per axis).
fn cumulate(m: &mut [f64]) {
    for r in 0..HS {
        for g in 0..HS {
            for b in 1..HS {
                m[at(r, g, b)] += m[at(r, g, b - 1)];
            }
        }
    }
    for r in 0..HS {
        for g in 1..HS {
            for b in 0..HS {
                m[at(r, g, b)] += m[at(r, g - 1, b)];
            }
        }
    }
    for r in 1..HS {
        for g in 0..HS {
            for b in 0..HS {
                m[at(r, g, b)] += m[at(r - 1, g, b)];
            }
        }
    }
}

/// A box over histogram bins; lower bounds exclusive, upper inclusive,
/// all in 0..=NBINS (the classical Wu convention for SAT lookups).
#[derive(Clone, Copy, Debug)]
struct Bx {
    r0: usize,
    r1: usize,
    g0: usize,
    g1: usize,
    b0: usize,
    b1: usize,
}

impl Bx {
    fn full() -> Self {
        Bx {
            r0: 0,
            r1: NBINS,
            g0: 0,
            g1: NBINS,
            b0: 0,
            b1: NBINS,
        }
    }

    fn bounds(&self, axis: usize) -> (usize, usize) {
        match axis {
            0 => (self.r0, self.r1),
            1 => (self.g0, self.g1),
            _ => (self.b0, self.b1),
        }
    }

    fn with_upper(mut self, axis: usize, v: usize) -> Self {
        match axis {
            0 => self.r1 = v,
            1 => self.g1 = v,
            _ => self.b1 = v,
        }
        self
    }

    fn with_lower(mut self, axis: usize, v: usize) -> Self {
        match axis {
            0 => self.r0 = v,
            1 => self.g0 = v,
            _ => self.b0 = v,
        }
        self
    }
}

/// Sum of one SAT over a box via 8-corner inclusion–exclusion.
fn vol(m: &[f64], bx: Bx) -> f64 {
    m[at(bx.r1, bx.g1, bx.b1)] - m[at(bx.r1, bx.g1, bx.b0)] - m[at(bx.r1, bx.g0, bx.b1)]
        + m[at(bx.r1, bx.g0, bx.b0)]
        - m[at(bx.r0, bx.g1, bx.b1)]
        + m[at(bx.r0, bx.g1, bx.b0)]
        + m[at(bx.r0, bx.g0, bx.b1)]
        - m[at(bx.r0, bx.g0, bx.b0)]
}

/// (weight, sum x, sum y, sum z) of a box.
fn sums(m: &Moments, bx: Bx) -> [f64; 4] {
    [vol(&m.w, bx), vol(&m.x, bx), vol(&m.y, bx), vol(&m.z, bx)]
}

/// Weighted within-box variance: E|p|² - |E p|² · w.
fn variance(m: &Moments, bx: Bx) -> f64 {
    let s = sums(m, bx);
    if s[0] <= 0.0 {
        return 0.0;
    }
    (vol(&m.m2, bx) - (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]) / s[0]).max(0.0)
}

/// Best (axis, cut) for a box, maximizing between-class variance
/// Σ |m₁|²/w over the two halves. Deterministic: first maximum wins
/// (axes in L, a, b order; cut positions ascending).
fn best_cut(m: &Moments, bx: Bx) -> Option<(usize, usize)> {
    let whole = sums(m, bx);
    if whole[0] <= 0.0 {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for axis in 0..3 {
        let (lo, hi) = bx.bounds(axis);
        for cut in lo + 1..hi {
            let b = sums(m, bx.with_upper(axis, cut));
            let bw = b[0];
            let tw = whole[0] - bw;
            if bw <= 0.0 || tw <= 0.0 {
                continue;
            }
            let (tx, ty, tz) = (whole[1] - b[1], whole[2] - b[2], whole[3] - b[3]);
            let score =
                (b[1] * b[1] + b[2] * b[2] + b[3] * b[3]) / bw + (tx * tx + ty * ty + tz * tz) / tw;
            if score > best_score {
                best_score = score;
                best = Some((axis, cut));
            }
        }
    }
    best
}

/// Build a k-color Oklab palette with Wu quantization. Matches
/// `kmeans::build_palette`'s contract: luma-sorted, deduped, never empty.
pub fn build_palette_wu(samples: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    assert!(k > 0, "k must be >= 1");
    if samples.is_empty() {
        return vec![[0.0; 3]; 1];
    }
    let m = Moments::build(samples);
    let mut boxes = vec![Bx::full()];
    let mut variances = vec![variance(&m, boxes[0])];

    while boxes.len() < k {
        // Split the box with the largest variance; first index wins ties.
        let mut pick = None;
        let mut pick_var = 1e-12f64;
        for (i, &v) in variances.iter().enumerate() {
            if v > pick_var {
                pick_var = v;
                pick = Some(i);
            }
        }
        let Some(i) = pick else { break };
        let Some((axis, cut)) = best_cut(&m, boxes[i]) else {
            variances[i] = 0.0; // Unsplittable (single occupied bin).
            continue;
        };
        let bottom = boxes[i].with_upper(axis, cut);
        let top = boxes[i].with_lower(axis, cut);
        boxes[i] = bottom;
        variances[i] = variance(&m, bottom);
        boxes.push(top);
        variances.push(variance(&m, top));
    }

    let mut centers: Vec<[f32; 3]> = boxes
        .iter()
        .filter_map(|&bx| {
            let s = sums(&m, bx);
            if s[0] <= 0.0 {
                return None;
            }
            Some([
                (s[1] / s[0]) as f32,
                (s[2] / s[0] - 0.5) as f32,
                (s[3] / s[0] - 0.5) as f32,
            ])
        })
        .collect();

    centers.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    centers.dedup_by(|a, b| oklab_dist2(*a, *b) < 1e-12);
    if centers.is_empty() {
        centers.push([0.0; 3]);
    }
    centers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{oklab_to_srgb8, srgb8_to_oklab};

    #[test]
    fn two_obvious_clusters_are_found() {
        let mut samples = Vec::new();
        for i in 0..50 {
            let j = i as f32 * 1e-4;
            samples.push([0.1 + j, 0.0, 0.0]);
            samples.push([0.9 + j, 0.0, 0.0]);
        }
        let p = build_palette_wu(&samples, 2);
        assert_eq!(p.len(), 2);
        assert!((p[0][0] - 0.1025).abs() < 0.01, "{p:?}");
        assert!((p[1][0] - 0.9025).abs() < 0.01, "{p:?}");
        assert!(p[0][0] < p[1][0], "luma-sorted");
    }

    #[test]
    fn recovers_exact_colors_when_k_matches_well_separated_inputs() {
        let colors = [
            srgb8_to_oklab(200, 30, 30),
            srgb8_to_oklab(30, 200, 30),
            srgb8_to_oklab(30, 30, 200),
            srgb8_to_oklab(240, 240, 240),
        ];
        let mut samples = Vec::new();
        for (i, c) in colors.iter().enumerate() {
            for _ in 0..(10 + i) {
                samples.push(*c);
            }
        }
        let p = build_palette_wu(&samples, 4);
        assert_eq!(p.len(), 4);
        for c in &colors {
            let hit = p.iter().any(|e| oklab_dist2(*e, *c) < 1e-6);
            assert!(hit, "missing {:?} in {:?}", oklab_to_srgb8(*c), p);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let samples: Vec<[f32; 3]> = (0..500)
            .map(|i| {
                let t = i as f32 / 500.0;
                [t, (t * 7.0).sin() * 0.1, (t * 13.0).cos() * 0.1]
            })
            .collect();
        let a = build_palette_wu(&samples, 16);
        let b = build_palette_wu(&samples, 16);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn fewer_unique_colors_than_k_does_not_pad_duplicates() {
        let samples = vec![[0.2, 0.0, 0.0]; 10];
        let p = build_palette_wu(&samples, 4);
        assert_eq!(p.len(), 1);
        assert!((p[0][0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn k_one_returns_the_overall_mean() {
        let samples = vec![[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]];
        let p = build_palette_wu(&samples, 1);
        assert_eq!(p.len(), 1);
        assert!((p[0][0] - 0.3).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn dominant_population_gets_the_finer_split() {
        // 90% of samples form two nearby dark clusters, 10% one bright one.
        // With k=3, Wu must separate the two dark clusters instead of
        // wasting a box on splitting the bright singleton region.
        let mut samples = Vec::new();
        for _ in 0..45 {
            samples.push([0.20, 0.05, 0.0]);
            samples.push([0.30, -0.05, 0.0]);
        }
        for _ in 0..10 {
            samples.push([0.9, 0.0, 0.0]);
        }
        let p = build_palette_wu(&samples, 3);
        assert_eq!(p.len(), 3);
        assert!((p[0][0] - 0.20).abs() < 0.02, "{p:?}");
        assert!((p[1][0] - 0.30).abs() < 0.02, "{p:?}");
        assert!((p[2][0] - 0.9).abs() < 0.02, "{p:?}");
    }
}
