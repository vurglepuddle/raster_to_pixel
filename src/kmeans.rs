//! Adaptive palette building: median-cut initialization + Lloyd k-means,
//! all in Oklab. Fully deterministic (no RNG): identical input + k always
//! yields an identical, luma-sorted palette (PLAN.md bug risk #5).

use crate::color::oklab_dist2;

/// Build a k-color palette (Oklab points) from Oklab samples.
/// `samples` should already exclude fully-transparent pixels.
pub fn build_palette(samples: &[[f32; 3]], k: usize, max_iters: usize) -> Vec<[f32; 3]> {
    assert!(k > 0, "k must be >= 1");
    if samples.is_empty() {
        return vec![[0.0; 3]; 1];
    }
    let mut centers = median_cut_init(samples, k);

    let mut assign = vec![0u32; samples.len()];
    for _ in 0..max_iters {
        let mut moved = false;
        for (i, s) in samples.iter().enumerate() {
            let best = nearest(centers.as_slice(), *s) as u32;
            if assign[i] != best {
                assign[i] = best;
                moved = true;
            }
        }
        // Recompute centroids.
        let mut sums = vec![[0f64; 3]; centers.len()];
        let mut counts = vec![0usize; centers.len()];
        for (i, s) in samples.iter().enumerate() {
            let c = assign[i] as usize;
            counts[c] += 1;
            for ch in 0..3 {
                sums[c][ch] += s[ch] as f64;
            }
        }
        for (c, center) in centers.iter_mut().enumerate() {
            if counts[c] == 0 {
                // Reseed empty cluster to the sample farthest from its center.
                *center = farthest_sample(samples, &assign, &centers_snapshot(&sums, &counts));
                moved = true;
            } else {
                for ch in 0..3 {
                    center[ch] = (sums[c][ch] / counts[c] as f64) as f32;
                }
            }
        }
        if !moved {
            break;
        }
    }

    // Luma-sorted for stable indexed output; f32 total order is fine here
    // (no NaNs can appear from finite inputs).
    centers.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    centers.dedup_by(|a, b| oklab_dist2(*a, *b) < 1e-12);
    centers
}

fn centers_snapshot(sums: &[[f64; 3]], counts: &[usize]) -> Vec<[f32; 3]> {
    sums.iter()
        .zip(counts)
        .map(|(s, &n)| {
            if n == 0 {
                [f32::INFINITY; 3]
            } else {
                [
                    (s[0] / n as f64) as f32,
                    (s[1] / n as f64) as f32,
                    (s[2] / n as f64) as f32,
                ]
            }
        })
        .collect()
}

fn farthest_sample(samples: &[[f32; 3]], assign: &[u32], centers: &[[f32; 3]]) -> [f32; 3] {
    let mut best = samples[0];
    let mut best_d = -1.0f32;
    for (i, s) in samples.iter().enumerate() {
        let c = centers[assign[i] as usize];
        if c[0].is_finite() {
            let d = oklab_dist2(*s, c);
            if d > best_d {
                best_d = d;
                best = *s;
            }
        }
    }
    best
}

/// Index of nearest palette entry to `p` (Oklab squared distance).
pub fn nearest(palette: &[[f32; 3]], p: [f32; 3]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in palette.iter().enumerate() {
        let d = oklab_dist2(*c, p);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Deterministic median-cut: repeatedly split the box with the largest
/// channel range at the median of its widest channel.
fn median_cut_init(samples: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let mut boxes: Vec<Vec<[f32; 3]>> = vec![samples.to_vec()];
    while boxes.len() < k {
        // Pick the splittable box with the widest channel range.
        let mut pick: Option<(usize, usize, f32)> = None; // (box, channel, range)
        for (bi, b) in boxes.iter().enumerate() {
            if b.len() < 2 {
                continue;
            }
            for ch in 0..3 {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for s in b {
                    lo = lo.min(s[ch]);
                    hi = hi.max(s[ch]);
                }
                let range = hi - lo;
                if pick.is_none_or(|(_, _, r)| range > r) {
                    pick = Some((bi, ch, range));
                }
            }
        }
        let Some((bi, ch, range)) = pick else { break };
        if range <= 0.0 {
            break; // All remaining boxes are single colors.
        }
        let mut b = boxes.swap_remove(bi);
        b.sort_by(|a, c| {
            a[ch]
                .partial_cmp(&c[ch])
                .unwrap()
                .then(a[(ch + 1) % 3].partial_cmp(&c[(ch + 1) % 3]).unwrap())
                .then(a[(ch + 2) % 3].partial_cmp(&c[(ch + 2) % 3]).unwrap())
        });
        let mid = b.len() / 2;
        let hi = b.split_off(mid);
        boxes.push(b);
        boxes.push(hi);
    }
    boxes
        .iter()
        .map(|b| {
            let n = b.len() as f64;
            let mut m = [0f64; 3];
            for s in b {
                for ch in 0..3 {
                    m[ch] += s[ch] as f64;
                }
            }
            [(m[0] / n) as f32, (m[1] / n) as f32, (m[2] / n) as f32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_obvious_clusters_are_found() {
        let mut samples = Vec::new();
        for i in 0..50 {
            let j = i as f32 * 1e-4;
            samples.push([0.1 + j, 0.0, 0.0]);
            samples.push([0.9 + j, 0.0, 0.0]);
        }
        let p = build_palette(&samples, 2, 32);
        assert_eq!(p.len(), 2);
        assert!((p[0][0] - 0.1025).abs() < 0.01, "{p:?}");
        assert!((p[1][0] - 0.9025).abs() < 0.01, "{p:?}");
        // Luma-sorted.
        assert!(p[0][0] < p[1][0]);
    }

    #[test]
    fn deterministic_across_runs() {
        let samples: Vec<[f32; 3]> = (0..300)
            .map(|i| {
                let t = i as f32 / 300.0;
                [t, (t * 7.0).sin() * 0.1, (t * 13.0).cos() * 0.1]
            })
            .collect();
        let a = build_palette(&samples, 8, 64);
        let b = build_palette(&samples, 8, 64);
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn fewer_unique_colors_than_k_does_not_pad_duplicates() {
        let samples = vec![[0.2, 0.0, 0.0]; 10];
        let p = build_palette(&samples, 4, 16);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn nearest_picks_correct_entry() {
        let pal = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(nearest(&pal, [0.2, 0.0, 0.0]), 0);
        assert_eq!(nearest(&pal, [0.8, 0.0, 0.0]), 1);
    }
}
