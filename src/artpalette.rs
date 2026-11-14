//! Palette inference for Auto: recover the colours the art was drawn with,
//! and how many there are, from the art-pixel grid.
//!
//! Model. Each art pixel was painted with one of a small set of colours; what
//! we observe per cell is that colour plus noise (compression, generation,
//! resampling). In colour space the observations therefore form one density
//! peak per intended colour, however much area that colour covers. So:
//!
//! * If the cells are noise-free (neighbouring cells of the same region are
//!   bit-identical) every distinct colour is intended: keep them all.
//! * Otherwise candidate colours are the density modes (mean shift with a
//!   small bandwidth: an over-segmentation), and two close candidates are one
//!   colour or two depending on *where* their cells are. Noise scatters two
//!   shades through each other inside one region; two materials occupy
//!   separate regions. So candidates merge when their cells are spatially
//!   intermixed (up to `MIXED_MERGE_DIST` apart in colour) and stay apart
//!   when they are segregated (unless closer than a noise-sized distance,
//!   which nobody can see). A three-cell eye highlight far from every other
//!   colour is its own mode and survives; the shades of one noisy flat
//!   collapse. The count falls out of the data instead of being guessed and
//!   rounded to a preset.
//!
//! Hand-made palettes (the truth set and ~50 Aseprite palettes of up to 32
//! colours) almost never put two entries closer than 0.03 in Oklab, which is
//! why intermixed shades below ~0.04 are safe to treat as one colour.
//!
//! Deterministic: ordered maps, explicit tie-breaks, f32 add/multiply only.

use std::collections::BTreeMap;

use crate::color::{oklab_dist2, oklab_to_srgb8, srgb8_to_oklab};

/// At most this many distinct colours are kept verbatim from a clean grid.
pub const EXACT_MAX_COLORS: usize = 256;
/// Auto never builds a larger inferred palette than this; the bandwidth is
/// widened until the mode count fits.
pub const AUTO_MAX_COLORS: usize = 64;

/// Neighbouring cells closer than this are "the same region" for the purpose
/// of measuring noise.
const SAME_REGION_DIST: f32 = 0.04;
/// Fewer same-region neighbour pairs than this say nothing about noise.
const MIN_NOISE_PAIRS: usize = 8;
/// Mean-shift bandwidth for the candidate colours (Oklab), as a multiple of
/// the measured noise and within fixed limits: small, so that candidates
/// over-segment and the spatial test decides what merges.
const CANDIDATE_BANDWIDTH_PER_NOISE: f32 = 1.2;
const CANDIDATE_BANDWIDTH_MIN: f32 = 0.015;
const CANDIDATE_BANDWIDTH: f32 = 0.015;
/// More candidates than this make the pairwise tests slow; the lightest are
/// folded into their nearest neighbours first. (Widening the bandwidth
/// instead swallows small distinct hues whole: a busy scene lost its teal
/// foliage to the olive next to it that way.)
const MAX_CANDIDATES: usize = 160;
/// Candidates whose cells are fully intermixed merge up to this distance.
const MIXED_MERGE_DIST: f32 = 0.04;
/// Candidates in separate regions merge only below this distance: a multiple
/// of the noise, within limits. Large flat regions make differences of 0.01
/// plainly visible (two stones of slightly different hue), so the floor is
/// low; the ceiling is where even small areas differ visibly.
const SEGREGATED_MERGE_PER_NOISE: f32 = 1.5;
const SEGREGATED_MERGE_MIN: f32 = 0.02;
const SEGREGATED_MERGE_MAX: f32 = 0.02;
/// Mixing ratio (observed / chance adjacency) counted as fully intermixed.
const FULL_MIXING: f32 = 0.5;
/// Entries closer than this fraction of the bandwidth are one colour.
const MERGE_FRACTION: f32 = 0.75;
/// An entry supported by fewer cells than this survives only if it is a
/// salient accent (farther than `ACCENT_DIST` from every other entry).
const MIN_SUPPORT_CELLS: f32 = 2.0;
const ACCENT_DIST: f32 = 0.08;
/// Brute-force mean shift is quadratic; coarsen the input above this.
const MAX_POINTS: usize = 2500;

/// The inferred palette and how it was obtained.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtPalette {
    /// sRGB entries, darkest first.
    pub colors: Vec<[u8; 3]>,
    /// True when the grid was noise-free and the colours were kept verbatim.
    pub exact: bool,
    /// Measured cell-to-cell noise scale (Oklab); 0 for a clean grid.
    pub noise: f32,
}

#[derive(Clone, Copy)]
struct Point {
    lab: [f32; 3],
    weight: f32,
}

/// Infer the palette of a `width x height` grid of cell colours. `cells[i]`
/// is `None` for transparent cells.
pub fn infer(cells: &[Option<[u8; 3]>], width: usize, max_colors: usize) -> ArtPalette {
    let max_colors = max_colors.max(1);
    let mut histogram: BTreeMap<[u8; 3], u32> = BTreeMap::new();
    for c in cells.iter().flatten() {
        *histogram.entry(*c).or_insert(0) += 1;
    }
    if histogram.is_empty() {
        return ArtPalette {
            colors: Vec::new(),
            exact: true,
            noise: 0.0,
        };
    }
    let distinct: Vec<([u8; 3], [f32; 3], f32)> = histogram
        .iter()
        .map(|(&rgb, &n)| (rgb, srgb8_to_oklab(rgb[0], rgb[1], rgb[2]), n as f32))
        .collect();

    // Without same-region neighbours to compare (dither, checker textures)
    // fall back on repetition: clean art reuses its colours many times over.
    let noise = noise_scale(cells, width).unwrap_or_else(|| {
        let opaque = distinct.iter().map(|d| d.2).sum::<f32>();
        if distinct.len() as f32 * 4.0 <= opaque {
            0.0
        } else {
            CANDIDATE_BANDWIDTH
        }
    });
    if noise == 0.0 && distinct.len() <= EXACT_MAX_COLORS.min(max_colors) {
        return ArtPalette {
            colors: sorted_by_luma(distinct.iter().map(|d| d.0).collect()),
            exact: true,
            noise,
        };
    }

    let limit = max_colors.min(AUTO_MAX_COLORS);
    let bandwidth = (noise * CANDIDATE_BANDWIDTH_PER_NOISE)
        .clamp(CANDIDATE_BANDWIDTH_MIN, CANDIDATE_BANDWIDTH);
    let mut clusters = modes(&distinct, bandwidth);
    fold_lightest(&mut clusters, MAX_CANDIDATES);
    let mut adjacency = label_adjacency(cells, width, &distinct, &clusters);
    let segregated = (noise * SEGREGATED_MERGE_PER_NOISE)
        .clamp(SEGREGATED_MERGE_MIN, SEGREGATED_MERGE_MAX);
    merge_by_mixing(&mut clusters, &mut adjacency, segregated);
    while clusters.len() > limit {
        merge_cheapest(&mut clusters, &mut adjacency);
    }
    ArtPalette {
        colors: sorted_by_luma(clusters.iter().map(Cluster::color).collect()),
        exact: false,
        noise,
    }
}

/// Count 4-neighbour cell pairs by candidate label: `adjacency[a][b]` for
/// `a != b` is the number of adjacent (a, b) pairs, `adjacency[a][a]` the
/// number of adjacent pairs inside `a`.
fn label_adjacency(
    cells: &[Option<[u8; 3]>],
    width: usize,
    distinct: &[([u8; 3], [f32; 3], f32)],
    clusters: &[Cluster],
) -> Vec<Vec<f32>> {
    let means: Vec<[f32; 3]> = clusters.iter().map(Cluster::mean).collect();
    let label_of: BTreeMap<[u8; 3], usize> = distinct
        .iter()
        .map(|(rgb, lab, _)| {
            let mut best = (0usize, f32::INFINITY);
            for (i, m) in means.iter().enumerate() {
                let d = oklab_dist2(*lab, *m);
                if d < best.1 {
                    best = (i, d);
                }
            }
            (*rgb, best.0)
        })
        .collect();
    let labels: Vec<Option<usize>> = cells.iter().map(|c| c.map(|c| label_of[&c])).collect();
    let height = cells.len() / width.max(1);
    let mut adjacency = vec![vec![0f32; clusters.len()]; clusters.len()];
    for y in 0..height {
        for x in 0..width {
            let Some(a) = labels[y * width + x] else {
                continue;
            };
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= width || ny >= height {
                    continue;
                }
                if let Some(b) = labels[ny * width + nx] {
                    adjacency[a][b] += 1.0;
                    if a != b {
                        adjacency[b][a] += 1.0;
                    }
                }
            }
        }
    }
    adjacency
}

/// Fold cluster `b` into `a` (`a < b`), keeping the matrix symmetric.
fn fold(clusters: &mut Vec<Cluster>, adjacency: &mut Vec<Vec<f32>>, a: usize, b: usize) {
    let row_b = adjacency[b].clone();
    adjacency[a][a] += row_b[a] + row_b[b];
    for c in 0..clusters.len() {
        if c != a && c != b {
            adjacency[a][c] += row_b[c];
            adjacency[c][a] = adjacency[a][c];
        }
    }
    adjacency.swap_remove(b);
    for row in adjacency.iter_mut() {
        row.swap_remove(b);
    }
    let other = clusters.swap_remove(b);
    clusters[a].absorb(other);
}

/// Area (cells) from which a colour is fully visible: past it a larger area
/// does not make a colour error any worse to look at.
const VISIBLE_AREA_CELLS: f32 = 16.0;

/// The palette is over the cap: merge the pair whose loss is least visible.
/// That is the smallest colour change weighted by the area it happens on
/// (Ward's criterion), with the area counted only up to
/// `VISIBLE_AREA_CELLS`: a few stray cells are cheap to recolour, but among
/// visible areas it is the closest pair of colours that goes, however large
/// they are. Squared error over all cells would instead thin out the hues
/// (a small teal patch costs less than two large greens a step apart) and
/// keep every shade of the dominant ramp. Pairs farther apart than
/// `ACCENT_DIST` (salient accents) are only touched when nothing else is
/// left.
fn merge_cheapest(clusters: &mut Vec<Cluster>, adjacency: &mut Vec<Vec<f32>>) {
    let means: Vec<[f32; 3]> = clusters.iter().map(Cluster::mean).collect();
    let accent2 = ACCENT_DIST * ACCENT_DIST;
    let mut best: Option<(usize, usize, bool, f32)> = None;
    for a in 0..clusters.len() {
        for b in a + 1..clusters.len() {
            let d2 = oklab_dist2(means[a], means[b]);
            let (wa, wb) = (clusters[a].weight, clusters[b].weight);
            let cost = (wa * wb / (wa + wb)).min(VISIBLE_AREA_CELLS) * d2;
            let accent = d2 >= accent2;
            let better = match best {
                None => true,
                Some((_, _, best_accent, best_cost)) => {
                    (!accent && best_accent) || (accent == best_accent && cost < best_cost)
                }
            };
            if better {
                best = Some((a, b, accent, cost));
            }
        }
    }
    if let Some((a, b, _, _)) = best {
        fold(clusters, adjacency, a, b);
    }
}

/// Fold the lightest cluster into its nearest neighbour until at most
/// `limit` remain (ties: the earlier cluster).
fn fold_lightest(clusters: &mut Vec<Cluster>, limit: usize) {
    while clusters.len() > limit.max(1) {
        let mut light = 0;
        for (i, c) in clusters.iter().enumerate() {
            if c.weight < clusters[light].weight {
                light = i;
            }
        }
        let mean = clusters[light].mean();
        let mut near = (usize::MAX, f32::INFINITY);
        for (i, c) in clusters.iter().enumerate() {
            let d = oklab_dist2(mean, c.mean());
            if i != light && d < near.1 {
                near = (i, d);
            }
        }
        let removed = clusters.swap_remove(light);
        // `swap_remove` moved the last cluster into `light`.
        let target = if near.0 == clusters.len() { light } else { near.0 };
        clusters[target].absorb(removed);
    }
}

/// Merge candidate pairs, best first, while any pair is closer than the
/// distance its spatial mixing allows.
fn merge_by_mixing(clusters: &mut Vec<Cluster>, adjacency: &mut Vec<Vec<f32>>, near: f32) {
    let far = MIXED_MERGE_DIST;
    loop {
        let means: Vec<[f32; 3]> = clusters.iter().map(Cluster::mean).collect();
        let mut best: Option<(usize, usize, f32)> = None;
        for a in 0..clusters.len() {
            for b in a + 1..clusters.len() {
                let d = oklab_dist2(means[a], means[b]).sqrt();
                if d >= far {
                    continue;
                }
                let between = adjacency[a][b];
                let pairs = adjacency[a][a] + adjacency[b][b] + between;
                let share = clusters[a].weight / (clusters[a].weight + clusters[b].weight);
                let chance = 2.0 * share * (1.0 - share) * pairs;
                let mixing = if chance > 0.0 { between / chance } else { 0.0 };
                let allowed = near + (far - near) * (mixing / FULL_MIXING).min(1.0);
                let margin = allowed - d;
                if margin > 0.0 && best.is_none_or(|(_, _, m)| margin > m) {
                    best = Some((a, b, margin));
                }
            }
        }
        let Some((a, b, _)) = best else { break };
        fold(clusters, adjacency, a, b);
    }
}

fn sorted_by_luma(mut colors: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    colors.sort_by(|a, b| {
        let (la, lb) = (
            srgb8_to_oklab(a[0], a[1], a[2]),
            srgb8_to_oklab(b[0], b[1], b[2]),
        );
        la[0].partial_cmp(&lb[0]).unwrap().then(a.cmp(b))
    });
    colors.dedup();
    colors
}

/// Median Oklab distance between 4-neighbour cells that belong to the same
/// region. Exactly 0 when flat regions are bit-identical (a clean grid);
/// `None` when there are too few such pairs to tell.
fn noise_scale(cells: &[Option<[u8; 3]>], width: usize) -> Option<f32> {
    let height = cells.len() / width.max(1);
    let limit2 = SAME_REGION_DIST * SAME_REGION_DIST;
    let mut zero = 0usize;
    let mut dists: Vec<f32> = Vec::new();
    let lab = |c: [u8; 3]| srgb8_to_oklab(c[0], c[1], c[2]);
    for y in 0..height {
        for x in 0..width {
            let Some(a) = cells[y * width + x] else {
                continue;
            };
            let mut la = None;
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= width || ny >= height {
                    continue;
                }
                let Some(b) = cells[ny * width + nx] else {
                    continue;
                };
                if a == b {
                    zero += 1;
                    continue;
                }
                let la = *la.get_or_insert_with(|| lab(a));
                let d2 = oklab_dist2(la, lab(b));
                if d2 < limit2 {
                    dists.push(d2);
                }
            }
        }
    }
    let total = zero + dists.len();
    if total < MIN_NOISE_PAIRS {
        return None;
    }
    if zero * 2 >= total {
        return Some(0.0);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Median over all same-region pairs, the identical ones included.
    Some(dists[total / 2 - zero].sqrt())
}

/// Density modes of the weighted colour set, as clusters of the input.
fn modes(distinct: &[([u8; 3], [f32; 3], f32)], bandwidth: f32) -> Vec<Cluster> {
    // Pre-bin on a lattice (coarsened until brute force is affordable).
    let mut step = bandwidth * 0.25;
    let mut points: Vec<Point>;
    let mut point_of: Vec<usize>;
    loop {
        let mut bins: BTreeMap<[i32; 3], usize> = BTreeMap::new();
        let mut sums: Vec<([f64; 3], f64)> = Vec::new();
        point_of = Vec::with_capacity(distinct.len());
        for (_, lab, w) in distinct {
            let key = [
                (lab[0] / step).floor() as i32,
                (lab[1] / step).floor() as i32,
                (lab[2] / step).floor() as i32,
            ];
            let next = sums.len();
            let idx = *bins.entry(key).or_insert(next);
            if idx == next {
                sums.push(([0.0; 3], 0.0));
            }
            for ch in 0..3 {
                sums[idx].0[ch] += lab[ch] as f64 * *w as f64;
            }
            sums[idx].1 += *w as f64;
            point_of.push(idx);
        }
        points = sums
            .iter()
            .map(|(s, w)| Point {
                lab: [(s[0] / w) as f32, (s[1] / w) as f32, (s[2] / w) as f32],
                weight: *w as f32,
            })
            .collect();
        if points.len() <= MAX_POINTS {
            break;
        }
        step *= 1.5;
    }

    // Mean shift (uniform window = gradient ascent on an Epanechnikov KDE).
    let h2 = bandwidth * bandwidth;
    let mut summit: Vec<[f32; 3]> = Vec::with_capacity(points.len());
    for start in &points {
        let mut x = start.lab;
        for _ in 0..40 {
            let (mut sum, mut w_sum) = ([0f32; 3], 0f32);
            for p in &points {
                if oklab_dist2(p.lab, x) <= h2 {
                    w_sum += p.weight;
                    for ch in 0..3 {
                        sum[ch] += p.lab[ch] * p.weight;
                    }
                }
            }
            if w_sum <= 0.0 {
                break;
            }
            let next = [sum[0] / w_sum, sum[1] / w_sum, sum[2] / w_sum];
            let moved = oklab_dist2(next, x);
            x = next;
            if moved < 1e-10 {
                break;
            }
        }
        summit.push(x);
    }

    // Points whose summits coincide form one cluster. Heaviest first so the
    // strongest colours define the cluster centres.
    let merge2 = (bandwidth * MERGE_FRACTION) * (bandwidth * MERGE_FRACTION);
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[b]
            .weight
            .partial_cmp(&points[a].weight)
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut centres: Vec<[f32; 3]> = Vec::new();
    let mut cluster_of = vec![0usize; points.len()];
    for &i in &order {
        let mut best = None;
        let mut best_d = merge2;
        for (c, centre) in centres.iter().enumerate() {
            let d = oklab_dist2(*centre, summit[i]);
            if d <= best_d {
                best_d = d;
                best = Some(c);
            }
        }
        cluster_of[i] = best.unwrap_or_else(|| {
            centres.push(summit[i]);
            centres.len() - 1
        });
    }

    // Cluster statistics from the original distinct colours.
    let mut clusters: Vec<Cluster> = (0..centres.len()).map(|_| Cluster::default()).collect();
    for (i, (rgb, lab, w)) in distinct.iter().enumerate() {
        clusters[cluster_of[point_of[i]]].add(*rgb, *lab, *w);
    }

    // Merge what is still closer than the merge distance, then fold weak
    // non-salient entries into their nearest neighbour.
    loop {
        let mut best: Option<(usize, usize, f32)> = None;
        for a in 0..clusters.len() {
            for b in a + 1..clusters.len() {
                let d = oklab_dist2(clusters[a].mean(), clusters[b].mean());
                if d < merge2 && best.is_none_or(|(_, _, bd)| d < bd) {
                    best = Some((a, b, d));
                }
            }
        }
        let Some((a, b, _)) = best else { break };
        let other = clusters.swap_remove(b);
        clusters[a].absorb(other);
    }
    let accent2 = ACCENT_DIST * ACCENT_DIST;
    loop {
        let mut victim: Option<(usize, usize)> = None;
        for a in 0..clusters.len() {
            if clusters[a].weight >= MIN_SUPPORT_CELLS || clusters.len() == 1 {
                continue;
            }
            let mut nearest = (usize::MAX, f32::INFINITY);
            for b in 0..clusters.len() {
                if b != a {
                    let d = oklab_dist2(clusters[a].mean(), clusters[b].mean());
                    if d < nearest.1 {
                        nearest = (b, d);
                    }
                }
            }
            if nearest.1 < accent2 {
                victim = Some((a, nearest.0));
                break;
            }
        }
        let Some((a, into)) = victim else { break };
        let taken = std::mem::take(&mut clusters[a]);
        clusters[into].absorb(taken);
        clusters.swap_remove(a);
    }

    clusters
}

#[derive(Default)]
struct Cluster {
    sum: [f64; 3],
    weight: f32,
    /// Heaviest member colour, for snapping the entry to a real colour.
    top: Option<([u8; 3], f32)>,
}

impl Cluster {
    fn add(&mut self, rgb: [u8; 3], lab: [f32; 3], w: f32) {
        for ch in 0..3 {
            self.sum[ch] += lab[ch] as f64 * w as f64;
        }
        self.weight += w;
        if self.top.is_none_or(|(c, tw)| w > tw || (w == tw && rgb < c)) {
            self.top = Some((rgb, w));
        }
    }

    fn absorb(&mut self, other: Cluster) {
        for ch in 0..3 {
            self.sum[ch] += other.sum[ch];
        }
        self.weight += other.weight;
        if let Some((rgb, w)) = other.top {
            if self.top.is_none_or(|(c, tw)| w > tw || (w == tw && rgb < c)) {
                self.top = Some((rgb, w));
            }
        }
    }

    fn mean(&self) -> [f32; 3] {
        let w = (self.weight as f64).max(f64::MIN_POSITIVE);
        [
            (self.sum[0] / w) as f32,
            (self.sum[1] / w) as f32,
            (self.sum[2] / w) as f32,
        ]
    }

    /// The entry: the dominant real colour when one holds most of the
    /// cluster (partly clean sources), else the centroid.
    fn color(&self) -> [u8; 3] {
        match self.top {
            Some((rgb, w)) if w * 2.0 >= self.weight => rgb,
            _ => oklab_to_srgb8(self.mean()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(width: usize, f: impl Fn(usize, usize) -> Option<[u8; 3]>) -> Vec<Option<[u8; 3]>> {
        (0..width * width).map(|i| f(i % width, i / width)).collect()
    }

    /// Small deterministic noise in -amp..=amp.
    fn jitter(x: usize, y: usize, ch: usize, amp: i32) -> i32 {
        let h = (x as u32)
            .wrapping_mul(73_856_093)
            ^ (y as u32).wrapping_mul(19_349_663)
            ^ (ch as u32).wrapping_mul(83_492_791);
        (h.wrapping_mul(2_654_435_761) >> 24) as i32 % (2 * amp + 1) - amp
    }

    fn noisy(base: [u8; 3], x: usize, y: usize, amp: i32) -> [u8; 3] {
        std::array::from_fn(|ch| (base[ch] as i32 + jitter(x, y, ch, amp)).clamp(0, 255) as u8)
    }

    #[test]
    fn clean_grid_keeps_every_colour_even_near_duplicates() {
        let colors = [[10u8, 10, 10], [12, 10, 10], [200, 180, 40], [202, 180, 40]];
        let cells = grid(16, |x, y| Some(colors[(x / 8) + 2 * (y / 8)]));
        let p = infer(&cells, 16, 256);
        assert!(p.exact);
        assert_eq!(p.colors.len(), 4);
    }

    #[test]
    fn noisy_flats_collapse_to_one_entry_each_and_accent_survives() {
        let (gold, dark, teal) = ([220u8, 170, 60], [40u8, 30, 30], [30u8, 90, 100]);
        let cells = grid(32, |x, y| {
            Some(if (14..16).contains(&x) && (14..16).contains(&y) {
                noisy(teal, x, y, 3)
            } else if x < 16 {
                noisy(gold, x, y, 3)
            } else {
                noisy(dark, x, y, 3)
            })
        });
        let p = infer(&cells, 32, 256);
        assert!(!p.exact && p.noise > 0.0);
        assert_eq!(p.colors.len(), 3, "{:?}", p.colors);
        let near = |c: [u8; 3], t: [u8; 3]| (0..3).all(|i| (c[i] as i32 - t[i] as i32).abs() <= 4);
        for target in [gold, dark, teal] {
            assert!(p.colors.iter().any(|&c| near(c, target)), "{target:?} lost");
        }
    }

    #[test]
    fn a_single_stray_cell_near_a_big_colour_is_not_an_entry() {
        let cells = grid(24, |x, y| {
            Some(if (x, y) == (5, 5) {
                [150, 110, 62]
            } else if (x + y) % 2 == 0 {
                noisy([140, 100, 60], x, y, 2)
            } else {
                noisy([60, 80, 140], x, y, 2)
            })
        });
        assert_eq!(infer(&cells, 24, 256).colors.len(), 2);
    }

    #[test]
    fn transparent_cells_are_ignored_and_empty_grids_are_empty() {
        let cells = grid(8, |x, _| (x < 4).then_some([9, 9, 9]));
        assert_eq!(infer(&cells, 8, 256).colors, vec![[9, 9, 9]]);
        assert!(infer(&grid(4, |_, _| None), 4, 256).colors.is_empty());
    }

    #[test]
    fn continuous_tone_is_capped() {
        let cells = grid(64, |x, y| Some([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]));
        let p = infer(&cells, 64, 256);
        assert!(p.colors.len() <= AUTO_MAX_COLORS && p.colors.len() >= 8);
    }
}
