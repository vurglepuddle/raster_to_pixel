//! Elastic sampling grid: cell boundaries that follow the edges near them.
//!
//! A fitted lattice describes the image as a whole. Generated pixel art also
//! contains parts drawn off that lattice: lettering half a cell out of phase,
//! strokes 1.4 cells wide, an ornament on its own offset grid, a slow drift.
//! A cell of the rigid lattice then straddles two art pixels, and whichever
//! colour is reported for it, the feature is lost or smeared.
//!
//! The model here keeps the lattice's cell count and lets every boundary move
//! by up to half a cell. Each output row owns its own column boundaries and
//! each output column its own row boundaries, so a shift needs only the
//! evidence of the band it happens in. One band is one small dynamic program:
//!
//!   maximise  sum_i E(b_i)                      edges that land on a boundary
//!           - ANCHOR  * sum_i ((b_i - t_i)/p)^2      distance from the lattice
//!           - STRETCH * sum_i ((b_i+1 - b_i - p)/p)^2    cell width change
//!
//! `E(x)` is the share of the band's scan lines with an edge (a local maximum
//! of the first difference) at pixel boundary `x`, so one transition counts
//! once however blurred or strong it is, and the penalties are in the same
//! unit: "one full-height edge". Where the source is on the lattice the
//! optimum is the lattice itself and nothing changes.
//!
//! `t_i` is the lattice line moved to the local phase of the art around it
//! (`local_phase`). A region drawn on its own offset grid, such as an
//! ornament pasted into a frame, then keeps its offset where it has no edges
//! (a stretch of identical rows), instead of drifting back to the global
//! lattice there and splitting its art pixels between two cells.
//!
//! Only additions, multiplications and comparisons on integers and `f64`s are
//! used, so native and WebAssembly builds agree bit for bit.

use crate::downsample::SamplingGrid;
use image::RgbaImage;

/// Cells narrower than this (source pixels) leave nothing to move.
pub const MIN_PITCH: f64 = 3.0;
/// An edge is a local maximum of the first difference (sum of absolute
/// premultiplied RGBA steps). At or below `EDGE_LOW` it counts for nothing,
/// from `EDGE_HIGH` fully.
const EDGE_LOW: i32 = 10;
const EDGE_HIGH: i32 = 40;
/// An edge is sharp when it carries at least this share of the five
/// differences around it (a step: 1, a two-pixel ramp or one side of a thin
/// line: 1/2, the even slope of an interpolated enlargement: 1/3 or less).
const SHARP_SHARE: f64 = 0.4;
/// Share of the edge weight that must be sharp for the image to be treated
/// as drawn cell by cell. Below it the source is an interpolated enlargement:
/// its slopes say nothing about boundaries, and its lattice is exact anyway.
const MIN_SHARP_EDGES: f64 = 0.5;
/// A lattice line already owns the edges this close to it (source pixels, or
/// this share of a cell if larger): resampling and wobble put edges there.
const OWN_PIXELS: f64 = 1.0;
const OWN_CELLS: f64 = 0.15;
/// Cost of a boundary sitting a whole cell away from its lattice line.
const ANCHOR: f64 = 0.5;
/// Cost of a cell one whole pitch wider or narrower than nominal.
const STRETCH: f64 = 0.5;
/// Largest boundary displacement, in cells.
const MAX_SHIFT: f64 = 0.5;
/// Narrowest cell, in cells.
const MIN_WIDTH: f64 = 0.45;
/// Local phase window: this many cells to each side of a boundary.
const PHASE_WINDOW: usize = 4;
/// Edges in the window must agree on one phase at least this well
/// (resultant length of their phase angles, 0..1).
const PHASE_MIN_AGREEMENT: f64 = 0.5;
/// Edge weight the window needs, in full-strength edges one cell long.
const PHASE_MIN_EDGES: f64 = 4.0;
/// A local phase nearer the lattice than this (pixels) is the lattice.
const PHASE_MIN_SHIFT: f64 = 0.75;

/// Per-band cell boundaries in source pixels.
#[derive(Clone, Debug)]
pub struct ElasticGrid {
    pub cols: usize,
    pub rows: usize,
    /// `rows * (cols + 1)`: column boundaries of each output row.
    xs: Vec<f64>,
    /// `cols * (rows + 1)`: row boundaries of each output column.
    ys: Vec<f64>,
    /// Boundaries that left their lattice line by at least half a pixel.
    pub moved: usize,
}

impl ElasticGrid {
    /// `[x0, x1, y0, y1]` of output cell `(cx, cy)`.
    pub fn bounds(&self, cx: usize, cy: usize) -> [f64; 4] {
        let x = cy * (self.cols + 1) + cx;
        let y = cx * (self.rows + 1) + cy;
        [self.xs[x], self.xs[x + 1], self.ys[y], self.ys[y + 1]]
    }
}

/// Edge weights (0..=255) of one scan line: `out[i]` is the edge between
/// pixels `i - 1` and `i`; `out[0]` stays 0. Adds the line's total and sharp
/// edge weight to `tally`.
fn line_edges(
    buf: &[u8],
    start: usize,
    stride: usize,
    len: usize,
    d: &mut Vec<i32>,
    out: &mut [u8],
    tally: &mut (u64, u64),
) {
    d.clear();
    let px = |i: usize| -> [i32; 4] {
        let p = &buf[start + i * stride..start + i * stride + 4];
        // Premultiplied, so colour hidden under transparency casts no edges.
        let a = p[3] as i32;
        [
            (p[0] as i32 * a + 127) / 255,
            (p[1] as i32 * a + 127) / 255,
            (p[2] as i32 * a + 127) / 255,
            a,
        ]
    };
    let mut prev = px(0);
    for i in 1..len {
        let cur = px(i);
        let mut sum = 0;
        for c in 0..4 {
            sum += (cur[c] - prev[c]).abs();
        }
        d.push(sum);
        prev = cur;
    }
    for i in 0..d.len() {
        let v = d[i];
        if v <= EDGE_LOW {
            continue;
        }
        let left = if i > 0 { d[i - 1] } else { 0 };
        let right = if i + 1 < d.len() { d[i + 1] } else { 0 };
        if v >= left && v > right {
            let w = ((v - EDGE_LOW) * 255 / (EDGE_HIGH - EDGE_LOW)).min(255);
            out[i + 1] = w as u8;
            let lo = i.saturating_sub(2);
            let hi = (i + 2).min(d.len() - 1);
            let around: i32 = d[lo..=hi].iter().sum();
            tally.0 += w as u64;
            if v as f64 >= SHARP_SHARE * around as f64 {
                tally.1 += w as u64;
            }
        }
    }
}

/// Reusable buffers of the per-band dynamic program.
#[derive(Default)]
struct Solver {
    profile: Vec<f64>,
    smooth: Vec<f64>,
    /// Candidate positions of all boundaries, flattened.
    pos: Vec<f64>,
    score: Vec<f64>,
    back: Vec<u32>,
    /// First candidate of each boundary (plus the end).
    first: Vec<usize>,
}

impl Solver {
    /// Boundaries `out[0..=cells]` along a band of `len` source pixels whose
    /// edge profile is `self.profile` (`len + 1` entries). Returns how many
    /// moved by at least half a pixel.
    /// `anchor[i]` moves boundary `i`'s home from its lattice line by that
    /// many pixels (the local phase of the art around it).
    fn solve(
        &mut self,
        len: usize,
        origin: f64,
        pitch: f64,
        cells: usize,
        anchor: &[f64],
        out: &mut [f64],
    ) -> usize {
        let n = len as f64;
        // A wobbling edge spreads over neighbouring positions: let each
        // position see half of its neighbours' share.
        self.smooth.clear();
        for i in 0..=len {
            let left = if i > 0 { self.profile[i - 1] } else { 0.0 };
            let right = if i < len { self.profile[i + 1] } else { 0.0 };
            self.smooth.push(self.profile[i] + 0.5 * (left + right));
        }

        self.pos.clear();
        self.score.clear();
        self.back.clear();
        self.first.clear();
        let reach = MAX_SHIFT * pitch;
        let min_width = (MIN_WIDTH * pitch).max(1.0);
        for i in 0..=cells {
            let nominal = (origin + i as f64 * pitch + anchor[i]).clamp(0.0, n);
            self.first.push(self.pos.len());
            // Candidates in order of preference: the lattice line itself,
            // then whole-pixel positions by increasing distance from it.
            self.pos.push(nominal);
            let base = nominal.round();
            let span = reach.ceil() as i64 + 1;
            for step in 0..=span {
                for sign in [1.0, -1.0] {
                    if step == 0 && sign < 0.0 {
                        continue;
                    }
                    let p = base + sign * step as f64;
                    if p < 0.0 || p > n || p == nominal || (p - nominal).abs() > reach {
                        continue;
                    }
                    self.pos.push(p);
                }
            }
        }
        self.first.push(self.pos.len());

        let own = OWN_PIXELS.max(OWN_CELLS * pitch);
        let reward = |smooth: &[f64], p: f64, nominal: f64| -> f64 {
            let edge = if p == nominal {
                // The lattice line is credited with the best edge it owns.
                let lo = (nominal - own).ceil().max(0.0) as usize;
                let hi = (nominal + own).floor().min(n) as usize;
                let mut best = 0.0;
                for k in lo..=hi {
                    if smooth[k] > best {
                        best = smooth[k];
                    }
                }
                best
            } else {
                smooth[p as usize]
            };
            let shift = (p - nominal) / pitch;
            edge - ANCHOR * shift * shift
        };
        self.score.resize(self.pos.len(), f64::NEG_INFINITY);
        self.back.resize(self.pos.len(), u32::MAX);
        for i in 0..=cells {
            let nominal = self.pos[self.first[i]];
            for s in self.first[i]..self.first[i + 1] {
                let p = self.pos[s];
                let own = reward(&self.smooth, p, nominal);
                if i == 0 {
                    self.score[s] = own;
                    continue;
                }
                let mut best = f64::NEG_INFINITY;
                let mut best_t = u32::MAX;
                for t in self.first[i - 1]..self.first[i] {
                    let width = p - self.pos[t];
                    if width < min_width || self.score[t] == f64::NEG_INFINITY {
                        continue;
                    }
                    let stretch = (width - pitch) / pitch;
                    let v = self.score[t] - STRETCH * stretch * stretch;
                    if v > best {
                        best = v;
                        best_t = t as u32;
                    }
                }
                if best_t != u32::MAX {
                    self.score[s] = best + own;
                    self.back[s] = best_t;
                }
            }
        }

        // Best end state (first wins ties: nearest the lattice), then trace.
        let mut s = self.first[cells];
        for t in self.first[cells]..self.first[cells + 1] {
            if self.score[t] > self.score[s] {
                s = t;
            }
        }
        if self.score[s] == f64::NEG_INFINITY {
            // Cannot happen (the lattice itself is always feasible); stay rigid.
            for (i, b) in out.iter_mut().enumerate().take(cells + 1) {
                *b = (origin + i as f64 * pitch).clamp(0.0, n);
            }
            return 0;
        }
        let mut moved = 0;
        for i in (0..=cells).rev() {
            out[i] = self.pos[s];
            let lattice = (origin + i as f64 * pitch).clamp(0.0, n);
            if (self.pos[s] - lattice).abs() >= 0.5 {
                moved += 1;
            }
            if i > 0 {
                s = self.back[s] as usize;
            }
        }
        moved
    }
}

/// Local phase of the edges that column (or row) boundaries follow, as a
/// whole-pixel shift for every boundary of every band (`band * (cells + 1) +
/// boundary`). `edge(along, across)` is the edge weight at pixel boundary
/// `along` on scan line `across`; boundaries split the `along` axis into
/// `cells` cells from `origin` at `pitch`, bands split the `across` axis into
/// `bands` cells from `band_origin` at `band_pitch`.
///
/// Every edge votes with its position on the lattice as an angle (a full
/// pitch is a full turn); the circular mean over the cells around a boundary
/// is the phase the art there is drawn at. Where the edges agree on a phase
/// at least a pixel from the lattice line, the boundary's home moves there:
/// art drawn on its own offset grid (an ornament, a sprite pasted in) keeps
/// its offset through stretches without edges, instead of drifting back to
/// the global lattice halfway through.
#[allow(clippy::too_many_arguments)]
fn local_phase(
    edge: &dyn Fn(usize, usize) -> u8,
    along_len: usize,
    across_len: usize,
    origin: f64,
    pitch: f64,
    cells: usize,
    band_origin: f64,
    band_pitch: f64,
    bands: usize,
) -> Vec<f64> {
    let stride = cells + 1;
    let turn = std::f64::consts::TAU / pitch;
    let (cos_t, sin_t): (Vec<f64>, Vec<f64>) = (0..=along_len)
        .map(|a| {
            let (s, c) = crate::fmath::sin_cos(((a as f64 - origin) * turn) as f32);
            (c as f64, s as f64)
        })
        .unzip();
    // Per (band, boundary) block sums of cos, sin and weight, then prefix
    // sums over both axes for the windows.
    let mut sums = vec![[0f64; 3]; (bands + 1) * (stride + 1)];
    for across in 0..across_len {
        let b = (((across as f64 - band_origin) / band_pitch).floor().max(0.0) as usize).min(bands - 1);
        for a in 1..along_len {
            let v = edge(a, across);
            if v == 0 {
                continue;
            }
            let i = (((a as f64 - origin) / pitch).round().max(0.0) as usize).min(cells);
            let wgt = v as f64;
            let cell = &mut sums[(b + 1) * (stride + 1) + i + 1];
            cell[0] += wgt * cos_t[a];
            cell[1] += wgt * sin_t[a];
            cell[2] += wgt;
        }
    }
    for b in 1..=bands {
        for i in 1..=stride {
            for k in 0..3 {
                let v = sums[b * (stride + 1) + i][k] + sums[(b - 1) * (stride + 1) + i][k]
                    + sums[b * (stride + 1) + i - 1][k]
                    - sums[(b - 1) * (stride + 1) + i - 1][k];
                sums[b * (stride + 1) + i][k] = v;
            }
        }
    }
    let min_weight = PHASE_MIN_EDGES * 255.0 * pitch;
    let mut out = vec![0f64; bands * stride];
    for b in 0..bands {
        let (b0, b1) = (b.saturating_sub(PHASE_WINDOW), (b + PHASE_WINDOW + 1).min(bands));
        for i in 0..stride {
            let (i0, i1) = (i.saturating_sub(PHASE_WINDOW), (i + PHASE_WINDOW + 1).min(stride));
            let at = |bb: usize, ii: usize, k: usize| sums[bb * (stride + 1) + ii][k];
            let total = |k: usize| at(b1, i1, k) - at(b0, i1, k) - at(b1, i0, k) + at(b0, i0, k);
            let (c, s, wgt) = (total(0), total(1), total(2));
            if wgt < min_weight || (c * c + s * s).sqrt() < PHASE_MIN_AGREEMENT * wgt {
                continue;
            }
            let shift = crate::fmath::atan2(s as f32, c as f32) as f64 / turn;
            if shift.abs() >= PHASE_MIN_SHIFT {
                out[b * stride + i] = shift.round();
            }
        }
    }
    out
}

/// Fit elastic boundaries for `cols x rows` cells of `grid` over `src`.
/// Returns `None` when the cells are too small to move anything.
pub fn fit(src: &RgbaImage, grid: SamplingGrid, cols: usize, rows: usize) -> Option<ElasticGrid> {
    if grid.cell_w < MIN_PITCH || grid.cell_h < MIN_PITCH || cols == 0 || rows == 0 {
        return None;
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    let buf = src.as_raw();

    // Edge maps: `ex[y * w + x]` between columns x-1 and x, `ey` between rows.
    let mut ex = vec![0u8; w * h];
    let mut ey = vec![0u8; w * h];
    let mut d = Vec::new();
    let mut tally = (0u64, 0u64);
    for y in 0..h {
        line_edges(buf, y * w * 4, 4, w, &mut d, &mut ex[y * w..(y + 1) * w], &mut tally);
    }
    let mut column = vec![0u8; h];
    for x in 0..w {
        column.iter_mut().for_each(|v| *v = 0);
        line_edges(buf, x * 4, w * 4, h, &mut d, &mut column, &mut tally);
        for y in 0..h {
            ey[y * w + x] = column[y];
        }
    }

    let sharp = if tally.0 == 0 {
        0.0
    } else {
        tally.1 as f64 / tally.0 as f64
    };
    if sharp < MIN_SHARP_EDGES {
        return None;
    }

    let band = |origin: f64, pitch: f64, index: usize, len: usize| -> (usize, usize) {
        let a = (origin + index as f64 * pitch).round().clamp(0.0, len as f64) as usize;
        let b = (origin + (index + 1) as f64 * pitch).round().clamp(0.0, len as f64) as usize;
        let a = a.min(len - 1);
        (a, b.max(a + 1).min(len))
    };

    let phase_x = local_phase(
        &|a, across| ex[across * w + a],
        w,
        h,
        grid.origin_x,
        grid.cell_w,
        cols,
        grid.origin_y,
        grid.cell_h,
        rows,
    );
    let phase_y = local_phase(
        &|a, across| ey[a * w + across],
        h,
        w,
        grid.origin_y,
        grid.cell_h,
        rows,
        grid.origin_x,
        grid.cell_w,
        cols,
    );

    let mut solver = Solver::default();
    let mut moved = 0;
    let mut xs = vec![0.0; rows * (cols + 1)];
    for cy in 0..rows {
        let (y0, y1) = band(grid.origin_y, grid.cell_h, cy, h);
        solver.profile.clear();
        solver.profile.resize(w + 1, 0.0);
        for y in y0..y1 {
            for x in 1..w {
                solver.profile[x] += ex[y * w + x] as f64;
            }
        }
        let scale = 1.0 / (255.0 * (y1 - y0) as f64);
        solver.profile.iter_mut().for_each(|v| *v *= scale);
        moved += solver.solve(
            w,
            grid.origin_x,
            grid.cell_w,
            cols,
            &phase_x[cy * (cols + 1)..(cy + 1) * (cols + 1)],
            &mut xs[cy * (cols + 1)..(cy + 1) * (cols + 1)],
        );
    }
    let mut ys = vec![0.0; cols * (rows + 1)];
    for cx in 0..cols {
        let (x0, x1) = band(grid.origin_x, grid.cell_w, cx, w);
        solver.profile.clear();
        solver.profile.resize(h + 1, 0.0);
        for y in 1..h {
            let mut sum = 0u32;
            for x in x0..x1 {
                sum += ey[y * w + x] as u32;
            }
            solver.profile[y] = sum as f64;
        }
        let scale = 1.0 / (255.0 * (x1 - x0) as f64);
        solver.profile.iter_mut().for_each(|v| *v *= scale);
        moved += solver.solve(
            h,
            grid.origin_y,
            grid.cell_h,
            rows,
            &phase_y[cx * (rows + 1)..(cx + 1) * (rows + 1)],
            &mut ys[cx * (rows + 1)..(cx + 1) * (rows + 1)],
        );
    }
    Some(ElasticGrid {
        cols,
        rows,
        xs,
        ys,
        moved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn lattice(pitch: f64) -> SamplingGrid {
        SamplingGrid {
            origin_x: 0.0,
            origin_y: 0.0,
            cell_w: pitch,
            cell_h: pitch,
        }
    }

    /// 8x8 cells of 6 px, pseudo-random colours from four.
    fn on_grid_art() -> RgbaImage {
        let colours = [[30u8, 30, 60], [220, 200, 90], [90, 160, 80], [200, 80, 70]];
        RgbaImage::from_fn(48, 48, |x, y| {
            let (cx, cy) = (x / 6, y / 6);
            let i = (cx * 7 + cy * 13 + (cx * cy) % 5) % 4;
            let c = colours[i as usize];
            Rgba([c[0], c[1], c[2], 255])
        })
    }

    #[test]
    fn on_grid_art_keeps_the_lattice() {
        let grid = fit(&on_grid_art(), lattice(6.0), 8, 8).unwrap();
        assert_eq!(grid.moved, 0);
        assert_eq!(grid.bounds(3, 5), [18.0, 24.0, 30.0, 36.0]);
    }

    #[test]
    fn an_off_grid_stroke_gets_its_own_cells() {
        // A vertical stroke 6 px wide, 3 px (half a cell) off the lattice.
        let img = RgbaImage::from_fn(48, 48, |x, _| {
            if (21..27).contains(&x) {
                Rgba([240, 220, 80, 255])
            } else {
                Rgba([40, 40, 50, 255])
            }
        });
        let grid = fit(&img, lattice(6.0), 8, 8).unwrap();
        assert!(grid.moved > 0);
        // Some cell of every row is exactly the stroke.
        for cy in 0..8 {
            let exact = (0..8).any(|cx| {
                let b = grid.bounds(cx, cy);
                b[0] == 21.0 && b[1] == 27.0
            });
            assert!(exact, "row {cy}");
        }
    }

    #[test]
    fn an_offset_region_keeps_its_offset_through_a_stretch_without_edges() {
        // 12 x 12 cells of 6 px. Every art row is drawn 2 px above its
        // lattice row (boundaries at 4, 10, 16, ...), and art rows 4..=8 are
        // identical, so the boundaries between them have no edge to follow.
        let colours = [[30u8, 30, 60], [220, 200, 90], [90, 160, 80], [200, 80, 70]];
        let img = RgbaImage::from_fn(72, 72, |x, y| {
            let row = (y + 2) / 6;
            let row = if (4..=8).contains(&row) { 4 } else { row };
            let c = colours[((x / 6) * 3 + row * 5) as usize % 4];
            Rgba([c[0], c[1], c[2], 255])
        });
        let grid = fit(&img, lattice(6.0), 12, 12).unwrap();
        for cx in 2..10 {
            for cy in 2..11 {
                let y0 = grid.bounds(cx, cy)[2];
                assert_eq!(y0, cy as f64 * 6.0 - 2.0, "column {cx} row {cy}");
            }
        }
    }

    #[test]
    fn small_cells_are_left_alone() {
        assert!(fit(&on_grid_art(), lattice(2.0), 24, 24).is_none());
    }
}
