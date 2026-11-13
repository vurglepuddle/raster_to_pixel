//! Auditable metric definitions (all distances use Oklab's native 0..~1 units).
//!
//! Exact RGBA classes canonicalize alpha=0 to [0,0,0,0]; partial alpha remains
//! distinct. DeltaE is Euclidean Oklab RGB distance, canonical transparent RGB
//! black; it deliberately has no alpha term. alpha_error separately counts any
//! alpha-byte mismatch (opaque black vs transparent black has DeltaE=0).
//! p95 uses the nearest-rank percentile, ceil(.95*N)-1.
//!
//! Palette metrics ignore transparent RGB and alpha differences. "Output palette"
//! is the engine's declared palette, including unused entries; size_used also
//! reports actual nontransparent RGB classes. Recall matches each truth color
//! independently within PALETTE_EPS; it is not a one-to-one assignment. Spurious
//! counts declared entries farther than PALETTE_EPS from every truth color.
//! Area-weighted recall uses truth frequencies; weighted spurious uses actual
//! output frequencies (unused entries have zero weight). No approximate NN index.
//!
//! Boundaries mark both endpoints of every 4-neighbour unequal RGBA pair.
//! Precision/recall independently allow Manhattan radius 1; matches need not be
//! bijective. Empty/empty gives F1=1, only one empty gives 0. Thus this metric
//! tolerates translations and can over-credit dense boundary maps.
//!
//! Features are same-RGBA 8-connected components of area 1 or 2 (including holes).
//! A feature is recalled only if EVERY cell matches exact RGBA at the exact
//! position. flat_mottling pools 8-connected truth components of area >=12:
//! count cells differing from that component's output RGBA majority / total area.
//!
//! Manufactured transition: center and neighbours must be opaque. Among distinct
//! 8-neighbour color pairs, separation >=.04, project center onto their Oklab
//! segment; require t in [.1,.9], distance <=.015, and distance to both endpoints
//! >=.02. Count output hits only where the analogous truth-neighbour predicate
//! is false. Divide by all cells. This is a local heuristic, NOT proof that an
//! intermediate color is unintended; thin intentional ramps can be penalized.
//!
//! Dimension mismatch NEVER resamples: only geometry/palette metrics are emitted.
//! Aggregates are macro means of available per-case values, with explicit match
//! and case counts. No scalar "quality score"; real images have no truth scores.
use super::*;
use raster_to_pixel::color::{oklab_dist2, srgb8_to_oklab};
use std::collections::VecDeque;
const PALETTE_EPS: f32 = 0.02;
const COLS: [&str; 25] = [
    "truth_w",
    "truth_h",
    "output_w",
    "output_h",
    "grid_match",
    "grid_dx",
    "grid_dy",
    "scale_relative_error",
    "exact_accuracy",
    "deltae_mean",
    "deltae_p95",
    "alpha_error",
    "palette_size",
    "palette_size_used",
    "truth_palette_size",
    "truth_color_recall",
    "truth_color_recall_area",
    "spurious_entries",
    "spurious_fraction",
    "spurious_area",
    "boundary_f1",
    "small_feature_recall",
    "small_feature_count",
    "flat_mottling",
    "transition_rate",
];
type Pixel = [u8; 4];
fn classes(im: &RgbaImage) -> Vec<Pixel> {
    im.pixels()
        .map(|p| if p[3] == 0 { [0; 4] } else { p.0 })
        .collect()
}
fn lab(p: Pixel) -> [f32; 3] {
    srgb8_to_oklab(p[0], p[1], p[2])
}
fn labs(p: &[Pixel]) -> Vec<[f32; 3]> {
    let mut cache = BTreeMap::new();
    p.iter()
        .map(|&p| *cache.entry(p).or_insert_with(|| lab(p)))
        .collect()
}
fn rgb_counts(im: &RgbaImage) -> BTreeMap<[u8; 3], usize> {
    let mut m = BTreeMap::new();
    for p in im.pixels().filter(|p| p[3] > 0) {
        *m.entry([p[0], p[1], p[2]]).or_insert(0) += 1
    }
    m
}
fn ratio(n: usize, d: usize) -> J {
    if d == 0 {
        J::Null
    } else {
        (n as f64 / d as f64).into()
    }
}
fn nearest_distance(l: [f32; 3], palette: &[[f32; 3]]) -> f32 {
    palette
        .iter()
        .map(|&p| oklab_dist2(l, p))
        .fold(f32::INFINITY, f32::min)
        .sqrt()
}
fn boundary(p: &[Pixel], w: usize, h: usize) -> Vec<bool> {
    let mut b = vec![false; p.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w && p[i] != p[i + 1] {
                b[i] = true;
                b[i + 1] = true
            }
            if y + 1 < h && p[i] != p[i + w] {
                b[i] = true;
                b[i + w] = true
            }
        }
    }
    b
}
fn near(b: &[bool], i: usize, w: usize, h: usize) -> bool {
    b[i] || (i % w > 0 && b[i - 1])
        || (i % w + 1 < w && b[i + 1])
        || (i / w > 0 && b[i - w])
        || (i / w + 1 < h && b[i + w])
}
fn boundary_f1(a: &[Pixel], b: &[Pixel], w: usize, h: usize) -> f64 {
    let a = boundary(a, w, h);
    let b = boundary(b, w, h);
    let na = a.iter().filter(|&&v| v).count();
    let nb = b.iter().filter(|&&v| v).count();
    if na == 0 || nb == 0 {
        return if na == nb { 1. } else { 0. };
    }
    let recall = a
        .iter()
        .enumerate()
        .filter(|(i, v)| **v && near(&b, *i, w, h))
        .count() as f64
        / na as f64;
    let precision = b
        .iter()
        .enumerate()
        .filter(|(i, v)| **v && near(&a, *i, w, h))
        .count() as f64
        / nb as f64;
    if recall + precision == 0. {
        0.
    } else {
        2. * recall * precision / (recall + precision)
    }
}
fn neighbours(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let x = i % w;
    let y = i / w;
    let x0 = x.saturating_sub(1);
    let x1 = (x + 1).min(w - 1);
    let y0 = y.saturating_sub(1);
    let y1 = (y + 1).min(h - 1);
    (y0..=y1).flat_map(move |yy| {
        (x0..=x1).filter_map(move |xx| {
            let j = yy * w + xx;
            if i == j {
                None
            } else {
                Some(j)
            }
        })
    })
}
fn components(a: &[Pixel], b: &[Pixel], w: usize, h: usize) -> (J, J, usize) {
    let mut seen = vec![false; a.len()];
    let mut queue = VecDeque::new();
    let mut small = 0;
    let mut recalled = 0;
    let mut flat = 0;
    let mut mottle = 0;
    for i in 0..a.len() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        queue.push_back(i);
        let mut members = Vec::new();
        while let Some(j) = queue.pop_front() {
            members.push(j);
            for k in neighbours(j, w, h) {
                if !seen[k] && a[k] == a[i] {
                    seen[k] = true;
                    queue.push_back(k)
                }
            }
        }
        if members.len() <= 2 {
            small += 1;
            if members.iter().all(|&j| a[j] == b[j]) {
                recalled += 1
            }
        }
        if members.len() >= 12 {
            let mut counts = BTreeMap::new();
            for &j in &members {
                *counts.entry(b[j]).or_insert(0usize) += 1
            }
            flat += members.len();
            mottle += members.len() - counts.values().max().unwrap();
        }
    }
    (ratio(recalled, small), ratio(mottle, flat), small)
}
fn intermediate(i: usize, p: &[Pixel], l: &[[f32; 3]], w: usize, h: usize) -> bool {
    if p[i][3] != 255 {
        return false;
    }
    let mut ns = Vec::with_capacity(8);
    for j in neighbours(i, w, h) {
        if p[j][3] == 255 && !ns.iter().any(|&k| p[k] == p[j]) {
            ns.push(j)
        }
    }
    for (offset, &a) in ns.iter().enumerate() {
        for &b in &ns[offset + 1..] {
            let a = l[a];
            let b = l[b];
            let c = l[i];
            let len2 = oklab_dist2(a, b);
            if len2 < 0.04 * 0.04
                || oklab_dist2(c, a) < 0.02 * 0.02
                || oklab_dist2(c, b) < 0.02 * 0.02
            {
                continue;
            }
            let t = ((c[0] - a[0]) * (b[0] - a[0])
                + (c[1] - a[1]) * (b[1] - a[1])
                + (c[2] - a[2]) * (b[2] - a[2]))
                / len2;
            if !(0.1..=0.9).contains(&t) {
                continue;
            }
            let projection = std::array::from_fn(|ch| a[ch] + t * (b[ch] - a[ch]));
            if oklab_dist2(c, projection) <= 0.015 * 0.015 {
                return true;
            }
        }
    }
    false
}
fn calculate(truth: &RgbaImage, out: &RgbaImage, palette: &[[u8; 3]]) -> J {
    let mut m = obj(COLS.iter().map(|&c| (c, J::Null)));
    let (tw, th) = truth.dimensions();
    let (ow, oh) = out.dimensions();
    for (k, v) in [
        ("truth_w", tw as f64),
        ("truth_h", th as f64),
        ("output_w", ow as f64),
        ("output_h", oh as f64),
        ("grid_dx", ow as f64 - tw as f64),
        ("grid_dy", oh as f64 - th as f64),
        ("grid_match", if (tw, th) == (ow, oh) { 1. } else { 0. }),
    ] {
        m.set(k, v)
    }
    let tc = rgb_counts(truth);
    let oc = rgb_counts(out);
    let pl: Vec<_> = palette
        .iter()
        .map(|p| srgb8_to_oklab(p[0], p[1], p[2]))
        .collect();
    let tl: Vec<_> = tc
        .keys()
        .map(|p| srgb8_to_oklab(p[0], p[1], p[2]))
        .collect();
    let hits: Vec<_> = tl
        .iter()
        .map(|&l| nearest_distance(l, &pl) <= PALETTE_EPS)
        .collect();
    m.set("palette_size", palette.len());
    m.set("palette_size_used", oc.len());
    m.set("truth_palette_size", tc.len());
    m.set(
        "truth_color_recall",
        ratio(hits.iter().filter(|&&b| b).count(), tc.len()),
    );
    let area: usize = tc.values().sum();
    let matched_area: usize = tc
        .values()
        .zip(&hits)
        .filter(|(_, b)| **b)
        .map(|(&n, _)| n)
        .sum();
    m.set("truth_color_recall_area", ratio(matched_area, area));
    let spurious = pl
        .iter()
        .filter(|&&p| nearest_distance(p, &tl) > PALETTE_EPS)
        .count();
    m.set("spurious_entries", spurious);
    m.set("spurious_fraction", ratio(spurious, palette.len()));
    let spurious_area: usize = oc
        .iter()
        .filter(|(p, _)| nearest_distance(srgb8_to_oklab(p[0], p[1], p[2]), &tl) > PALETTE_EPS)
        .map(|(_, n)| n)
        .sum();
    m.set("spurious_area", ratio(spurious_area, oc.values().sum()));
    if (tw, th) != (ow, oh) {
        return m;
    }
    let a = classes(truth);
    let b = classes(out);
    let al = labs(&a);
    let bl = labs(&b);
    let count = a.len();
    let w = tw as usize;
    let h = th as usize;
    m.set(
        "exact_accuracy",
        ratio(a.iter().zip(&b).filter(|(a, b)| a == b).count(), count),
    );
    m.set(
        "alpha_error",
        ratio(
            a.iter().zip(&b).filter(|(a, b)| a[3] != b[3]).count(),
            count,
        ),
    );
    let mut deltas: Vec<f32> = al
        .iter()
        .zip(&bl)
        .map(|(&a, &b)| oklab_dist2(a, b).sqrt())
        .collect();
    m.set(
        "deltae_mean",
        deltas.iter().map(|&x| x as f64).sum::<f64>() / count as f64,
    );
    deltas.sort_by(f32::total_cmp);
    m.set(
        "deltae_p95",
        deltas[((count as f64 * 0.95).ceil() as usize).saturating_sub(1)] as f64,
    );
    m.set("boundary_f1", boundary_f1(&a, &b, w, h));
    let (small, mottle, n) = components(&a, &b, w, h);
    m.set("small_feature_recall", small);
    m.set("small_feature_count", n);
    m.set("flat_mottling", mottle);
    let manufactured = (0..count)
        .filter(|&i| intermediate(i, &b, &bl, w, h) && !intermediate(i, &a, &al, w, h))
        .count();
    m.set("transition_rate", ratio(manufactured, count));
    m
}
pub(super) fn score(args: &Args) -> E<()> {
    let label = label(args.need("--label")?)?;
    let root = PathBuf::from(OUT).join("runs").join(label);
    let start = Instant::now();
    let mut found = false;
    let mut all_summaries = String::new();
    for mode in MODES {
        let dir = root.join(mode);
        if !dir.join("summary.json").exists() {
            continue;
        }
        found = true;
        let summary = read_json(dir.join("summary.json"))?;
        let mut rows = BTreeMap::new();
        let mut csv = format!(
            "id,status,split,family,tags,source_hash,truth_hash,runtime_ms,{}\n",
            COLS.join(",")
        );
        for id in summary.get("case_ids").a() {
            let record = read_json(dir.join(format!("{}.json", id.s())))?;
            let case = record.get("case");
            let mut row = obj([
                ("case", case.clone()),
                ("status", record.get("status").clone()),
                ("runtime_ms", record.get("runtime_ms").clone()),
            ]);
            if record.get("status").s() == "ok" && !case.get("truth").s().is_empty() {
                let truth = load(case.get("truth").s())?;
                if image_hash(&truth) != case.get("truth_hash").s() {
                    return Err(
                        format!("truth changed for {}; regenerate corpus/run", id.s()).into(),
                    );
                }
                let out = load(record.get("output").s())?;
                if image_hash(&out) != record.get("output_hash").s() {
                    return Err(format!("output changed for {}", id.s()).into());
                }
                let mut metrics = calculate(&truth, &out, &parse_palette(record.get("palette")));
                let estimated = record
                    .get("diagnostics")
                    .get("detectedPixelSize")
                    .n()
                    .or_else(|| record.get("resolved_config").get("pixel_size").n());
                if let (Some(estimate), Some(scale)) = (estimated, case.get("scale").n()) {
                    metrics.set("scale_relative_error", estimate / scale - 1.)
                }
                row.set("metrics", metrics);
            }
            let tags = case
                .get("tags")
                .a()
                .iter()
                .map(J::s)
                .collect::<Vec<_>>()
                .join(";");
            let mut fields = vec![
                id.s().to_string(),
                record.get("status").s().to_string(),
                case.get("split").s().into(),
                case.get("family").s().into(),
                tags,
                case.get("source_hash").s().into(),
                case.get("truth_hash").s().into(),
                number(record.get("runtime_ms")),
            ];
            fields.extend(COLS.iter().map(|k| number(row.get("metrics").get(k))));
            csv.push_str(
                &fields
                    .iter()
                    .map(|s| csv_escape(s))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            csv.push('\n');
            rows.insert(id.s().to_string(), row);
        }
        write(dir.join("metrics.csv"), csv.as_bytes())?;
        let mut score_index = J::Obj(rows.clone());
        score_index.set(
            "_run_summary_hash",
            format!("{:016x}", hash(summary.dump().as_bytes())),
        );
        score_index.set(
            "_metric_definitions_hash",
            format!("{:016x}", hash(include_bytes!("metrics.rs"))),
        );
        write_json(dir.join("scores.json"), &score_index)?;
        let mut groups: BTreeMap<String, Vec<&J>> = BTreeMap::new();
        for row in rows.values() {
            if row.get("case").get("truth").s().is_empty() {
                continue;
            }
            groups.entry("overall".into()).or_default().push(row);
            groups
                .entry(format!("family:{}", row.get("case").get("family").s()))
                .or_default()
                .push(row);
            for tag in row.get("case").get("tags").a() {
                groups
                    .entry(format!("tag:{}", tag.s()))
                    .or_default()
                    .push(row)
            }
        }
        let mut aggregate = format!("group,cases,grid_matches,errors,{}\n", COLS.join(","));
        let mut report = format!(
            "\n{label}/{mode} | commit {} | engine dirty {}\n",
            summary.get("engine").get("commit").s(),
            summary.get("engine").get("engine_dirty").dump()
        );
        report.push_str("| group | cases | grid matches | accuracy | dE mean | color recall | boundary F1 | feature recall |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for (group, rs) in groups {
            let matches = rs
                .iter()
                .filter(|r| r.get("metrics").get("grid_match").n() == Some(1.))
                .count();
            let errors = rs.iter().filter(|r| r.get("status").s() != "ok").count();
            let mean = |k: &str| {
                let nums: Vec<_> = rs
                    .iter()
                    .filter_map(|r| r.get("metrics").get(k).n())
                    .collect();
                if nums.is_empty() {
                    None
                } else {
                    Some(nums.iter().sum::<f64>() / nums.len() as f64)
                }
            };
            let mut fields = vec![
                group.clone(),
                rs.len().to_string(),
                matches.to_string(),
                errors.to_string(),
            ];
            fields.extend(
                COLS.iter()
                    .map(|k| mean(k).map(|v| format!("{v:.8}")).unwrap_or_default()),
            );
            aggregate.push_str(
                &fields
                    .iter()
                    .map(|s| csv_escape(s))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            aggregate.push('\n');
            report.push_str(&format!(
                "| {group} | {} | {matches} | {} | {} | {} | {} | {} |\n",
                rs.len(),
                fmt(mean("exact_accuracy")),
                fmt(mean("deltae_mean")),
                fmt(mean("truth_color_recall")),
                fmt(mean("boundary_f1")),
                fmt(mean("small_feature_recall"))
            ));
        }
        report.push_str("Macro means over available values; pixel metrics omit dimension mismatches. Real inputs have diagnostics only.\n");
        write(dir.join("aggregates.csv"), aggregate.as_bytes())?;
        write(dir.join("score_summary.txt"), report.as_bytes())?;
        print!("{report}");
        all_summaries.push_str(&report);
    }
    if !found {
        return Err("no completed run modes for label".into());
    }
    write(root.join("scores.txt"), all_summaries.as_bytes())?;
    println!("score wall: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
fn number(j: &J) -> String {
    j.n().map(|v| format!("{v:.8}")).unwrap_or_default()
}
fn fmt(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into())
}
fn csv_escape(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unused_palette_entry_has_no_area_weight() {
        let im = RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 255, 255]));
        let m = calculate(&im, &im, &[[0, 0, 255], [255, 0, 0]]);
        assert_eq!(m.get("spurious_entries").n(), Some(1.));
        assert_eq!(m.get("spurious_fraction").n(), Some(0.5));
        assert_eq!(m.get("spurious_area").n(), Some(0.));
    }
    #[test]
    fn transparent_hole_counts_as_small_feature() {
        let mut im = RgbaImage::from_pixel(5, 5, image::Rgba([100, 100, 100, 255]));
        im.put_pixel(2, 2, image::Rgba([0, 0, 0, 0]));
        let out = RgbaImage::from_pixel(5, 5, image::Rgba([100, 100, 100, 255]));
        let m = calculate(&im, &out, &rgb_palette(&out));
        assert_eq!(m.get("small_feature_count").n(), Some(1.));
        assert_eq!(m.get("small_feature_recall").n(), Some(0.));
    }
    #[test]
    fn identity_and_mismatch_are_not_resampled() {
        let im = RgbaImage::from_fn(5, 5, |x, y| {
            image::Rgba(if x == 2 && y == 2 {
                [250, 20, 50, 255]
            } else {
                [10, 30, 80, 255]
            })
        });
        let m = calculate(&im, &im, &rgb_palette(&im));
        for k in [
            "exact_accuracy",
            "boundary_f1",
            "small_feature_recall",
            "truth_color_recall",
        ] {
            assert_eq!(m.get(k).n(), Some(1.))
        }
        for k in [
            "deltae_mean",
            "flat_mottling",
            "transition_rate",
            "spurious_area",
        ] {
            assert_eq!(m.get(k).n(), Some(0.))
        }
        let m = calculate(&im, &RgbaImage::new(4, 5), &[]);
        assert_eq!(m.get("grid_match").n(), Some(0.));
        assert_eq!(m.get("exact_accuracy"), &J::Null);
    }
    #[test]
    fn rare_accent_is_not_area_weighted_away() {
        let truth = RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba(if x == 0 && y == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            })
        });
        let out = RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 255, 255]));
        let m = calculate(&truth, &out, &[[0, 0, 255]]);
        assert_eq!(m.get("truth_color_recall").n(), Some(0.5));
        assert_eq!(m.get("truth_color_recall_area").n(), Some(63. / 64.));
        assert_eq!(m.get("small_feature_recall").n(), Some(0.));
    }
    #[test]
    fn invisible_rgb_is_one_class() {
        let a = RgbaImage::from_pixel(3, 3, image::Rgba([255, 0, 80, 0]));
        let b = RgbaImage::new(3, 3);
        assert_eq!(calculate(&a, &b, &[]).get("exact_accuracy").n(), Some(1.));
    }
    #[test]
    fn blend_predicate_detects_midpoint() {
        let mut im = RgbaImage::from_pixel(3, 3, image::Rgba([255, 255, 255, 255]));
        im.put_pixel(0, 1, image::Rgba([0, 0, 0, 255]));
        im.put_pixel(1, 1, image::Rgba([99, 99, 99, 255]));
        let c = classes(&im);
        assert!(intermediate(4, &c, &labs(&c), 3, 3));
    }
}
