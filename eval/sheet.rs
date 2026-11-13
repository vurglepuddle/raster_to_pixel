//! Static, offline HTML + nearest-neighbour PNG montages; no font renderer.
use super::*;
use image::{
    imageops::{self, FilterType},
    Rgba,
};
const TILE: u32 = 288;
fn html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
fn tile(im: &RgbaImage, region: Option<[f64; 4]>) -> RgbaImage {
    let cropped = region.map(|r| {
        let x = (r[0] * im.width() as f64)
            .floor()
            .min((im.width() - 1) as f64) as u32;
        let y = (r[1] * im.height() as f64)
            .floor()
            .min((im.height() - 1) as f64) as u32;
        let w = (r[2] * im.width() as f64).ceil().max(1.) as u32;
        let h = (r[3] * im.height() as f64).ceil().max(1.) as u32;
        imageops::crop_imm(im, x, y, w.min(im.width() - x), h.min(im.height() - y)).to_image()
    });
    let im = cropped.as_ref().unwrap_or(im);
    let scale = (TILE as f64 / im.width() as f64).min(TILE as f64 / im.height() as f64);
    // Fit every panel to the same physical canvas; use nearest even at
    // noninteger presentation scales so source/truth are equally enlarged.
    let small = imageops::resize(
        im,
        (im.width() as f64 * scale).round().max(1.) as u32,
        (im.height() as f64 * scale).round().max(1.) as u32,
        FilterType::Nearest,
    );
    let mut out = RgbaImage::from_fn(TILE, TILE, |x, y| {
        if (x / 8 + y / 8) % 2 == 0 {
            Rgba([52, 55, 60, 255])
        } else {
            Rgba([76, 80, 86, 255])
        }
    });
    imageops::overlay(
        &mut out,
        &small,
        ((TILE - small.width()) / 2) as i64,
        ((TILE - small.height()) / 2) as i64,
    );
    out
}
fn montage(images: &[RgbaImage], region: Option<[f64; 4]>) -> RgbaImage {
    let mut out = RgbaImage::new(TILE * images.len() as u32, TILE);
    for (i, im) in images.iter().enumerate() {
        imageops::replace(&mut out, &tile(im, region), (i as u32 * TILE) as i64, 0)
    }
    out
}
fn changed_region(outputs: &[RgbaImage]) -> [f64; 4] {
    // Compare in the first output's logical grid. Other dimensions are mapped
    // by normalized coordinates ONLY for visual crop selection, never scoring.
    let first = &outputs[0];
    let w = first.width() as usize;
    let h = first.height() as usize;
    let mut sat = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.;
        for x in 0..w {
            let a = first.get_pixel(x as u32, y as u32);
            let mut largest = 0f64;
            for other in &outputs[1..] {
                let xx = ((x as u64 * other.width() as u64) / w as u64) as u32;
                let yy = ((y as u64 * other.height() as u64) / h as u64) as u32;
                let b = other.get_pixel(xx, yy);
                let d = (0..4)
                    .map(|ch| {
                        let aa = if a[3] == 0 && ch < 3 { 0 } else { a[ch] };
                        let bb = if b[3] == 0 && ch < 3 { 0 } else { b[ch] };
                        (aa as f64 - bb as f64).abs()
                    })
                    .sum::<f64>();
                largest = largest.max(d);
            }
            row += largest;
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let size = (w.min(h) / 5).clamp(4, 24);
    let cw = size.min(w);
    let ch = size.min(h);
    let mut best = (-1., 0, 0);
    for y in 0..=h - ch {
        for x in 0..=w - cw {
            let sum = sat[(y + ch) * (w + 1) + x + cw]
                - sat[y * (w + 1) + x + cw]
                - sat[(y + ch) * (w + 1) + x]
                + sat[y * (w + 1) + x];
            if sum > best.0 {
                best = (sum, x, y)
            }
        }
    }
    [
        best.1 as f64 / w as f64,
        best.2 as f64 / h as f64,
        cw as f64 / w as f64,
        ch as f64 / h as f64,
    ]
}
fn change(scores: &[J], id: &str) -> f64 {
    let first = scores[0].get(id).get("metrics");
    let mut worst = 0f64;
    let mut best = 0f64;
    for next in &scores[1..] {
        let b = next.get(id).get("metrics");
        for (k, weight) in [
            ("grid_match", 4.),
            ("exact_accuracy", 1.),
            ("truth_color_recall", 0.5),
            ("boundary_f1", 0.25),
            ("small_feature_recall", 0.25),
        ] {
            if let (Some(a), Some(b)) = (first.get(k).n(), b.get(k).n()) {
                let d = (b - a) * weight;
                worst = worst.min(d);
                best = best.max(d);
            }
        }
        for (k, weight) in [
            ("deltae_mean", 1.),
            ("flat_mottling", 0.25),
            ("transition_rate", 0.25),
        ] {
            if let (Some(a), Some(b)) = (first.get(k).n(), b.get(k).n()) {
                let d = -(b - a) * weight;
                worst = worst.min(d);
                best = best.max(d);
            }
        }
    }
    if worst < 0. {
        worst
    } else {
        best
    }
}
pub(super) fn sheet(args: &Args) -> E<()> {
    let start = Instant::now();
    let labels: Vec<_> = args
        .need("--labels")?
        .split(',')
        .map(label)
        .collect::<E<_>>()?;
    if labels.len() < 2 || labels.len() > 3 {
        return Err("sheet needs two or three comma-separated labels".into());
    }
    let mode = args.get("--mode", "auto");
    if !MODES.contains(&mode) {
        return Err("sheet: invalid mode".into());
    }
    let mut records: Vec<BTreeMap<String, J>> = Vec::new();
    let mut scores = Vec::new();
    let mut stamps = Vec::new();
    let mut ids = BTreeSet::new();
    for label in &labels {
        let dir = PathBuf::from(OUT).join("runs").join(label).join(mode);
        let summary = read_json(dir.join("summary.json"))?;
        let score_path = dir.join("scores.json");
        if !score_path.exists() {
            return Err(format!("score --label {label} first").into());
        }
        let score_index = read_json(score_path)?;
        if score_index.get("_run_summary_hash").s()
            != format!("{:016x}", hash(summary.dump().as_bytes()))
        {
            return Err(format!("run {label}/{mode} changed since scoring; score it again").into());
        }
        if score_index.get("_metric_definitions_hash").s()
            != format!("{:016x}", hash(include_bytes!("metrics.rs")))
        {
            return Err(format!("metric definitions changed; score --label {label} again").into());
        }
        scores.push(score_index);
        stamps.push(summary.get("engine").clone());
        let mut map = BTreeMap::new();
        for id in summary.get("case_ids").a() {
            let record = read_json(dir.join(format!("{}.json", id.s())))?;
            if args.matches(record.get("case")) {
                ids.insert(id.s().to_string());
                map.insert(id.s().to_string(), record);
            }
        }
        records.push(map);
    }
    let mut ordered: Vec<_> = ids.into_iter().collect();
    ordered.sort_by(|a, b| {
        let is_real = |id: &str| {
            records
                .iter()
                .find_map(|m| m.get(id))
                .is_some_and(|r| r.get("case").get("truth").s().is_empty())
        };
        is_real(a)
            .cmp(&is_real(b))
            .then_with(|| {
                let rank = |id: &str| {
                    let c = change(&scores, id);
                    if c < 0. {
                        (0, c)
                    } else if c > 0. {
                        (1, -c)
                    } else {
                        (2, 0.)
                    }
                };
                let aa = rank(a);
                let bb = rank(b);
                aa.0.cmp(&bb.0).then_with(|| aa.1.total_cmp(&bb.1))
            })
            .then(a.cmp(b))
    });
    let name = format!("REVIEW_{}", labels.join("_vs_"));
    let directory = PathBuf::from(OUT).join("sheets").join(&name).join(mode);
    fs::create_dir_all(&directory)?;
    let mut page=String::from("<!doctype html><meta charset=\"utf-8\"><title>Pixeline evaluation</title><style>body{background:#181b20;color:#e6e8eb;font:14px system-ui;margin:24px}article{border-top:1px solid #666;margin-top:24px;padding-top:12px}img{max-width:100%;image-rendering:pixelated}table{border-collapse:collapse}td,th{border:1px solid #555;padding:5px 9px}code{overflow-wrap:anywhere}.warn{color:#ffbf69}</style>");
    page.push_str(&format!("<h1>{} — {}</h1><p>Synthetic regressions first, then improvements, unchanged cases, then real cases. Ordering uses the largest weighted adverse metric delta (gains cannot cancel a regression), then largest favorable delta; it is only a review aid. Numbers below retain separate metrics. PNG panels use nearest-neighbour sampling and equal 288 px canvases; checkerboard reveals alpha. Large inputs may be reduced for presentation.</p>",html(&labels.join(" vs ")),html(mode)));
    for (label, stamp) in labels.iter().zip(stamps) {
        page.push_str(&format!(
            "<p><b>{}</b>: commit {} · engine dirty {} · content {}</p>",
            html(label),
            html(stamp.get("commit").s()),
            stamp.get("engine_dirty").dump(),
            html(stamp.get("content_hash_fnv1a64").s())
        ));
    }
    let mut count = 0;
    for id in ordered {
        let reference = records.iter().find_map(|m| m.get(&id)).unwrap();
        let case = reference.get("case");
        page.push_str(&format!(
            "<article><h2>{}</h2><p>{} · {} · tags {}</p>",
            html(&id),
            html(case.get("split").s()),
            html(case.get("family").s()),
            html(&case.get("tags").dump())
        ));
        if records
            .iter()
            .any(|m| m.get(&id).is_none_or(|r| r.get("status").s() != "ok"))
        {
            page.push_str("<p class=\"warn\">Missing or failed case in at least one label; no comparable montage.</p></article>");
            continue;
        }
        for map in &records {
            let other = map.get(&id).unwrap().get("case");
            if other.get("source_hash") != case.get("source_hash")
                || other.get("truth_hash") != case.get("truth_hash")
            {
                return Err(
                    format!("{id}: corpus differs between labels; comparison refused").into(),
                );
            }
        }
        let source = load(case.get("source").s())?;
        if image_hash(&source) != case.get("source_hash").s() {
            return Err(format!("{id}: source changed").into());
        }
        let mut images = vec![source];
        let has_truth = !case.get("truth").s().is_empty();
        if has_truth {
            let truth = load(case.get("truth").s())?;
            if image_hash(&truth) != case.get("truth_hash").s() {
                return Err(format!("{id}: truth changed").into());
            }
            images.push(truth)
        }
        let mut outputs = Vec::new();
        for map in &records {
            let record = map.get(&id).unwrap();
            let im = load(record.get("output").s())?;
            if image_hash(&im) != record.get("output_hash").s() {
                return Err(format!("{id}: output changed").into());
            }
            outputs.push(im)
        }
        images.extend(outputs.iter().cloned());
        let png = directory.join(format!("{id}.png"));
        montage(&images, None).save(&png)?;
        let headings = std::iter::once("source".to_string())
            .chain(if has_truth {
                vec!["truth".into()]
            } else {
                vec![]
            })
            .chain(labels.iter().map(|s| s.to_string()))
            .collect::<Vec<_>>();
        page.push_str(&format!(
            "<p>{}</p><img loading=\"lazy\" src=\"{}\" alt=\"{}\">",
            html(&headings.join(" | ")),
            html(&pathstr(png.strip_prefix(OUT)?)),
            html(&id)
        ));
        if !has_truth {
            let region = changed_region(&outputs);
            let zoom = directory.join(format!("{id}-zoom.png"));
            montage(&images, Some(region)).save(&zoom)?;
            page.push_str(&format!("<p>Most-changed region (normalized x/y/w/h {:?}); differing output grids use normalized coordinates only for crop selection.</p><img loading=\"lazy\" src=\"{}\" alt=\"zoom\">",region,html(&pathstr(zoom.strip_prefix(OUT)?))));
        }
        page.push_str("<table><tr><th>label</th><th>grid</th><th>ms</th><th>palette</th><th>accuracy (Δ)</th><th>dE mean (Δ)</th><th>color recall (Δ)</th><th>boundary F1 (Δ)</th><th>feature recall (Δ)</th><th>mottling (Δ)</th><th>transitions (Δ)</th></tr>");
        let first = scores[0].get(&id).get("metrics");
        for (i, label) in labels.iter().enumerate() {
            let r = records[i].get(&id).unwrap();
            let d = r.get("diagnostics");
            let m = scores[i].get(&id).get("metrics");
            page.push_str(&format!(
                "<tr><th>{}</th><td>{}×{}</td><td>{:.1}</td><td>{}</td>",
                html(label),
                d.get("outWidth").dump(),
                d.get("outHeight").dump(),
                r.get("runtime_ms").n().unwrap_or(0.),
                d.get("paletteLen").dump()
            ));
            for k in [
                "exact_accuracy",
                "deltae_mean",
                "truth_color_recall",
                "boundary_f1",
                "small_feature_recall",
                "flat_mottling",
                "transition_rate",
            ] {
                let value = m
                    .get(k)
                    .n()
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "—".into());
                let delta = match (first.get(k).n(), m.get(k).n()) {
                    (Some(a), Some(b)) if i > 0 => format!(" ({:+.4})", b - a),
                    _ => String::new(),
                };
                page.push_str(&format!("<td>{value}{delta}</td>"));
            }
            page.push_str("</tr>");
        }
        page.push_str("</table></article>");
        count += 1;
    }
    let path = PathBuf::from(OUT).join(format!("{name}.html"));
    write(&path, page.as_bytes())?;
    println!(
        "sheet: {count} montages in {:.3}s; {}",
        start.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zoom_finds_changed_corner() {
        let a = RgbaImage::from_pixel(20, 20, Rgba([0, 0, 0, 255]));
        let mut b = a.clone();
        for y in 16..20 {
            for x in 16..20 {
                b.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        assert_eq!(changed_region(&[a, b]), [0.8, 0.8, 0.2, 0.2]);
    }
}
