use super::*;
use image::{
    imageops::{self, FilterType},
    Rgba,
};
use std::io::Cursor;

/// Whitespace fields, optional quoted paths, and # comments outside quotes.
fn tokens(line: &str) -> E<Vec<String>> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    for c in line.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None
            } else {
                word.push(c)
            }
        } else {
            match c {
                '#' => break,
                '"' | '\'' => quote = Some(c),
                c if c.is_whitespace() => {
                    if !word.is_empty() {
                        out.push(std::mem::take(&mut word))
                    }
                }
                _ => word.push(c),
            }
        }
    }
    if quote.is_some() {
        return Err("unclosed manifest quote".into());
    }
    if !word.is_empty() {
        out.push(word)
    }
    Ok(out)
}
#[derive(Clone)]
struct Truth {
    path: PathBuf,
    crop: [u32; 4],
    tags: Vec<String>,
    native: u32,
    core: bool,
}
fn truth_manifest(path: &Path) -> E<Vec<Truth>> {
    let mut out = Vec::new();
    for (i, line) in fs::read_to_string(path)?.lines().enumerate() {
        let t = tokens(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if t.is_empty() {
            continue;
        }
        if t.len() < 6 {
            return Err(format!(
                "{}:{}: need path x y w h tags [native=k] [variants=core|full]",
                path.display(),
                i + 1
            )
            .into());
        }
        let mut tr = Truth {
            path: PathBuf::from(&t[0]),
            crop: [t[1].parse()?, t[2].parse()?, t[3].parse()?, t[4].parse()?],
            tags: t[5].split(',').map(String::from).collect(),
            native: 1,
            core: false,
        };
        for opt in &t[6..] {
            if let Some(n) = opt.strip_prefix("native=") {
                tr.native = n.parse()?;
                if tr.native == 0 {
                    return Err("native must be >=1".into());
                }
            } else if let Some(v) = opt.strip_prefix("variants=") {
                match v {
                    "core" => tr.core = true,
                    "full" => tr.core = false,
                    _ => return Err("variants must be core or full".into()),
                }
            } else {
                return Err(format!("unknown manifest token {opt}").into());
            }
        }
        out.push(tr);
    }
    Ok(out)
}
fn native_image(tr: &Truth) -> E<RgbaImage> {
    let mut im = load(&tr.path)?;
    if tr.native > 1 {
        let k = tr.native;
        if im.width() % k != 0 || im.height() % k != 0 {
            return Err(format!(
                "{}: native={k} does not divide dimensions",
                tr.path.display()
            )
            .into());
        }
        let small = imageops::resize(&im, im.width() / k, im.height() / k, FilterType::Nearest);
        // Byte-exact RGBA verification, including hidden RGB under alpha=0.
        if imageops::resize(&small, im.width(), im.height(), FilterType::Nearest) != im {
            return Err(format!(
                "{}: native={k} is not an exact nearest upscale",
                tr.path.display()
            )
            .into());
        }
        im = small;
    }
    let [x, y, w, h] = tr.crop;
    if w == 0
        || h == 0
        || x.checked_add(w).is_none_or(|r| r > im.width())
        || y.checked_add(h).is_none_or(|r| r > im.height())
    {
        return Err(format!("{}: crop outside native art dimensions", tr.path.display()).into());
    }
    Ok(imageops::crop_imm(&im, x, y, w, h).to_image())
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / 9007199254740992.
    }
    // Deterministic polar Box-Muller, with no platform random source.
    fn gaussian(&mut self) -> f64 {
        loop {
            let x = 2. * self.unit() - 1.;
            let y = 2. * self.unit() - 1.;
            let r = x * x + y * y;
            if r > 0. && r < 1. {
                return x * (-2. * r.ln() / r).sqrt();
            }
        }
    }
}
#[derive(Clone)]
struct Variant {
    name: String,
    family: &'static str,
    scale: f64,
    filter: FilterType,
    blur: f32,
    unsharp: bool,
    jpeg: u8,
    warp: f64,
    noise: f64,
    flat: bool,
    phase: bool,
}
fn variants(core: bool, alpha: bool) -> Vec<Variant> {
    let v = |name: &str, family, scale, filter| Variant {
        name: name.into(),
        family,
        scale,
        filter,
        blur: 0.,
        unsharp: false,
        jpeg: 0,
        warp: 0.,
        noise: 0.,
        flat: false,
        phase: false,
    };
    let mut out = Vec::new();
    for s in [3, 4, 5, 8] {
        if !core || s == 5 {
            let mut q = v(
                &format!("nearest_{s}"),
                "nearest",
                s as f64,
                FilterType::Nearest,
            );
            q.phase = true;
            out.push(q)
        }
    }
    for (s, f, n) in [
        (4.37, FilterType::Triangle, "bilinear"),
        (5.6, FilterType::CatmullRom, "catmull"),
        (6.25, FilterType::Lanczos3, "lanczos"),
    ] {
        if !core || n == "catmull" {
            out.push(v(&format!("{n}_{s}"), "resample", s, f))
        }
    }
    for b in if core { vec![0.5] } else { vec![0.3, 0.6, 0.8] } {
        let mut q = v(&format!("blur_5_{b}"), "blur", 5., FilterType::Nearest);
        q.blur = b * 5.;
        out.push(q);
    }
    if !core {
        let mut q = v("unsharp_5", "unsharp", 5., FilterType::CatmullRom);
        q.unsharp = true;
        out.push(q)
    }
    for quality in if core { vec![90] } else { vec![70, 90] } {
        let mut q = v(
            &format!("jpeg_5_q{quality}"),
            "jpeg",
            5.,
            FilterType::Nearest,
        );
        q.jpeg = quality;
        out.push(q)
    }
    if !core {
        for amp in [0.2, 0.6] {
            let mut q = v(&format!("warp_5_{amp}"), "warp", 5., FilterType::CatmullRom);
            q.warp = amp;
            out.push(q)
        }
        let mut q = v("noise_5", "noise", 5., FilterType::Nearest);
        q.noise = 4.;
        out.push(q);
    }
    let mut q = v(
        "combo_lanczos_5.6_warp_jpeg85",
        "combination",
        5.6,
        FilterType::Lanczos3,
    );
    q.warp = 0.35;
    q.jpeg = 85;
    out.push(q);
    if alpha {
        if !core {
            out.push(v(
                "alpha_naive_lanczos_5.6",
                "alpha",
                5.6,
                FilterType::Lanczos3,
            ))
        }
        let mut q = v("alpha_flat_5", "alpha", 5., FilterType::CatmullRom);
        q.flat = true;
        out.push(q);
    }
    out
}
fn composite(im: &RgbaImage) -> RgbaImage {
    let bg = [89u8, 119, 151];
    RgbaImage::from_fn(im.width(), im.height(), |x, y| {
        let p = im.get_pixel(x, y);
        let a = p[3] as u32;
        Rgba([
            ((p[0] as u32 * a + bg[0] as u32 * (255 - a) + 127) / 255) as u8,
            ((p[1] as u32 * a + bg[1] as u32 * (255 - a) + 127) / 255) as u8,
            ((p[2] as u32 * a + bg[2] as u32 * (255 - a) + 127) / 255) as u8,
            255,
        ])
    })
}
fn bilinear(im: &RgbaImage, x: f64, y: f64) -> Rgba<u8> {
    let x = x.clamp(0., (im.width() - 1) as f64);
    let y = y.clamp(0., (im.height() - 1) as f64);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(im.width() - 1);
    let y1 = (y0 + 1).min(im.height() - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    Rgba(std::array::from_fn(|i| {
        ((1. - fy)
            * ((1. - fx) * im.get_pixel(x0, y0)[i] as f64 + fx * im.get_pixel(x1, y0)[i] as f64)
            + fy * ((1. - fx) * im.get_pixel(x0, y1)[i] as f64
                + fx * im.get_pixel(x1, y1)[i] as f64))
            .round()
            .clamp(0., 255.) as u8
    }))
}
fn degrade_one(truth: &RgbaImage, v: &Variant, seed: u64) -> E<(RgbaImage, RgbaImage, J)> {
    let mut rng = Rng(seed);
    let mut target = if v.flat || v.jpeg > 0 {
        composite(truth)
    } else {
        truth.clone()
    };
    let mut crop_x = 0;
    let mut crop_y = 0;
    let w = (truth.width() as f64 * v.scale).ceil() as u32;
    let h = (truth.height() as f64 * v.scale).ceil() as u32;
    if w as u64 * h as u64 > 16_000_000 {
        return Err("variant exceeds 16 MP; crop truth smaller or use variants=core".into());
    }
    let mut im = if v.phase {
        let s = v.scale as u32;
        crop_x = (rng.next() % s as u64) as u32;
        crop_y = (rng.next() % s as u64) as u32;
        let phase_x = (s - crop_x) % s;
        let phase_y = (s - crop_y) % s;
        // Edge-extend the missing leading partial cell. Cropping the padded
        // upscale leaves phase_x/y source pixels before the first full art cell,
        // preserving the complete truth and exact oracle dimensions.
        RgbaImage::from_fn(w + phase_x, h + phase_y, |x, y| {
            let tx = ((x as i64 - phase_x as i64).div_euclid(s as i64))
                .clamp(0, truth.width() as i64 - 1) as u32;
            let ty = ((y as i64 - phase_y as i64).div_euclid(s as i64))
                .clamp(0, truth.height() as i64 - 1) as u32;
            *target.get_pixel(tx, ty)
        })
    } else {
        imageops::resize(&target, w, h, v.filter)
    };
    if v.blur > 0. {
        im = imageops::blur(&im, v.blur)
    }
    if v.unsharp {
        im = imageops::unsharpen(&im, (v.scale * 0.5) as f32, 4)
    }
    let wave_x = rng.unit() * std::f64::consts::TAU;
    let wave_y = rng.unit() * std::f64::consts::TAU;
    if v.warp > 0. {
        let amp = v.warp * v.scale;
        let (w, h) = im.dimensions();
        let xs: Vec<f64> = (0..h)
            .map(|y| amp * (std::f64::consts::TAU * 2.5 * y as f64 / h as f64 + wave_x).sin())
            .collect();
        let ys: Vec<f64> = (0..w)
            .map(|x| amp * (std::f64::consts::TAU * 2. * x as f64 / w as f64 + wave_y).sin())
            .collect();
        im = RgbaImage::from_fn(w, h, |x, y| {
            bilinear(&im, x as f64 + xs[y as usize], y as f64 + ys[x as usize])
        });
    }
    if v.noise > 0. {
        for p in im.pixels_mut() {
            for ch in 0..3 {
                p[ch] = (p[ch] as f64 + v.noise * rng.gaussian())
                    .round()
                    .clamp(0., 255.) as u8
            }
        }
    }
    if v.jpeg > 0 {
        let rgb = image::DynamicImage::ImageRgba8(im).to_rgb8();
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, v.jpeg).encode(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )?;
        // Canonical pure-Rust decoder: degraded.png freezes identical RGBA inputs
        // for native/WASM engine comparisons, independent of JPEG decoder choices.
        im = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()?
            .decode()?
            .to_rgba8();
    }
    // Canonicalize hidden RGB for truth classes, without affecting native=k verification.
    for p in target.pixels_mut() {
        if p[3] == 0 {
            *p = Rgba([0, 0, 0, 0])
        }
    }
    let params=obj([("seed_hex",format!("{seed:016x}").into()),("filter",format!("{:?}",v.filter).into()),
        ("blur_sigma_source_px",(v.blur as f64).into()),("unsharp_sigma_source_px",(if v.unsharp{v.scale*0.5}else{0.}).into()),
        ("unsharp_threshold",4usize.into()),("jpeg_quality",(v.jpeg as u32).into()),
        ("warp_amplitude_art_px",v.warp.into()),("warp_periods_x",2.5.into()),("warp_periods_y",2f64.into()),
        ("warp_phase_x",wave_x.into()),("warp_phase_y",wave_y.into()),("noise_sigma_srgb8",v.noise.into()),
        ("flat_background",J::Arr(vec![89usize.into(),119usize.into(),151usize.into()])),
        ("composited",(v.flat || v.jpeg>0).into()),("phase_crop_left",crop_x.into()),("phase_crop_top",crop_y.into()),
        ("oracle_geometry_note","nominal prewarp isotropic grid; fractional resize dimensions ceil to whole source pixels; effective x/y scales recorded".into()),
        ("alpha_resampling","straight RGBA, no premultiplication".into())]);
    Ok((im, target, params))
}
pub(super) fn degrade(args: &Args) -> E<()> {
    let start = Instant::now();
    let seed = args.get("--seed", "20261113").parse::<u64>()?;
    let manifest = Path::new(args.get("--truth-manifest", "eval/truth_manifest.txt"));
    let truths = truth_manifest(manifest)?;
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for tr in truths {
        let key = format!("{}:{:?}:native={}", pathstr(&tr.path), tr.crop, tr.native);
        if !args.get("--filter", "").is_empty()
            && !format!("{key} {:?}", tr.tags).contains(args.get("--filter", ""))
        {
            continue;
        }
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate truth crop: {key}").into());
        }
        let truth = native_image(&tr)?;
        let stem = tr.path.file_stem().unwrap_or_default().to_string_lossy();
        let id = format!("{}-{:016x}", slug(&stem), hash(key.as_bytes()));
        let has_alpha = truth.pixels().any(|p| p[3] < 255);
        for v in variants(tr.core, has_alpha) {
            let case_seed = seed ^ hash(format!("{id}/{}", v.name).as_bytes());
            let (im, target, params) = degrade_one(&truth, &v, case_seed)?;
            let dir = PathBuf::from(OUT).join("synth").join(&id).join(&v.name);
            fs::create_dir_all(&dir)?;
            im.save(dir.join("degraded.png"))?;
            target.save(dir.join("truth.png"))?;
            let step = v.scale as u32;
            let phase_x = (step - params.get("phase_crop_left").n().unwrap_or(0.) as u32) % step;
            let phase_y = (step - params.get("phase_crop_top").n().unwrap_or(0.) as u32) % step;
            let case = obj([
                ("id", format!("{id}--{}", v.name).into()),
                ("source", pathstr(dir.join("degraded.png")).into()),
                ("truth", pathstr(dir.join("truth.png")).into()),
                ("origin", pathstr(&tr.path).into()),
                (
                    "origin_crop",
                    J::Arr(tr.crop.iter().map(|&n| n.into()).collect()),
                ),
                ("native", tr.native.into()),
                ("variant", v.name.into()),
                ("family", v.family.into()),
                (
                    "tags",
                    J::Arr(tr.tags.iter().cloned().map(J::from).collect()),
                ),
                // A truth image tagged `holdout` makes every one of its variants holdout,
                // so `--split dev` never sees degraded copies of held-out art.
                (
                    "split",
                    if tr.tags.iter().any(|t| t == "holdout") { "holdout" } else { "dev" }.into(),
                ),
                ("scale", v.scale.into()),
                ("phase_x", phase_x.into()),
                ("phase_y", phase_y.into()),
                ("source_w", im.width().into()),
                ("source_h", im.height().into()),
                ("truth_w", target.width().into()),
                ("truth_h", target.height().into()),
                (
                    "effective_scale_x",
                    ((im.width() as f64 - phase_x as f64) / (target.width() as f64)).into(),
                ),
                (
                    "effective_scale_y",
                    ((im.height() as f64 - phase_y as f64) / (target.height() as f64)).into(),
                ),
                ("truth_palette_size", rgb_palette(&target).len().into()),
                ("source_hash", image_hash(&im).into()),
                ("truth_hash", image_hash(&target).into()),
                ("parameters", params),
            ]);
            write_json(dir.join("meta.json"), &case)?;
            cases.push(case);
        }
        println!(
            "degrade {id}: {} ({})",
            if tr.core { "core" } else { "full" },
            truth.width() * truth.height()
        );
    }
    if let Some(dir) = args.opts.get("--input-dir") {
        let cases = scan_real(Path::new(dir))?;
        write_json("eval_out/imported_real.json", &J::Arr(cases))?
    }
    let n = cases.len();
    write_json(
        "eval_out/corpus.json",
        &obj([
            ("schema", 1usize.into()),
            ("seed_hex", format!("{seed:016x}").into()),
            ("truth_manifest", pathstr(manifest).into()),
            ("cases", J::Arr(cases)),
        ]),
    )?;
    println!(
        "degrade: {n} synthetic variants in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
fn real_case(p: &Path, split: &str, tags: Vec<String>) -> E<J> {
    let im = load(p)?;
    let stem = p.file_stem().unwrap_or_default().to_string_lossy();
    let assumed = stem.split('_').rev().find_map(|t| t.parse::<u32>().ok());
    Ok(obj([
        (
            "id",
            format!("real-{}-{:016x}", slug(&stem), hash(pathstr(p).as_bytes())).into(),
        ),
        ("source", pathstr(p).into()),
        ("truth", J::Null),
        ("family", "real".into()),
        ("split", split.into()),
        ("tags", J::Arr(tags.into_iter().map(J::from).collect())),
        ("presumed_scale", assumed.map(J::from).unwrap_or(J::Null)),
        ("source_w", im.width().into()),
        ("source_h", im.height().into()),
        ("source_hash", image_hash(&im).into()),
    ]))
}
fn scan_real(dir: &Path) -> E<Vec<J>> {
    let mut paths = Vec::new();
    collect_files(dir, &mut paths)?;
    paths
        .into_iter()
        .filter(|p| {
            p.extension().is_some_and(|e| {
                matches!(
                    e.to_string_lossy().to_ascii_lowercase().as_str(),
                    "png" | "jpg" | "jpeg" | "webp"
                )
            })
        })
        .map(|p| real_case(&p, "dev", vec!["imported".into()]))
        .collect()
}
pub(super) fn real_cases(args: &Args) -> E<Vec<J>> {
    let manifest = Path::new(args.get("--real-manifest", "eval/real_manifest.txt"));
    let mut out = Vec::new();
    if manifest.exists() {
        for (i, line) in fs::read_to_string(manifest)?.lines().enumerate() {
            let t = tokens(line)?;
            if t.is_empty() {
                continue;
            }
            if t.len() != 3 || !["dev", "holdout"].contains(&t[1].as_str()) {
                return Err(format!(
                    "{}:{}: expected path dev|holdout comma-separated-tags",
                    manifest.display(),
                    i + 1
                )
                .into());
            }
            out.push(real_case(
                Path::new(&t[0]),
                &t[1],
                t[2].split(',').map(String::from).collect(),
            )?);
        }
    }
    if let Some(dir) = args.opts.get("--input-dir") {
        out.extend(scan_real(Path::new(dir))?)
    } else if Path::new("eval_out/imported_real.json").exists() {
        out.extend(
            read_json("eval_out/imported_real.json")?
                .a()
                .iter()
                .cloned(),
        )
    }
    let mut unique = BTreeMap::new();
    for c in out {
        unique.entry(c.get("id").s().to_string()).or_insert(c);
    }
    Ok(unique.into_values().collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_crop_preserves_all_truth() {
        let im = RgbaImage::from_fn(3, 4, |x, y| Rgba([x as u8 * 80, y as u8 * 50, 15, 255]));
        for s in [3, 4, 5, 8] {
            let v = variants(false, false)
                .into_iter()
                .find(|v| v.name == format!("nearest_{s}"))
                .unwrap();
            for seed in 0..8 {
                let (im, truth, p) = degrade_one(&im, &v, seed).unwrap();
                let px = (s - p.get("phase_crop_left").n().unwrap() as u32) % s;
                let py = (s - p.get("phase_crop_top").n().unwrap() as u32) % s;
                let cfg = Config {
                    pixel_size: Some(s as f64),
                    phase_x: Some(px),
                    phase_y: Some(py),
                    palette: PaletteChoice::HexList(
                        rgb_palette(&truth)
                            .iter()
                            .map(|p| format!("{:02x}{:02x}{:02x}\n", p[0], p[1], p[2]))
                            .collect(),
                    ),
                    highlight_collapse: 0.,
                    shadow_collapse: 0.,
                    ..auto_config()
                };
                assert_eq!(pipeline::convert(&im, &cfg).unwrap().image, truth);
            }
        }
    }
    #[test]
    fn seeded_degradation_is_repeatable() {
        let im = RgbaImage::from_pixel(4, 4, Rgba([80, 100, 150, 255]));
        let v = variants(false, false)
            .into_iter()
            .find(|v| v.family == "noise")
            .unwrap();
        assert_eq!(
            degrade_one(&im, &v, 77).unwrap().0,
            degrade_one(&im, &v, 77).unwrap().0
        );
    }
}
