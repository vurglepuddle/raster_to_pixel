//! Pixeline's local evaluation instrument. Run from the repository root.
//!
//! Baseline (owner-supplied, local-only truth is listed in eval/truth_manifest.txt):
//!   cargo run --release --example eval -- degrade
//!   cargo run --release --example eval -- run --label baseline --mode all
//!   cargo run --release --example eval -- score --label baseline
//! After changing the engine, repeat run/score with --label candidate, then:
//!   cargo run --release --example eval -- sheet --labels baseline,candidate --mode auto
//!
//! Modes: auto, oracle-grid, oracle-palette, oracle-both (or all for run).
//! Real cases always use Auto and are emitted only under auto, even with mode=all.
//! Optional run/degrade --input-dir DIR adds degraded-only real files recursively;
//! degrade remembers them in eval_out/imported_real.json. --filter SUBSTR selects
//! case IDs, paths or tags. --split dev|holdout selects cases; synthetic variants inherit holdout from their truth tag.
//! --truth-manifest PATH overrides the default; --real-manifest PATH likewise.
//! --seed U64 controls degradation randomness (default 20261113).
//!   cargo run --release --example eval -- selftest
//! creates tiny private fixtures/manifests under eval_out/selftest, never examples/.
//!
//! All generated files stay in eval_out. Labels may not overwrite existing modes;
//! choose a fresh label or pass --replace to explicitly replace that mode's index.
//! Old unreferenced files are harmless: indices, never directory globs, define runs.
//! --replace never deletes directories. No git writes, engine changes or WASM build.
//! PNG outputs are raw art grids, not presentation-scale exports. Runtime excludes
//! decoding and PNG encoding; summary wall time includes both. Score never resamples.
//! See metrics.rs for thresholds, alpha conventions and exact metric definitions.
mod corpus;
mod json;
mod metrics;
mod sheet;

use image::RgbaImage;
use json::{obj, J};
use raster_to_pixel::{
    image_io,
    pipeline::{self, Config, PaletteChoice},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type E<T> = Result<T, Box<dyn std::error::Error>>;
const OUT: &str = "eval_out";
const MODES: [&str; 4] = ["auto", "oracle-grid", "oracle-palette", "oracle-both"];

#[derive(Default)]
struct Args {
    cmd: String,
    opts: BTreeMap<String, String>,
}
impl Args {
    fn parse() -> E<Self> {
        let mut it = env::args().skip(1);
        let cmd = it
            .next()
            .ok_or("expected degrade|run|score|sheet|selftest")?;
        let mut opts = BTreeMap::new();
        while let Some(k) = it.next() {
            if !k.starts_with("--") {
                return Err(format!("unexpected argument {k}").into());
            }
            let v = if k == "--replace" {
                "true".into()
            } else {
                it.next().ok_or_else(|| format!("missing value for {k}"))?
            };
            if opts.insert(k, v).is_some() {
                return Err("duplicate option".into());
            }
        }
        let allowed = match cmd.as_str() {
            "degrade" => vec!["--truth-manifest", "--input-dir", "--seed", "--filter"],
            "run" => vec![
                "--label",
                "--mode",
                "--input-dir",
                "--real-manifest",
                "--filter",
                "--split",
                "--replace",
            ],
            "score" => vec!["--label"],
            "sheet" => vec!["--labels", "--mode", "--filter"],
            "selftest" => vec![],
            _ => return Err(format!("unknown subcommand {cmd}").into()),
        };
        for k in opts.keys() {
            if !allowed.contains(&k.as_str()) {
                return Err(format!("unknown option {k} for {cmd}").into());
            }
        }
        Ok(Self { cmd, opts })
    }
    fn get<'a>(&'a self, k: &str, default: &'a str) -> &'a str {
        self.opts.get(k).map(String::as_str).unwrap_or(default)
    }
    fn need(&self, k: &str) -> E<&str> {
        self.opts
            .get(k)
            .map(String::as_str)
            .ok_or_else(|| format!("required: {k}").into())
    }
    fn matches(&self, c: &J) -> bool {
        let needle = self.get("--filter", "");
        (needle.is_empty() || c.dump().contains(needle))
            && (self.get("--split", "").is_empty() || c.get("split").s() == self.get("--split", ""))
    }
}
fn main() {
    if let Err(e) = entry() {
        eprintln!("eval: {e}");
        std::process::exit(1)
    }
}
fn entry() -> E<()> {
    let args = Args::parse()?;
    if !Path::new("Cargo.toml").is_file() || !Path::new("src/pipeline.rs").is_file() {
        return Err("run from the repository root".into());
    }
    fs::create_dir_all(OUT)?;
    match args.cmd.as_str() {
        "degrade" => corpus::degrade(&args),
        "run" => run(&args),
        "score" => metrics::score(&args),
        "sheet" => sheet::sheet(&args),
        "selftest" => selftest(),
        _ => unreachable!(),
    }
}
fn read_json(p: impl AsRef<Path>) -> E<J> {
    Ok(json::parse(&fs::read_to_string(p)?)?)
}
fn write_json(p: impl AsRef<Path>, j: &J) -> E<()> {
    write(p, format!("{}\n", j.dump()).as_bytes())
}
fn write(p: impl AsRef<Path>, bytes: &[u8]) -> E<()> {
    if let Some(parent) = p.as_ref().parent() {
        fs::create_dir_all(parent)?
    }
    fs::write(p, bytes)?;
    Ok(())
}
fn load(p: impl AsRef<Path>) -> E<RgbaImage> {
    Ok(image_io::load_rgba_from_path(p.as_ref())?)
}
fn pathstr(p: impl AsRef<Path>) -> String {
    p.as_ref().to_string_lossy().replace('\\', "/")
}
fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    })
}
fn image_hash(im: &RgbaImage) -> String {
    let mut b = im.width().to_le_bytes().to_vec();
    b.extend(im.height().to_le_bytes());
    b.extend(im.as_raw());
    format!("{:016x}", hash(&b))
}
fn slug(s: &str) -> String {
    let v: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    v.chars().take(90).collect()
}
fn label(s: &str) -> E<&str> {
    if s.is_empty()
        || s.len() > 80
        || !s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("labels must be 1..80 ASCII letters, digits, hyphens or underscores".into());
    }
    Ok(s)
}
fn rgb_palette(im: &RgbaImage) -> Vec<[u8; 3]> {
    im.pixels()
        .filter(|p| p[3] > 0)
        .map(|p| [p[0], p[1], p[2]])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
fn palette_json(p: &[[u8; 3]]) -> J {
    J::Arr(
        p.iter()
            .map(|p| J::Arr(p.iter().map(|&c| J::from(c as u32)).collect()))
            .collect(),
    )
}
fn parse_palette(j: &J) -> Vec<[u8; 3]> {
    j.a()
        .iter()
        .filter(|p| p.a().len() == 3)
        .map(|p| std::array::from_fn(|i| p.a()[i].n().unwrap_or(0.) as u8))
        .collect()
}
fn auto_config() -> Config {
    // The library's Auto preset (GUI initial state, README quick start). Until
    // engine v2 this function pinned the old defaults by hand; the baseline
    // label was produced with those (detail cells, k-means, bucket-count colours).
    Config {
        max_output_pixels: Some(20_000_000),
        // Lab switch for A/B labels; the shipped preset decides the default.
        elastic: std::env::var("PIXELINE_ELASTIC").map_or(Config::auto().elastic, |v| v != "0"),
        ..Config::auto()
    }
}
fn config_json(c: &Config) -> J {
    // Include Debug as an exhaustive snapshot, plus structured frequently compared fields.
    obj([
        ("rust_debug", format!("{c:?}").into()),
        ("size", c.size.into()),
        ("pixel_size", c.pixel_size.map(J::from).unwrap_or(J::Null)),
        ("auto_pixel_size", c.auto_pixel_size.into()),
        ("auto_colors", c.auto_colors.into()),
        ("snap_grid", c.snap_grid.into()),
        ("colors", c.colors.into()),
        ("palette", format!("{:?}", c.palette).into()),
        ("cell", format!("{:?}", c.cell).into()),
        ("dominant_threshold", (c.dominant_threshold as f64).into()),
        ("highlight_collapse", (c.highlight_collapse as f64).into()),
        ("shadow_collapse", (c.shadow_collapse as f64).into()),
        ("palette_merge", (c.palette_merge as f64).into()),
        ("phase_x", c.phase_x.map(J::from).unwrap_or(J::Null)),
        ("phase_y", c.phase_y.map(J::from).unwrap_or(J::Null)),
        ("alpha_threshold", (c.alpha_threshold as u32).into()),
        ("scale", c.scale.into()),
    ])
}
fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(["-c", "safe.directory=E:/SD/Raster_to_Pixel"])
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().into())
        .unwrap_or_else(|| "UNAVAILABLE".into())
}
fn engine_stamp() -> E<J> {
    let mut paths = vec![PathBuf::from("Cargo.toml"), PathBuf::from("Cargo.lock")];
    collect_files(Path::new("src"), &mut paths)?;
    paths.sort();
    let mut bytes = Vec::new();
    for p in paths {
        bytes.extend(pathstr(&p).as_bytes());
        bytes.push(0);
        bytes.extend(fs::read(p)?);
        bytes.push(0)
    }
    let engine_dirty = git(&[
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "src",
        "Cargo.toml",
        "Cargo.lock",
    ]);
    let worktree = git(&["status", "--porcelain"]);
    Ok(obj([
        ("commit", git(&["rev-parse", "HEAD"]).into()),
        ("engine_dirty", (!engine_dirty.is_empty()).into()),
        ("engine_status", engine_dirty.into()),
        ("worktree_dirty", (!worktree.is_empty()).into()),
        (
            "content_hash_fnv1a64",
            format!("{:016x}", hash(&bytes)).into(),
        ),
    ]))
}
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> E<()> {
    let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let ty = e.file_type()?;
        if ty.is_dir() {
            collect_files(&e.path(), out)?
        } else if ty.is_file() {
            out.push(e.path())
        }
    }
    Ok(())
}
fn run(args: &Args) -> E<()> {
    let name = label(args.need("--label")?)?;
    let requested = args.get("--mode", "auto");
    let modes = if requested == "all" {
        MODES.to_vec()
    } else if MODES.contains(&requested) {
        vec![requested]
    } else {
        return Err("invalid mode".into());
    };
    if !["", "dev", "holdout"].contains(&args.get("--split", "")) {
        return Err("split must be dev or holdout".into());
    }
    let mut cases = if Path::new("eval_out/corpus.json").exists() {
        read_json("eval_out/corpus.json")?.get("cases").a().to_vec()
    } else {
        Vec::new()
    };
    cases.extend(corpus::real_cases(args)?);
    cases.retain(|c| args.matches(c));
    cases.sort_by_key(|c| c.get("id").s().to_string());
    let mut seen = BTreeSet::new();
    for c in &cases {
        if !seen.insert(c.get("id").s()) {
            return Err("duplicate case ID".into());
        }
    }
    if cases.is_empty() {
        return Err("no cases: run degrade, supply real_manifest, or --input-dir".into());
    }
    let stamp = engine_stamp()?;
    for mode in modes {
        let dir = PathBuf::from(OUT).join("runs").join(name).join(mode);
        if dir.join("summary.json").exists() && args.get("--replace", "") != "true" {
            return Err(format!("{} exists; use another label or --replace", dir.display()).into());
        }
        fs::create_dir_all(&dir)?;
        let start = Instant::now();
        let mut records = Vec::new();
        let mut failures = 0usize;
        let mut convert_ms = 0.;
        for case in &cases {
            if mode != "auto" && case.get("truth").s().is_empty() {
                continue;
            }
            let id = case.get("id").s();
            let mut record = obj([("case", case.clone()), ("mode", mode.into())]);
            let result = (|| -> E<()> {
                let source = load(case.get("source").s())?;
                if image_hash(&source) != case.get("source_hash").s() {
                    return Err("source changed since corpus/index creation".into());
                }
                let mut cfg = auto_config();
                if mode == "oracle-grid" || mode == "oracle-both" {
                    cfg.auto_pixel_size = false;
                    cfg.pixel_size = Some(case.get("scale").n().ok_or("missing synthetic scale")?);
                    cfg.phase_x = Some(case.get("phase_x").n().unwrap_or(0.) as u32);
                    cfg.phase_y = Some(case.get("phase_y").n().unwrap_or(0.) as u32);
                }
                if mode == "oracle-palette" || mode == "oracle-both" {
                    let truth = load(case.get("truth").s())?;
                    if image_hash(&truth) != case.get("truth_hash").s() {
                        return Err("truth changed since degrade".into());
                    }
                    let palette = rgb_palette(&truth);
                    cfg.palette = if palette.is_empty() {
                        PaletteChoice::Adaptive
                    } else {
                        PaletteChoice::HexList(
                            palette
                                .iter()
                                .map(|p| format!("{:02x}{:02x}{:02x}\n", p[0], p[1], p[2]))
                                .collect(),
                        )
                    };
                    cfg.auto_colors = false;
                }
                record.set("resolved_config", config_json(&cfg));
                let now = Instant::now();
                let converted = pipeline::convert(&source, &cfg)?;
                let ms = now.elapsed().as_secs_f64() * 1000.;
                convert_ms += ms;
                converted.image.save(dir.join(format!("{id}.png")))?;
                record.set("runtime_ms", ms);
                record.set(
                    "diagnostics",
                    json::parse(&converted.diagnostics_json(&cfg))?,
                );
                record.set("palette", palette_json(&converted.palette));
                record.set("output_hash", image_hash(&converted.image));
                record.set("output", pathstr(dir.join(format!("{id}.png"))));
                Ok(())
            })();
            match result {
                Ok(()) => record.set("status", "ok"),
                Err(e) => {
                    failures += 1;
                    record.set("status", "error");
                    record.set("error", e.to_string());
                    eprintln!("{mode}/{id}: {e}")
                }
            }
            write_json(dir.join(format!("{id}.json")), &record)?;
            records.push(J::from(id));
        }
        let summary = obj([
            ("schema", 1usize.into()),
            ("label", name.into()),
            ("mode", mode.into()),
            ("engine", stamp.clone()),
            ("case_ids", J::Arr(records)),
            ("errors", failures.into()),
            ("convert_ms", convert_ms.into()),
            ("wall_ms", (start.elapsed().as_secs_f64() * 1000.).into()),
        ]);
        write_json(dir.join("summary.json"), &summary)?;
        println!(
            "{name}/{mode}: {} cases, {failures} errors, conversion {:.3}s, wall {:.3}s",
            summary.get("case_ids").a().len(),
            convert_ms / 1000.,
            start.elapsed().as_secs_f64()
        );
        if failures > 0 {
            return Err(format!("{failures} conversion errors; artifacts retained").into());
        }
    }
    Ok(())
}
fn selftest() -> E<()> {
    let dir = Path::new("eval_out/selftest");
    fs::create_dir_all(dir)?;
    for (name, w, h, alpha) in [
        ("icon", 16, 16, false),
        ("sprite", 17, 19, true),
        ("sheet", 24, 16, false),
    ] {
        let colors = [
            [20, 25, 40, 255],
            [230, 170, 40, 255],
            [50, 160, 100, 255],
            [245, 240, 220, 255],
            [210, 35, 85, 255],
        ];
        let mut im = RgbaImage::new(w, h);
        for (x, y, p) in im.enumerate_pixels_mut() {
            *p = image::Rgba(if alpha && (x < 2 || y < 2 || x > w - 3 || y > h - 3) {
                [0, 0, 0, 0]
            } else {
                colors[((x / 4 + y / 4) % 4) as usize]
            });
        }
        im.put_pixel(w / 2, h / 2, image::Rgba(colors[4]));
        im.put_pixel(w / 2 + 1, h / 2, image::Rgba(colors[4]));
        im.save(dir.join(format!("{name}.png")))?;
    }
    let sprite = load(dir.join("sprite.png"))?;
    image::imageops::resize(
        &sprite,
        sprite.width() * 3,
        sprite.height() * 3,
        image::imageops::FilterType::Nearest,
    )
    .save(dir.join("sprite3.png"))?;
    write(dir.join("truth_manifest.txt"),b"eval_out/selftest/icon.png 0 0 16 16 icon,accent variants=full\neval_out/selftest/sprite3.png 0 0 17 19 alpha,sprite native=3 variants=core\neval_out/selftest/sheet.png 2 1 20 14 sheet,crop variants=core\n")?;
    println!("Created eval_out/selftest/truth_manifest.txt (does not edit the owner's manifest).");
    Ok(())
}
