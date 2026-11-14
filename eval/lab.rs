//! Engine-development probes that are too narrow for the main harness.
//!
//!   cargo run --release --example lab -- grid [--all] [--verbose] [FILTER]
//!
//! `grid` runs only grid inference over the synthetic corpus (dev split unless
//! `--all`) and the real manifest, and prints output-size matches per variant.
//! It reads `eval_out/corpus.json` (create it with the eval example's
//! `degrade`) and never writes anything.
#[allow(dead_code)]
#[path = "json.rs"]
mod json;

use raster_to_pixel::{
    grid::{detect, GridDecision},
    image_io,
};
use std::{collections::BTreeMap, fs, path::Path};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("grid") => grid(&args[1..]),
        _ => eprintln!("usage: lab grid [--all] [--verbose] [FILTER]"),
    }
}

fn grid(args: &[String]) {
    let all = args.iter().any(|a| a == "--all");
    let verbose = args.iter().any(|a| a == "--verbose");
    let filter = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_default();
    let corpus = json::parse(&fs::read_to_string("eval_out/corpus.json").expect("corpus.json"))
        .expect("corpus json");
    let mut table: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let start = std::time::Instant::now();
    for case in corpus.get("cases").a() {
        let id = case.get("id").s();
        if !id.contains(&filter) || (!all && case.get("split").s() != "dev") {
            continue;
        }
        let img = image_io::load_rgba_from_path(Path::new(case.get("source").s())).expect("load");
        let truth = (
            case.get("truth_w").n().unwrap() as u32,
            case.get("truth_h").n().unwrap() as u32,
        );
        let decision = detect(&img);
        let got = match decision {
            GridDecision::Lattice(fit) => (fit.cols, fit.rows),
            GridDecision::NoGrid { .. } => img.dimensions(),
        };
        let entry = table.entry(case.get("variant").s().to_string()).or_default();
        entry.1 += 1;
        if got == truth {
            entry.0 += 1;
        } else if verbose {
            println!("MISS {id} truth {truth:?} got {got:?} {decision:?}");
        }
    }
    let (mut ok, mut total) = (0, 0);
    for (variant, (hit, n)) in &table {
        println!("{variant:34} {hit:3}/{n:3}");
        ok += hit;
        total += n;
    }
    println!("TOTAL {ok}/{total} in {:.1}s", start.elapsed().as_secs_f64());

    for line in fs::read_to_string("eval/real_manifest.txt")
        .unwrap_or_default()
        .lines()
    {
        let Some(path) = line.split_whitespace().next().filter(|p| !p.starts_with('#')) else {
            continue;
        };
        if !path.contains(&filter) {
            continue;
        }
        let Ok(img) = image_io::load_rgba_from_path(Path::new(path)) else {
            continue;
        };
        let t = std::time::Instant::now();
        let decision = detect(&img);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let (w, h) = img.dimensions();
        match decision {
            GridDecision::Lattice(f) => println!(
                "{path:44} {w}x{h} -> {}x{} pitch {:.3},{:.3} origin {:.1},{:.1} R {:.3} z {:.1} ({ms:.0} ms)",
                f.cols, f.rows, f.pitch_x, f.pitch_y, f.origin_x, f.origin_y, f.coherence, f.significance
            ),
            GridDecision::NoGrid { coherence, significance } => println!(
                "{path:44} {w}x{h} -> no grid (R {coherence:.3} z {significance:.1}) ({ms:.0} ms)"
            ),
        }
    }
}
