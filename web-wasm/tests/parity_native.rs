//! Writes what the native pipeline produces, for `parity.mjs` to compare the browser
//! build against. Generates its own fixtures (`examples/` is local-only, so CI has no
//! images) and optionally adds a folder of real images:
//!
//!   cargo run --release --example parity_native -- <out dir> [real image dir]
//!
//! Output: <out>/manifest.json, <out>/fixtures/*, <out>/native/*

use std::{fmt::Write as _, fs, io::Cursor, path::Path};

use image::{codecs::webp::WebPEncoder, DynamicImage, ImageFormat, Rgba, RgbaImage};
use raster_to_pixel::{
    api::{self, ApiResponse, Limits},
    image_io,
};

/// The frontend's defaults, in the form `settingsParams()` sends them.
const GUI_DEFAULTS: &str = "sizeMode=auto&size=64&pixelSize=6&snapGrid=true&colors=16\
&palette=adaptive&customHex=&highlightCollapse=0.03&shadowCollapse=0.16&paletteMerge=0\
&quantizer=kmeans&cell=detail&dominantThreshold=0.25&adaptiveIterations=3&contrastExpansion=0\
&dither=none&ditherStrength=0.35&alphaThreshold=128&alphaMode=preserve&bgTolerance=0.1\
&cleanup=none&outline=none&scale=1&colorKey=00ff00&protectDetails=true&autoColors=true\
&compare=false";

/// Each case overrides the GUI defaults; together they reach every pipeline stage.
const CASES: &[(&str, &str)] = &[
    ("defaults", ""),
    (
        "fixed_kmeans",
        "sizeMode=size&size=48&autoColors=false&colors=16",
    ),
    (
        "pixel_wu_merge",
        "sizeMode=pixel&pixelSize=4&autoColors=false&colors=32&quantizer=wu&paletteMerge=0.04",
    ),
    (
        "many_colors",
        "sizeMode=size&size=96&autoColors=false&colors=256",
    ),
    ("no_snap", "sizeMode=pixel&pixelSize=5.5&snapGrid=false"),
    ("cell_box", "cell=box"),
    ("cell_median", "cell=median"),
    ("cell_dominant", "cell=dominant&dominantThreshold=0.5"),
    (
        "cell_adaptive",
        "sizeMode=size&size=64&cell=adaptive&adaptiveIterations=2",
    ),
    ("pico8_bayer4", "palette=pico8&dither=bayer4"),
    (
        "gameboy_bayer8",
        "palette=gameboy&dither=bayer8&ditherStrength=0.6",
    ),
    ("sweetie16", "palette=sweetie16"),
    (
        "custom_hex",
        "palette=custom&customHex=%231a1c2c%0A5d275d%0Ab13e53%0Aef7d57%0Affcd75",
    ),
    ("alpha_binary", "alphaMode=binary&alphaThreshold=90"),
    ("alpha_bgfill", "alphaMode=bgfill&bgTolerance=0.2"),
    (
        "alpha_key",
        "alphaMode=key&colorKey=102030&bgTolerance=0.15",
    ),
    ("cleanup_balanced", "cleanup=balanced"),
    (
        "cleanup_aggressive",
        "cleanup=aggressive&protectDetails=false",
    ),
    ("outline_repair", "outline=repair"),
    ("outline_enforce", "outline=enforce"),
    ("detail_guard", "contrastExpansion=2"),
    ("crop", "crop=8%2C12%2C100%2C80"),
    (
        "scaled_compare",
        "sizeMode=size&size=40&scale=4&compare=true",
    ),
];

/// Real images are big: run the cases most likely to expose platform drift.
const REAL_IMAGE_CASES: &[&str] = &[
    "defaults",
    "pixel_wu_merge",
    "cell_adaptive",
    "pico8_bayer4",
    "alpha_bgfill",
    "cleanup_balanced",
    "scaled_compare",
];

const LOSSY_WEBP: &[u8] = include_bytes!("fixtures/lossy.webp");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .ok_or("usage: parity_native <out dir> [real image dir]")?;
    let out = Path::new(&out);
    fs::create_dir_all(out.join("fixtures"))?;
    fs::create_dir_all(out.join("native"))?;

    let mut fixtures: Vec<(String, Vec<u8>, bool)> = synthetic_fixtures()?
        .into_iter()
        .map(|(name, bytes)| (name, bytes, false))
        .collect();
    if let Some(dir) = args.next() {
        let mut paths: Vec<_> = fs::read_dir(dir)?.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            ) {
                let name = format!("real_{}", path.file_name().unwrap().to_string_lossy());
                fixtures.push((name, fs::read(&path)?, true));
            }
        }
    }

    let mut manifest = String::from("{\n");
    let decoder = if cfg!(windows) { "gdi+" } else { "image-crate" };
    writeln!(
        manifest,
        "\"nativeJpegDecoder\": \"{decoder}\",\n\"fixtures\": ["
    )?;
    for (index, (name, bytes, real)) in fixtures.iter().enumerate() {
        eprintln!("native: {name}");
        fs::write(out.join("fixtures").join(name), bytes)?;
        let src = image_io::load_rgba_from_memory(bytes)?;
        let session = api::session_response(&src);
        write!(
            manifest,
            "{}{{\"name\":{},\"session\":{},\"outputs\":[",
            if index == 0 { "" } else { ",\n" },
            json(name),
            json(&String::from_utf8_lossy(&session.body)),
        )?;
        let mut first = true;
        for (case, overrides) in CASES {
            if *real && !REAL_IMAGE_CASES.contains(case) {
                continue;
            }
            let settings = format!("{GUI_DEFAULTS}&{overrides}");
            let mut emit = |op: &str, settings: &str, reply: ApiResponse| -> std::io::Result<()> {
                let file = format!("{name}__{case}__{op}");
                fs::write(out.join("native").join(&file), &reply.body)?;
                write!(
                    manifest,
                    "{}\n  {{\"case\":{},\"op\":{},\"settings\":{},\"status\":{},\"contentType\":{},\"headers\":{},\"file\":{}}}",
                    if first { "" } else { "," },
                    json(case),
                    json(op),
                    json(settings),
                    reply.status,
                    json(reply.content_type),
                    json(&reply.headers.join("\n")),
                    json(&file),
                )
                .unwrap();
                first = false;
                Ok(())
            };
            let limits = Limits::default();
            emit(
                "preview",
                &settings,
                api::convert_response(Some(&src), &settings, true, limits),
            )?;
            if *case == "scaled_compare" || *case == "defaults" {
                emit(
                    "export",
                    &settings,
                    api::convert_response(Some(&src), &settings, false, limits),
                )?;
            }
            if *case == "defaults" {
                for format in ["hex", "gpl", "png"] {
                    let settings = format!("{settings}&paletteFormat={format}");
                    let reply = api::palette_response(Some(&src), &settings, limits);
                    let op = format!("palette_{format}");
                    emit(&op, &settings, reply)?;
                }
            }
        }
        manifest.push_str("\n]}");
    }
    manifest.push_str("\n]\n}\n");
    fs::write(out.join("manifest.json"), manifest)?;
    Ok(())
}

fn json(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

type NamedBytes = (String, Vec<u8>);

fn synthetic_fixtures() -> Result<Vec<NamedBytes>, Box<dyn std::error::Error>> {
    let sprite = fuzzy_sprite();
    let encode = |img: DynamicImage, format| -> Result<Vec<u8>, image::ImageError> {
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), format)?;
        Ok(bytes)
    };

    let opaque = DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(gradient_scene()).to_rgb8());
    let gray16 = DynamicImage::ImageRgba8(sprite.clone()).to_luma_alpha16();

    let mut webp = Vec::new();
    WebPEncoder::new_lossless(&mut webp).encode(
        sprite.as_raw(),
        sprite.width(),
        sprite.height(),
        image::ExtendedColorType::Rgba8,
    )?;

    Ok(vec![
        (
            "sprite_alpha.png".into(),
            encode(sprite.into(), ImageFormat::Png)?,
        ),
        (
            "gradient_opaque.png".into(),
            encode(opaque.clone(), ImageFormat::Png)?,
        ),
        (
            "gray_alpha_16bit.png".into(),
            encode(gray16.into(), ImageFormat::Png)?,
        ),
        ("sprite_alpha_lossless.webp".into(), webp),
        ("lossy.webp".into(), LOSSY_WEBP.to_vec()),
        ("gradient.jpg".into(), encode(opaque, ImageFormat::Jpeg)?),
        ("non_interleaved.jpg".into(), non_interleaved_jpeg()),
    ])
}

/// Tiny deterministic generator, so fixtures are identical on every machine.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self, bound: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 16) % bound
    }
}

/// Fuzzy "AI pixel art": 5px blocks with a soft-edged transparent background, offset
/// grid phase, generated noise, and a keyed-color corner. Exercises alpha handling.
fn fuzzy_sprite() -> RgbaImage {
    const COLORS: [[u8; 3]; 6] = [
        [36, 28, 44],
        [178, 62, 83],
        [239, 125, 87],
        [255, 205, 117],
        [56, 183, 100],
        [246, 246, 244],
    ];
    let mut rng = Lcg(7);
    let mut blocks = [[0usize; 40]; 40];
    for row in blocks.iter_mut() {
        for cell in row.iter_mut() {
            *cell = rng.next(COLORS.len() as u32) as usize;
        }
    }
    RgbaImage::from_fn(203, 203, |x, y| {
        let (bx, by) = (
            ((x + 2) / 5).min(39) as usize,
            ((y + 3) / 5).min(39) as usize,
        );
        let [r, g, b] = COLORS[blocks[by][bx]];
        let (dx, dy) = (x as i32 - 101, y as i32 - 101);
        let d = dx * dx + dy * dy;
        let alpha = if d < 80 * 80 {
            255
        } else if d < 92 * 92 {
            (255 * (92 * 92 - d) / (92 * 92 - 80 * 80)) as u8
        } else {
            0
        };
        if x < 12 && y < 12 {
            return Rgba([16, 32, 48, 255]);
        }
        let n = rng.next(9) as i32 - 4;
        let jitter = |c: u8| (c as i32 + n).clamp(0, 255) as u8;
        Rgba([jitter(r), jitter(g), jitter(b), alpha])
    })
}

/// Opaque smooth gradients plus hard-edged shapes.
fn gradient_scene() -> RgbaImage {
    RgbaImage::from_fn(240, 180, |x, y| {
        let inside = (40..110).contains(&x) && (50..120).contains(&y);
        if inside {
            Rgba([230, 80, 60, 255])
        } else {
            Rgba([
                (x * 255 / 239) as u8,
                (y * 255 / 179) as u8,
                ((x + y) / 2) as u8,
                255,
            ])
        }
    })
}

/// A baseline JPEG whose components each get their own scan: the layout (found in
/// generated images such as examples/girl_8.jpeg) that the pure-Rust decoder renders
/// neon green. `image` cannot encode this, so it is assembled by hand from flat blocks.
fn non_interleaved_jpeg() -> Vec<u8> {
    const SIZE: u16 = 64;
    // Brown (Y, Cb, Cr) = RGB ~(124, 94, 68), the expected result checked by parity.mjs.
    let (luma, cb, cr) = (100i32, 110i32, 145i32);

    let mut jpeg = vec![0xff, 0xd8];
    push_segment(&mut jpeg, 0xe0, b"JFIF      ");
    // One quantization table of all ones, so DC coefficients are 8 * (sample - 128).
    let mut dqt = vec![0u8];
    dqt.extend([1u8; 64]);
    push_segment(&mut jpeg, 0xdb, &dqt);
    let [h1, h0] = SIZE.to_be_bytes();
    let frame = [8, h1, h0, h1, h0, 3, 1, 0x22, 0, 2, 0x11, 0, 3, 0x11, 0];
    push_segment(&mut jpeg, 0xc0, &frame);
    // DC table: symbols 0-11 with 4-bit codes. AC table: only end-of-block, code "0".
    let mut dht = vec![0x00, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    dht.extend(0u8..12);
    dht.extend([0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00]);
    push_segment(&mut jpeg, 0xc4, &dht);

    let scans = [
        (1u8, luma, SIZE / 8),
        (2, cb, SIZE / 16),
        (3, cr, SIZE / 16),
    ];
    for (component, sample, blocks_per_side) in scans {
        push_segment(&mut jpeg, 0xda, &[1, component, 0x00, 0, 63, 0]);
        let mut bits = BitWriter::default();
        let mut previous = 0;
        for _ in 0..blocks_per_side as u32 * blocks_per_side as u32 {
            let dc = (sample - 128) * 8;
            let diff = dc - previous;
            previous = dc;
            let category = 32 - diff.unsigned_abs().leading_zeros();
            bits.push(category, 4);
            let raw = if diff < 0 {
                diff + (1 << category) - 1
            } else {
                diff
            };
            bits.push(raw as u32, category);
            bits.push(0, 1); // End of block.
        }
        jpeg.extend(bits.finish());
    }
    jpeg.extend([0xff, 0xd9]);
    jpeg
}

fn push_segment(jpeg: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    jpeg.extend([0xff, marker]);
    jpeg.extend(((payload.len() + 2) as u16).to_be_bytes());
    jpeg.extend(payload);
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    accumulator: u32,
    count: u32,
}

impl BitWriter {
    fn push(&mut self, value: u32, width: u32) {
        for shift in (0..width).rev() {
            self.accumulator = self.accumulator << 1 | (value >> shift & 1);
            self.count += 1;
            if self.count == 8 {
                self.flush_byte();
            }
        }
    }

    fn flush_byte(&mut self) {
        let byte = self.accumulator as u8;
        self.bytes.push(byte);
        if byte == 0xff {
            self.bytes.push(0);
        }
        self.accumulator = 0;
        self.count = 0;
    }

    /// Pad the last byte with ones, as the JPEG spec requires.
    fn finish(mut self) -> Vec<u8> {
        while self.count != 0 {
            self.push(1, 1);
        }
        self.bytes
    }
}
