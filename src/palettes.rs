//! Built-in palettes + palette text parsing/export.
//!
//! Accepted text formats are Lospec-style hex lists ("RRGGBB" or "#RRGGBB",
//! one per line) and GIMP `.gpl` palettes (`R G B name` rows).

/// Game Boy DMG 4-shade green, dark -> light.
pub const GAMEBOY: &[[u8; 3]] = &[
    [0x0f, 0x38, 0x0f],
    [0x30, 0x62, 0x30],
    [0x8b, 0xac, 0x0f],
    [0x9b, 0xbc, 0x0f],
];

/// PICO-8 fantasy console, 16 colors, official order.
pub const PICO8: &[[u8; 3]] = &[
    [0x00, 0x00, 0x00],
    [0x1d, 0x2b, 0x53],
    [0x7e, 0x25, 0x53],
    [0x00, 0x87, 0x51],
    [0xab, 0x52, 0x36],
    [0x5f, 0x57, 0x4f],
    [0xc2, 0xc3, 0xc7],
    [0xff, 0xf1, 0xe8],
    [0xff, 0x00, 0x4d],
    [0xff, 0xa3, 0x00],
    [0xff, 0xec, 0x27],
    [0x00, 0xe4, 0x36],
    [0x29, 0xad, 0xff],
    [0x83, 0x76, 0x9c],
    [0xff, 0x77, 0xa8],
    [0xff, 0xcc, 0xaa],
];

/// Sweetie 16 by GrafxKid (Lospec), 16 colors.
pub const SWEETIE16: &[[u8; 3]] = &[
    [0x1a, 0x1c, 0x2c],
    [0x5d, 0x27, 0x5d],
    [0xb1, 0x3e, 0x53],
    [0xef, 0x7d, 0x57],
    [0xff, 0xcd, 0x75],
    [0xa7, 0xf0, 0x70],
    [0x38, 0xb7, 0x64],
    [0x25, 0x71, 0x79],
    [0x29, 0x36, 0x6f],
    [0x3b, 0x5d, 0xc9],
    [0x41, 0xa6, 0xf6],
    [0x73, 0xef, 0xf7],
    [0xf4, 0xf4, 0xf4],
    [0x94, 0xb0, 0xc2],
    [0x56, 0x6c, 0x86],
    [0x33, 0x3c, 0x57],
];

/// Look up a built-in palette by CLI name.
pub fn builtin(name: &str) -> Option<&'static [[u8; 3]]> {
    match name.to_ascii_lowercase().as_str() {
        "gameboy" | "gb" | "dmg" => Some(GAMEBOY),
        "pico8" | "pico-8" => Some(PICO8),
        "sweetie16" | "sweetie-16" => Some(SWEETIE16),
        _ => None,
    }
}

/// Parse a Lospec-style hex palette file body.
pub fn parse_hex_list(text: &str) -> Result<Vec<[u8; 3]>, String> {
    parse_palette_list(text)
}

/// Parse a Lospec hex list or GIMP `.gpl` palette body.
pub fn parse_palette_list(text: &str) -> Result<Vec<[u8; 3]>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let mut line = raw.trim();
        if line.is_empty()
            || line.starts_with(';')
            || line.starts_with("//")
            || line.eq_ignore_ascii_case("GIMP Palette")
            || line.starts_with("Name:")
            || line.starts_with("Columns:")
        {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            let rest = rest.trim();
            if rest.len() == 6 && rest.bytes().all(|b| b.is_ascii_hexdigit()) {
                line = rest;
            } else {
                continue;
            }
        }
        if line.len() != 6 || !line.bytes().all(|b| b.is_ascii_hexdigit()) {
            if let Some(rgb) = parse_gpl_color_row(line) {
                out.push(rgb);
                continue;
            }
            return Err(format!(
                "line {}: expected RRGGBB or GIMP RGB row, got {raw:?}",
                i + 1
            ));
        }
        let v = u32::from_str_radix(line, 16).unwrap();
        out.push([(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    }
    if out.is_empty() {
        return Err("palette file contained no colors".into());
    }
    Ok(out)
}

fn parse_gpl_color_row(line: &str) -> Option<[u8; 3]> {
    let mut parts = line.split_whitespace();
    let r = parts.next()?.parse::<u16>().ok()?;
    let g = parts.next()?.parse::<u16>().ok()?;
    let b = parts.next()?.parse::<u16>().ok()?;
    if r > 255 || g > 255 || b > 255 {
        return None;
    }
    Some([r as u8, g as u8, b as u8])
}

/// Format a palette as one lowercase RRGGBB entry per line.
pub fn format_hex_list(palette: &[[u8; 3]]) -> String {
    let mut out = String::new();
    for [r, g, b] in palette {
        out.push_str(&format!("{r:02x}{g:02x}{b:02x}\n"));
    }
    out
}

/// Format a palette as a GIMP `.gpl` file.
pub fn format_gpl(palette: &[[u8; 3]], name: &str) -> String {
    let mut out = format!("GIMP Palette\nName: {name}\nColumns: 16\n#\n");
    for (i, [r, g, b]) in palette.iter().enumerate() {
        out.push_str(&format!("{r:3} {g:3} {b:3}\tColor {i}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_have_expected_sizes() {
        assert_eq!(GAMEBOY.len(), 4);
        assert_eq!(PICO8.len(), 16);
        assert_eq!(SWEETIE16.len(), 16);
        assert!(builtin("PICO-8").is_some());
        assert!(builtin("nope").is_none());
    }

    #[test]
    fn parses_hex_lists_with_hashes_comments_blanks() {
        let p = parse_hex_list("#1a1c2c\n\n; comment\n// also\nFFCCAA\n").unwrap();
        assert_eq!(p, vec![[0x1a, 0x1c, 0x2c], [0xff, 0xcc, 0xaa]]);
        assert!(parse_hex_list("xyz").is_err());
        assert!(parse_hex_list("; only comments\n").is_err());
    }

    #[test]
    fn parses_gimp_palette_rows() {
        let p = parse_palette_list(
            "GIMP Palette\nName: Demo\nColumns: 2\n# comment\n  26  28  44\tink\n255 204 170 skin\n",
        )
        .unwrap();
        assert_eq!(p, vec![[26, 28, 44], [255, 204, 170]]);
    }

    #[test]
    fn exports_hex_and_gpl_text() {
        let palette = [[0x1a, 0x1c, 0x2c], [0xff, 0xcc, 0xaa]];
        assert_eq!(format_hex_list(&palette), "1a1c2c\nffccaa\n");
        let gpl = format_gpl(&palette, "Demo");
        assert!(gpl.starts_with("GIMP Palette\nName: Demo\n"));
        assert!(gpl.contains(" 26  28  44\tColor 0"));
    }
}
