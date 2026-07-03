//! Built-in palettes + Lospec-style hex list parsing ("RRGGBB" or "#RRGGBB",
//! one per line, blank lines and ';' / '//' comments ignored).

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
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let line = line.strip_prefix('#').unwrap_or(line);
        if line.is_empty() || line.starts_with(';') || line.starts_with("//") {
            continue;
        }
        if line.len() != 6 || !line.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("line {}: expected RRGGBB, got {raw:?}", i + 1));
        }
        let v = u32::from_str_radix(line, 16).unwrap();
        out.push([(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    }
    if out.is_empty() {
        return Err("palette file contained no colors".into());
    }
    Ok(out)
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
}
