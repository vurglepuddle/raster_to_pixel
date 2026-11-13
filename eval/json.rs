//! Small JSON reader/writer for our own artifacts; no dependency or shell escaping.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(BTreeMap<String, J>),
}
impl From<&str> for J {
    fn from(v: &str) -> Self {
        Self::Str(v.into())
    }
}
impl From<String> for J {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}
impl From<bool> for J {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<f64> for J {
    fn from(v: f64) -> Self {
        if v.is_finite() {
            Self::Num(v)
        } else {
            Self::Null
        }
    }
}
impl From<u32> for J {
    fn from(v: u32) -> Self {
        Self::Num(v as f64)
    }
}
impl From<usize> for J {
    fn from(v: usize) -> Self {
        Self::Num(v as f64)
    }
}
pub fn obj(entries: impl IntoIterator<Item = (impl Into<String>, J)>) -> J {
    J::Obj(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
impl J {
    pub fn get(&self, key: &str) -> &J {
        if let Self::Obj(m) = self {
            m.get(key).unwrap_or(&J::Null)
        } else {
            &J::Null
        }
    }
    pub fn s(&self) -> &str {
        if let Self::Str(v) = self {
            v
        } else {
            ""
        }
    }
    pub fn n(&self) -> Option<f64> {
        if let Self::Num(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn a(&self) -> &[J] {
        if let Self::Arr(v) = self {
            v
        } else {
            &[]
        }
    }
    pub fn set(&mut self, key: &str, value: impl Into<J>) {
        if let Self::Obj(m) = self {
            m.insert(key.into(), value.into());
        }
    }
    pub fn dump(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(v) => v.to_string(),
            Self::Num(v) => v.to_string(),
            Self::Str(v) => {
                let mut out = String::from("\"");
                for c in v.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
                out
            }
            Self::Arr(v) => format!("[{}]", v.iter().map(J::dump).collect::<Vec<_>>().join(",")),
            Self::Obj(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!("{}:{}", J::from(k.clone()).dump(), v.dump()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}
pub fn parse(s: &str) -> Result<J, String> {
    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl Parser<'_> {
        fn ws(&mut self) {
            while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
                self.i += 1
            }
        }
        fn take(&mut self, c: u8) -> bool {
            self.ws();
            if self.b.get(self.i) == Some(&c) {
                self.i += 1;
                true
            } else {
                false
            }
        }
        fn string(&mut self) -> Result<String, String> {
            if !self.take(b'"') {
                return Err("expected JSON string".into());
            }
            let mut v = Vec::new();
            while let Some(&c) = self.b.get(self.i) {
                self.i += 1;
                match c {
                    b'"' => return String::from_utf8(v).map_err(|e| e.to_string()),
                    b'\\' => {
                        let e = *self.b.get(self.i).ok_or("truncated JSON escape")?;
                        self.i += 1;
                        match e {
                            b'"' | b'\\' | b'/' => v.push(e),
                            b'n' => v.push(b'\n'),
                            b'r' => v.push(b'\r'),
                            b't' => v.push(b'\t'),
                            b'b' => v.push(8),
                            b'f' => v.push(12),
                            b'u' => {
                                let h =
                                    self.b.get(self.i..self.i + 4).ok_or("truncated unicode")?;
                                let n = u32::from_str_radix(
                                    std::str::from_utf8(h).map_err(|e| e.to_string())?,
                                    16,
                                )
                                .map_err(|e| e.to_string())?;
                                self.i += 4;
                                let c = char::from_u32(n).ok_or("unsupported surrogate escape")?;
                                let mut buf = [0; 4];
                                v.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return Err("invalid JSON escape".into()),
                        }
                    }
                    0..=31 => return Err("control byte in JSON string".into()),
                    _ => v.push(c),
                }
            }
            Err("unterminated JSON string".into())
        }
        fn val(&mut self, depth: usize) -> Result<J, String> {
            if depth > 64 {
                return Err("JSON nesting limit".into());
            }
            self.ws();
            match self.b.get(self.i).copied().ok_or("truncated JSON")? {
                b'"' => Ok(J::Str(self.string()?)),
                b'{' => {
                    self.i += 1;
                    let mut m = BTreeMap::new();
                    if self.take(b'}') {
                        return Ok(J::Obj(m));
                    }
                    loop {
                        let k = self.string()?;
                        if !self.take(b':') {
                            return Err("expected colon".into());
                        }
                        m.insert(k, self.val(depth + 1)?);
                        if self.take(b'}') {
                            break;
                        }
                        if !self.take(b',') {
                            return Err("expected comma".into());
                        }
                    }
                    Ok(J::Obj(m))
                }
                b'[' => {
                    self.i += 1;
                    let mut a = Vec::new();
                    if self.take(b']') {
                        return Ok(J::Arr(a));
                    }
                    loop {
                        a.push(self.val(depth + 1)?);
                        if self.take(b']') {
                            break;
                        }
                        if !self.take(b',') {
                            return Err("expected comma".into());
                        }
                    }
                    Ok(J::Arr(a))
                }
                b't' | b'f' | b'n' => {
                    for (token, v) in [
                        (b"true".as_slice(), J::Bool(true)),
                        (b"false", J::Bool(false)),
                        (b"null", J::Null),
                    ] {
                        if self.b[self.i..].starts_with(token) {
                            self.i += token.len();
                            return Ok(v);
                        }
                    }
                    Err("invalid JSON literal".into())
                }
                _ => {
                    let start = self.i;
                    while self.b.get(self.i).is_some_and(|c| {
                        matches!(c, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                    }) {
                        self.i += 1
                    }
                    let n = std::str::from_utf8(&self.b[start..self.i])
                        .map_err(|e| e.to_string())?
                        .parse::<f64>()
                        .map_err(|e| e.to_string())?;
                    if !n.is_finite() {
                        return Err("non-finite JSON number".into());
                    }
                    Ok(J::Num(n))
                }
            }
        }
    }
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    let j = p.val(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err("trailing JSON".into());
    }
    Ok(j)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_round_trip() {
        let j = obj([
            ("unicode", J::from("猫\n\"\\")),
            ("array", J::Arr(vec![2usize.into(), J::Null])),
        ]);
        assert_eq!(parse(&j.dump()).unwrap(), j);
        assert!(parse("{oops").is_err());
    }
}
