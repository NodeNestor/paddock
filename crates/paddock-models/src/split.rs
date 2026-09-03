//! The GGUF split-file naming convention (llama.cpp `gguf-split`).
//!
//! A model larger than a host's file-size comfort zone (HF caps uploads at
//! ~50 GB) ships as N complete GGUF files named `<prefix>-%05d-of-%05d.gguf`,
//! 1-based in the name. Each shard carries `split.no` (0-based), `split.count`
//! and - on the first shard - `split.tensors.count` metadata; tensors are
//! whole per shard, never split across files. Reference semantics:
//! llama.cpp `llama_split_path`/`llama_split_prefix` + `llama-model-loader.cpp`
//! (studied at b9969, no code copied).

use std::path::{Path, PathBuf};

/// A shard filename decomposed: `/x/m-00002-of-00003.gguf` ->
/// `{ prefix: "/x/m", no_1based: 2, count: 3 }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitName {
    /// Everything before `-%05d-of-%05d.gguf`, directory included.
    pub prefix: String,
    pub no_1based: u32,
    pub count: u32,
}

impl SplitName {
    /// Path of sibling shard `no` (1-based) in the same family.
    pub fn sibling(&self, no_1based: u32) -> PathBuf {
        PathBuf::from(format!(
            "{}-{:05}-of-{:05}.gguf",
            self.prefix, no_1based, self.count
        ))
    }
}

/// Strict parse of the split naming convention; `None` for ordinary GGUFs.
/// Exactly five digits per field, both fields > 0, `no <= count` - anything
/// looser would misfire on models with number-bearing names.
pub fn parse_split_name(path: &Path) -> Option<SplitName> {
    let s = path.to_str()?;
    let body = s.strip_suffix(".gguf")?;
    // "...-NNNNN-of-MMMMM": fixed width makes the reverse parse unambiguous
    let (rest, count_s) = body.split_at(body.len().checked_sub(5)?);
    let rest = rest.strip_suffix("-of-")?;
    let (prefix, no_s) = rest.split_at(rest.len().checked_sub(5)?);
    let prefix = prefix.strip_suffix('-')?;
    if !count_s.bytes().all(|b| b.is_ascii_digit()) || !no_s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let no_1based: u32 = no_s.parse().ok()?;
    let count: u32 = count_s.parse().ok()?;
    if prefix.is_empty() || no_1based == 0 || count == 0 || no_1based > count {
        return None;
    }
    Some(SplitName {
        prefix: prefix.to_owned(),
        no_1based,
        count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_split_names() {
        let n = parse_split_name(Path::new("/m/gpt-oss-120b-mxfp4-00001-of-00003.gguf"))
            .expect("parses");
        assert_eq!(n.prefix, "/m/gpt-oss-120b-mxfp4");
        assert_eq!((n.no_1based, n.count), (1, 3));
        assert_eq!(
            n.sibling(2),
            PathBuf::from("/m/gpt-oss-120b-mxfp4-00002-of-00003.gguf")
        );
    }

    #[test]
    fn rejects_lookalikes() {
        for name in [
            "model.gguf",                   // no suffix at all
            "model-1-of-3.gguf",            // not five digits
            "model-00000-of-00003.gguf",    // shard numbers are 1-based
            "model-00004-of-00003.gguf",    // no > count
            "model-00001-of-00000.gguf",    // zero count
            "-00001-of-00002.gguf",         // empty prefix
            "model-00001-of-00003.notgguf", // wrong extension
            "model-abcde-of-00003.gguf",    // non-digits
        ] {
            assert!(parse_split_name(Path::new(name)).is_none(), "{name}");
        }
    }
}
