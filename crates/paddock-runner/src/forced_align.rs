//! Forced-alignment text machinery: the word splitter and the
//! timestamp monotonicity repair, both faithful ports of transformers main
//! `processing_qwen3_asr.py` (itself lifted from the upstream QwenLM/Qwen3-ASR
//! `qwen3_forced_aligner.py`). The engine returns raw time-bin argmaxes; this
//! module owns everything linguistic on either side of that call.
//!
//! Fixture-pinned against the HF reference on the asr-battery (the oracle
//! dump bring-up): the tests below carry real model output,
//! including out-of-order predictions the repair has to fix.

/// True for CJK ideographs - these split to one "word" per character.
/// Codepoint ranges as the reference's `_is_cjk_char`.
fn is_cjk(c: char) -> bool {
    matches!(u32::from(c),
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F)
}

/// Characters kept inside a word: letters, numbers, apostrophes, CJK.
/// Punctuation and symbols drop - a timestamp for a comma is not a thing.
fn is_kept(c: char) -> bool {
    c == '\'' || c.is_alphanumeric() || is_cjk(c)
}

/// Split a transcript into alignment units: CJK characters individually,
/// space-delimited words otherwise, punctuation dropped. This is the
/// reference's DEFAULT tokenizer - its Japanese/Korean paths use external
/// morphological libraries (nagisa/soynlp) we deliberately do not carry, so
/// ja/ko text goes through this splitter too and the endpoint says so.
pub fn split_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        if is_cjk(c) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(c.to_string());
        } else if c.is_whitespace() {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
        } else if is_kept(c) {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Make the predicted timestamps monotonically non-decreasing - the
/// reference's `_fix_timestamps`, value for value:
/// 1. longest non-decreasing subsequence (O(n²) DP) = the "good" values;
/// 2. outlier blocks of ≤2 snap to the nearer good neighbour;
/// 3. longer blocks interpolate linearly between the surrounding good values.
///    Input and output are milliseconds (bin × segment_ms).
pub fn fix_timestamps(raw: &[f64]) -> Vec<i64> {
    let n = raw.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for cur in 1..n {
        for prev in 0..cur {
            if raw[prev] <= raw[cur] && dp[prev] + 1 > dp[cur] {
                dp[cur] = dp[prev] + 1;
                parent[cur] = prev;
            }
        }
    }
    // the reference takes the first index of maximal length
    // (`list.index(max)`) - ties resolve low
    let max_len = *dp.iter().max().expect("non-empty");
    let mut idx = dp.iter().position(|&l| l == max_len).unwrap_or(0);
    let mut normal = vec![false; n];
    while idx != usize::MAX {
        normal[idx] = true;
        idx = parent[idx];
    }

    let mut result: Vec<f64> = raw.to_vec();
    let mut start = 0usize;
    while start < n {
        if normal[start] {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < n && !normal[end] {
            end += 1;
        }
        let left = (0..start).rev().find(|&i| normal[i]).map(|i| result[i]);
        let right = (end..n).find(|&i| normal[i]).map(|i| result[i]);
        let count = end - start;
        if count <= 2 {
            for pos in start..end {
                result[pos] = match (left, right) {
                    (None, Some(r)) => r,
                    (Some(l), None) => l,
                    (Some(l), Some(r)) => {
                        // distance measured in slots from the block's edges,
                        // exactly the reference's tie-break (ties go left)
                        if (pos as i64 - (start as i64 - 1)) <= (end as i64 - pos as i64) {
                            l
                        } else {
                            r
                        }
                    }
                    (None, None) => result[pos],
                };
            }
        } else {
            match (left, right) {
                (Some(l), Some(r)) => {
                    let step = (r - l) / (count + 1) as f64;
                    for pos in start..end {
                        result[pos] = l + step * (pos - start + 1) as f64;
                    }
                }
                (Some(l), None) => {
                    for pos in start..end {
                        result[pos] = l;
                    }
                }
                (None, Some(r)) => {
                    for pos in start..end {
                        result[pos] = r;
                    }
                }
                (None, None) => {}
            }
        }
        start = end;
    }
    // the reference truncates (`int(val)`), not rounds
    result.into_iter().map(|v| v as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ls-short-1 off the battery oracle: two ≤2-length outlier blocks
    /// (21->20 snaps up, 28->24 snaps up) - real model output, real repair.
    #[test]
    fn repair_matches_reference_en() {
        let raw: Vec<f64> = [
            7, 15, 15, 21, 20, 22, 22, 28, 24, 28, 29, 34, 34, 35, 35, 42,
        ]
        .iter()
        .map(|&b| b as f64 * 80.0)
        .collect();
        let want: Vec<i64> = vec![
            560, 1200, 1200, 1680, 1680, 1760, 1760, 2240, 2240, 2240, 2320, 2720, 2720, 2800,
            2800, 3360,
        ];
        assert_eq!(fix_timestamps(&raw), want);
    }

    /// mls-de: 64 slots with two single-slot snaps (46->45, 72->71).
    #[test]
    fn repair_matches_reference_de() {
        let raw: Vec<f64> = [
            6, 10, 10, 16, 17, 24, 25, 32, 32, 36, 36, 43, 43, 46, 45, 50, 50, 53, 53, 63, 70, 72,
            71, 74, 75, 81, 81, 87, 90, 96, 96, 106, 106, 109, 109, 117, 128, 134, 141, 143, 143,
            147, 147, 150, 150, 158, 165, 170, 170, 181, 186, 188, 188, 191, 191, 193, 202, 204,
            204, 207, 207, 211, 212, 219,
        ]
        .iter()
        .map(|&b: &i64| b as f64 * 80.0)
        .collect();
        let want: Vec<i64> = vec![
            480, 800, 800, 1280, 1360, 1920, 2000, 2560, 2560, 2880, 2880, 3440, 3440, 3680, 3680,
            4000, 4000, 4240, 4240, 5040, 5600, 5760, 5760, 5920, 6000, 6480, 6480, 6960, 7200,
            7680, 7680, 8480, 8480, 8720, 8720, 9360, 10240, 10720, 11280, 11440, 11440, 11760,
            11760, 12000, 12000, 12640, 13200, 13600, 13600, 14480, 14880, 15040, 15040, 15280,
            15280, 15440, 16160, 16320, 16320, 16560, 16560, 16880, 16960, 17520,
        ];
        assert_eq!(fix_timestamps(&raw), want);
    }

    #[test]
    fn monotone_input_is_identity() {
        let raw = [10.0, 20.0, 20.0, 35.5];
        assert_eq!(fix_timestamps(&raw), vec![10, 20, 20, 35]);
    }

    #[test]
    fn splitter_words_and_cjk() {
        assert_eq!(
            split_words("Hello, world's  fine."),
            vec!["Hello", "world's", "fine"]
        );
        // CJK chars split individually even without spaces; latin runs stay
        // words; punctuation drops entirely
        assert_eq!(
            split_words("你好ok世界"),
            vec!["你", "好", "ok", "世", "界"]
        );
        assert_eq!(split_words("-..."), Vec::<String>::new());
    }
}
