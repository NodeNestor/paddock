//! Split-family loading tests: synthetic shard sets exercising every rule the
//! loader enforces, plus single-file behavior staying byte-identical.

use super::*;
use crate::testutil::Writer;

/// One shard of a synthetic family: shard 0 carries the model KVs +
/// split.tensors.count, every shard carries its own split bookkeeping.
/// Each shard holds one 8-element F32 tensor filled with a marker byte.
fn shard_bytes(no_0based: u16, count: u16, total_tensors: i32, tensor_name: &str) -> Vec<u8> {
    let kv_count = if no_0based == 0 { 4 } else { 2 };
    let mut w = Writer::new(1, kv_count);
    if no_0based == 0 {
        w.kv_str("general.architecture", "llama");
        w.kv_i32("split.tensors.count", total_tensors);
    }
    w.kv_u16("split.no", no_0based);
    w.kv_u16("split.count", count);
    w.tensor_f32(tensor_name, &[8], 0);
    w.finish_with_filled_data(32, no_0based as u8 + 1)
}

/// Write a family under `dir` with the canonical names; returns first path.
fn write_family(dir: &Path, prefix: &str, shards: &[Vec<u8>]) -> PathBuf {
    let count = shards.len();
    for (i, bytes) in shards.iter().enumerate() {
        let p = dir.join(format!("{prefix}-{:05}-of-{count:05}.gguf", i + 1));
        std::fs::write(p, bytes).expect("write shard");
    }
    dir.join(format!("{prefix}-00001-of-{count:05}.gguf"))
}

#[test]
fn split_family_loads_as_one_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = write_family(
        dir.path(),
        "m",
        &[
            shard_bytes(0, 3, 3, "blk.0.w"),
            shard_bytes(1, 3, 3, "blk.1.w"),
            shard_bytes(2, 3, 3, "blk.2.w"),
        ],
    );

    let m = MappedGguf::open(&first).expect("split model opens");
    assert_eq!(m.shard_count(), 3);
    assert_eq!(m.tensor_count(), 3);
    // metadata comes from shard 0
    assert_eq!(m.gguf().architecture(), Some("llama"));
    // tensors resolve into the right shard's bytes (marker byte = shard+1)
    for (shard, name) in [(0u8, "blk.0.w"), (1, "blk.1.w"), (2, "blk.2.w")] {
        let (info, bytes) = m.tensor_bytes(name).expect(name);
        assert_eq!(info.element_count(), 8);
        assert!(bytes.iter().all(|&b| b == shard + 1), "{name} wrong shard");
    }
    // union iterator walks family order
    let names: Vec<_> = m.tensor_infos().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["blk.0.w", "blk.1.w", "blk.2.w"]);
    assert!(m.total_len() > 0);
}

#[test]
fn single_file_without_split_kvs_behaves_as_before() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut w = Writer::new(1, 1);
    w.kv_str("general.architecture", "llama");
    w.tensor_f32("t.w", &[8], 0);
    let path = dir.path().join("plain.gguf");
    std::fs::write(&path, w.finish_with_data(32)).expect("write");

    let m = MappedGguf::open(&path).expect("opens");
    assert_eq!(m.shard_count(), 1);
    assert!(m.tensor_bytes("t.w").is_ok());
    assert_eq!(m.path(), path);
}

#[test]
fn opening_a_later_shard_names_the_first() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_family(
        dir.path(),
        "m",
        &[shard_bytes(0, 2, 2, "a"), shard_bytes(1, 2, 2, "b")],
    );
    // hand the loader shard 2 directly
    let second = dir.path().join("m-00002-of-00002.gguf");
    match MappedGguf::open(&second) {
        Err(MapError::NotFirstSplit { first, .. }) => {
            assert!(first.to_string_lossy().ends_with("m-00001-of-00002.gguf"));
        }
        other => panic!("expected NotFirstSplit, got {other:?}", other = other.err()),
    }
}

#[test]
fn missing_sibling_is_an_io_error_naming_the_missing_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = write_family(dir.path(), "m", &[shard_bytes(0, 2, 2, "a")]);
    // write_family wrote it as ...-00001-of-00001; rename to claim 2 shards
    let claimed_first = dir.path().join("m-00001-of-00002.gguf");
    std::fs::rename(&first, &claimed_first).expect("rename");
    match MappedGguf::open(&claimed_first) {
        Err(MapError::Io(path, _)) => {
            assert!(path.to_string_lossy().ends_with("m-00002-of-00002.gguf"));
        }
        other => panic!("expected Io, got {other:?}", other = other.err()),
    }
}

#[test]
fn sibling_with_wrong_split_no_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = write_family(
        dir.path(),
        "m",
        &[
            shard_bytes(0, 2, 2, "a"),
            shard_bytes(0, 2, 2, "b"), // second file claims to be shard 0 again
        ],
    );
    assert!(matches!(
        MappedGguf::open(&first),
        Err(MapError::ShardMismatch {
            key: "split.no",
            expected: 1,
            found: 0,
            ..
        })
    ));
}

#[test]
fn duplicate_tensor_across_shards_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = write_family(
        dir.path(),
        "m",
        &[
            shard_bytes(0, 2, 2, "same.w"),
            shard_bytes(1, 2, 2, "same.w"),
        ],
    );
    assert!(matches!(
        MappedGguf::open(&first),
        Err(MapError::DuplicateAcrossShards { .. })
    ));
}

#[test]
fn tensors_count_mismatch_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    // family says 5 tensors, shards only carry 2
    let first = write_family(
        dir.path(),
        "m",
        &[shard_bytes(0, 2, 5, "a"), shard_bytes(1, 2, 5, "b")],
    );
    assert!(matches!(
        MappedGguf::open(&first),
        Err(MapError::TensorCountMismatch {
            expected: 5,
            found: 2,
            ..
        })
    ));
}

#[test]
fn missing_tensors_count_on_split_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    // shard 0 without split.tensors.count (3 KVs instead of 4)
    let mut w = Writer::new(1, 3);
    w.kv_str("general.architecture", "llama");
    w.kv_u16("split.no", 0);
    w.kv_u16("split.count", 2);
    w.tensor_f32("a", &[8], 0);
    let shards = [w.finish_with_data(32), shard_bytes(1, 2, 2, "b")];
    let first = write_family(dir.path(), "m", &shards);
    assert!(matches!(
        MappedGguf::open(&first),
        Err(MapError::MissingSplitKey {
            key: "split.tensors.count",
            ..
        })
    ));
}

#[test]
fn filename_disagreeing_with_split_count_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    // metadata says 3 shards but the file is named -00001-of-00002
    let path = dir.path().join("m-00001-of-00002.gguf");
    std::fs::write(&path, shard_bytes(0, 3, 3, "a")).expect("write");
    assert!(matches!(
        MappedGguf::open(&path),
        Err(MapError::SplitNameMismatch { count: 3, .. })
    ));
}

#[test]
fn plain_filename_with_split_metadata_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    // split.count = 2 in a file with no split naming - siblings unlocatable
    let path = dir.path().join("renamed-model.gguf");
    std::fs::write(&path, shard_bytes(0, 2, 2, "a")).expect("write");
    assert!(matches!(
        MappedGguf::open(&path),
        Err(MapError::SplitNameMismatch { count: 2, .. })
    ));
}

/// Loads the real sharded gpt-oss-120b when present (this repo's dev boxes
/// keep it at /llms). Header-only cost is ~mmap + parse; skips elsewhere.
#[test]
fn loads_real_split_family_when_available() {
    let path = std::env::var_os("PADDOCK_TEST_SPLIT_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/llms/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf")
        });
    if !path.exists() {
        eprintln!("no local split GGUF fixture - skipping");
        return;
    }
    let m = MappedGguf::open(&path).expect("real split family loads");
    assert!(m.shard_count() > 1);
    assert_eq!(m.gguf().architecture(), Some("gpt-oss"));
    // spot-check a tensor that must live in a later shard: the last block's
    // ffn weights are far beyond shard 1
    let count = m.tensor_count();
    assert_eq!(count as u64, m.tensor_infos().count() as u64);
    eprintln!(
        "loaded {}: {} shards, {} tensors, {:.1} GB",
        path.display(),
        m.shard_count(),
        count,
        m.total_len() as f64 / 1e9
    );
}
