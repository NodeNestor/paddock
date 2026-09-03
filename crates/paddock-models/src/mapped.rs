//! Memory-mapped GGUF access: parsed header + zero-copy tensor byte ranges,
//! transparently spanning split files (`-00001-of-00003.gguf` families).
//!
//! The mmaps are the single source of weight bytes for every consumer - CPU
//! reference dequants read straight from them, GPU upload streams from them.
//! A split model opens one map per shard and resolves each tensor to its
//! shard; single-file models keep the exact old behavior. Ranges were
//! validated at parse time; `tensor_bytes` re-checks anyway (defense in
//! depth, a file could have been truncated after probing).
//!
//! Split semantics mirror the defining implementation (llama.cpp b9969
//! model loader, studied not copied): must open via the first shard,
//! `split.no` is 0-based and must match each file's position, and the
//! first shard's `split.tensors.count` must equal the union - a missing or
//! wrong shard is a load error, never a silent partial model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::gguf::{GgufError, GgufFile, TensorInfo, Value};
use crate::split::parse_split_name;

const KV_SPLIT_NO: &str = "split.no";
const KV_SPLIT_COUNT: &str = "split.count";
const KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// Shard-count sanity cap: the format field is u16 and real families are
/// single digits; thousands of mmaps means a corrupt header, not a model.
const MAX_SPLITS: u64 = 2048;

#[derive(Debug, thiserror::Error)]
pub enum MapError {
    #[error("cannot open {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("{0}: {1}")]
    Parse(PathBuf, GgufError),
    #[error("tensor {0} not present in file")]
    NoSuchTensor(String),
    #[error("tensor {name} range {offset}+{len} exceeds file size {file_len}")]
    OutOfRange {
        name: String,
        offset: u64,
        len: u64,
        file_len: u64,
    },
    #[error("tensor {0} has an unverified type layout - cannot size its data")]
    Unsizable(String),
    #[error(
        "{path}: this is shard {no_1based} of a {count}-file split model - \
         load it via the first shard ({first})"
    )]
    NotFirstSplit {
        path: PathBuf,
        no_1based: u32,
        count: u32,
        first: PathBuf,
    },
    #[error(
        "{path}: declares split.count = {count} but the filename does not \
         follow the ...-00001-of-{count:05}.gguf convention needed to locate \
         its siblings"
    )]
    SplitNameMismatch { path: PathBuf, count: u64 },
    #[error("{path}: shard metadata {key} = {found}, expected {expected}")]
    ShardMismatch {
        path: PathBuf,
        key: &'static str,
        expected: u64,
        found: u64,
    },
    #[error("{path}: split model is missing required metadata key {key}")]
    MissingSplitKey { path: PathBuf, key: &'static str },
    #[error(
        "split model corrupt: split.tensors.count says {expected} tensors \
         but the {shards} shards contain {found}"
    )]
    TensorCountMismatch {
        expected: u64,
        found: u64,
        shards: usize,
    },
    #[error("tensor {name} appears in both shard {first_shard} and shard {second_shard}")]
    DuplicateAcrossShards {
        name: String,
        first_shard: usize,
        second_shard: usize,
    },
    #[error("split.count {count} exceeds sanity cap {MAX_SPLITS} - corrupt or hostile file")]
    TooManySplits { count: u64 },
}

/// One mmapped file of the model (the whole model, or one shard of it).
struct Shard {
    mmap: memmap2::Mmap,
    gguf: GgufFile,
    path: PathBuf,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, MapError> {
        let file = std::fs::File::open(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
        // SAFETY: read-only map; underlying file mutation during use is the
        // documented platform caveat (same stance as every mmap-based loader)
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| MapError::Io(path.to_path_buf(), e))?;
        let gguf = GgufFile::parse(&mmap).map_err(|e| MapError::Parse(path.to_path_buf(), e))?;
        Ok(Self {
            mmap,
            gguf,
            path: path.to_path_buf(),
        })
    }

    fn split_kv(&self, key: &'static str) -> Option<u64> {
        self.gguf.metadata.get(key).and_then(Value::as_u64)
    }
}

/// An open, parsed, memory-mapped GGUF model - one file or a split family.
///
/// Metadata (`gguf()`) always comes from the first shard, which is where the
/// writer puts the model KVs; later shards only carry their own tensor
/// directories plus split bookkeeping.
pub struct MappedGguf {
    shards: Vec<Shard>,
    /// tensor name -> (shard index, index into that shard's tensor list)
    by_name: HashMap<String, (usize, usize)>,
}

impl MappedGguf {
    pub fn open(path: &Path) -> Result<Self, MapError> {
        let first = Shard::open(path)?;

        let split_count = first.split_kv(KV_SPLIT_COUNT).unwrap_or(0);
        let mut shards = if split_count > 1 {
            Self::open_siblings(first, split_count)?
        } else {
            vec![first]
        };

        let mut by_name = HashMap::new();
        for (si, shard) in shards.iter().enumerate() {
            for (ti, t) in shard.gguf.tensors.iter().enumerate() {
                if let Some(&(prev, _)) = by_name.get(&t.name) {
                    return Err(MapError::DuplicateAcrossShards {
                        name: t.name.clone(),
                        first_shard: prev,
                        second_shard: si,
                    });
                }
                by_name.insert(t.name.clone(), (si, ti));
            }
        }

        // the first shard declares the family-wide tensor total; llama.cpp
        // treats it as required for splits and so do we - a missing sibling
        // that somehow parses must never load as a smaller model
        if split_count > 1 {
            let expected =
                shards[0]
                    .split_kv(KV_SPLIT_TENSORS_COUNT)
                    .ok_or(MapError::MissingSplitKey {
                        path: shards[0].path.clone(),
                        key: KV_SPLIT_TENSORS_COUNT,
                    })?;
            if expected != by_name.len() as u64 {
                return Err(MapError::TensorCountMismatch {
                    expected,
                    found: by_name.len() as u64,
                    shards: shards.len(),
                });
            }
        }

        shards.shrink_to_fit();
        Ok(Self { shards, by_name })
    }

    /// Resolve and open shards 2..N of a split family, verifying each file is
    /// the shard its name claims. `first` was already parsed by the caller.
    fn open_siblings(first: Shard, split_count: u64) -> Result<Vec<Shard>, MapError> {
        if split_count > MAX_SPLITS {
            return Err(MapError::TooManySplits { count: split_count });
        }
        // split.no is 0-based; the KV is required on every shard of a split
        let no = first
            .split_kv(KV_SPLIT_NO)
            .ok_or(MapError::MissingSplitKey {
                path: first.path.clone(),
                key: KV_SPLIT_NO,
            })?;

        // the filename is the only sibling locator the format defines, so it
        // must agree with the metadata before we derive paths from it
        let name = match parse_split_name(&first.path) {
            Some(n) if u64::from(n.count) == split_count => n,
            _ => {
                return Err(MapError::SplitNameMismatch {
                    path: first.path.clone(),
                    count: split_count,
                });
            }
        };

        if no != 0 {
            return Err(MapError::NotFirstSplit {
                path: first.path.clone(),
                no_1based: name.no_1based,
                count: name.count,
                first: name.sibling(1),
            });
        }

        let mut shards = Vec::with_capacity(split_count as usize);
        shards.push(first);
        for idx in 1..split_count {
            let sibling_path = name.sibling(idx as u32 + 1);
            let shard = Shard::open(&sibling_path)?;
            Self::check_shard_kv(&shard, KV_SPLIT_NO, idx)?;
            Self::check_shard_kv(&shard, KV_SPLIT_COUNT, split_count)?;
            shards.push(shard);
        }
        Ok(shards)
    }

    fn check_shard_kv(shard: &Shard, key: &'static str, expected: u64) -> Result<(), MapError> {
        let found = shard.split_kv(key).ok_or(MapError::MissingSplitKey {
            path: shard.path.clone(),
            key,
        })?;
        if found != expected {
            return Err(MapError::ShardMismatch {
                path: shard.path.clone(),
                key,
                expected,
                found,
            });
        }
        Ok(())
    }

    /// Model metadata: the first shard's header (arch, hparams, tokenizer -
    /// the writer puts all model KVs there). Its `tensors` list covers only
    /// that shard; use [`tensor_infos`](Self::tensor_infos) for the union.
    pub fn gguf(&self) -> &GgufFile {
        &self.shards[0].gguf
    }

    /// The path this model was opened from (the first shard for splits).
    pub fn path(&self) -> &Path {
        &self.shards[0].path
    }

    /// Number of files backing this model (1 unless split).
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Total bytes across all backing files.
    pub fn total_len(&self) -> u64 {
        self.shards.iter().map(|s| s.mmap.len() as u64).sum()
    }

    /// Every tensor in the model, shards in family order, file order within
    /// each shard - the union view consumers should iterate.
    pub fn tensor_infos(&self) -> impl Iterator<Item = &TensorInfo> {
        self.shards.iter().flat_map(|s| s.gguf.tensors.iter())
    }

    pub fn tensor_count(&self) -> usize {
        self.by_name.len()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let &(si, ti) = self.by_name.get(name)?;
        Some(&self.shards[si].gguf.tensors[ti])
    }

    /// Raw quantized/typed bytes of a tensor, straight from its shard's map.
    pub fn tensor_bytes(&self, name: &str) -> Result<(&TensorInfo, &[u8]), MapError> {
        let &(si, ti) = self
            .by_name
            .get(name)
            .ok_or_else(|| MapError::NoSuchTensor(name.to_owned()))?;
        let shard = &self.shards[si];
        let info = &shard.gguf.tensors[ti];
        let len = info
            .byte_size()
            .ok_or_else(|| MapError::Unsizable(name.to_owned()))?;
        let start = shard.gguf.data_offset + info.offset;
        let end = start + len;
        if end > shard.mmap.len() as u64 {
            return Err(MapError::OutOfRange {
                name: name.to_owned(),
                offset: info.offset,
                len,
                file_len: shard.mmap.len() as u64,
            });
        }
        Ok((info, &shard.mmap[start as usize..end as usize]))
    }
}

#[cfg(test)]
mod tests;
