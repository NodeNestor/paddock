//! Dump a GGUF's arch metadata + tensor-name summary - bring-up scoping tool.
//! Usage: gguf_meta <path.gguf> [name-filter]

use paddock_models::mapped::MappedGguf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_meta <path.gguf> [filter]");
    let filter = std::env::args().nth(2);
    let map = MappedGguf::open(std::path::Path::new(&path)).expect("open gguf");
    let g = map.gguf();

    let mut keys: Vec<_> = g.metadata.keys().collect();
    keys.sort();
    for k in keys {
        // tokenizer arrays are huge - show scalars/short values only. Set
        // PADDOCK_META_FULL to print everything: the length cut silently hides
        // mid-size arrays that matter (granite.deepstack_mapping is 40 entries
        // and was invisible here, which briefly read as "the key is absent").
        let s = format!("{:?}", g.metadata[k]);
        if s.len() <= 120 || std::env::var_os("PADDOCK_META_FULL").is_some() {
            println!("{k} = {s}");
        }
    }
    println!("---- tensors ----");
    let mut shown = 0;
    // union across shards - a split family walks all its files
    for info in map.tensor_infos() {
        let name = &info.name;
        let keep = match &filter {
            Some(f) => name.contains(f.as_str()),
            // default: block 0 + everything outside blk.* (embd, norms, nextn)
            None => !name.starts_with("blk.") || name.starts_with("blk.0."),
        };
        if keep {
            println!("{name}  dims {:?}  {:?}", info.dims, info.ggml_type);
            shown += 1;
        }
    }
    println!(
        "({} tensors shown of {}, {} file(s))",
        shown,
        map.tensor_count(),
        map.shard_count()
    );
}
