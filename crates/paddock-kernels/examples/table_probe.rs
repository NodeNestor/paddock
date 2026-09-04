// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

// Throwaway diagnostic: load the CUDA pack and print which f8-family table
// entries resolved non-NULL on this device (gemma4 sm_120 lane gates).
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: table_probe <pack.so>");
    let pack = paddock_kernels::KernelPack::load(std::path::Path::new(&path)).expect("load pack");
    let t = pack.kernels_v1().expect("table");
    println!("f8_gemm_w8      {}", t.f8_gemm_w8.is_some());
    println!("q8_0_to_f8w     {}", t.q8_0_to_f8w.is_some());
    println!("quantize_e4m3   {}", t.quantize_e4m3.is_some());
    println!("f8_gemv         {}", t.f8_gemv.is_some());
    println!("f8_gemv_batch   {}", t.f8_gemv_batch.is_some());
    println!("f8_gemm_mma_ks  {}", t.f8_gemm_mma_ks.is_some());
    println!("f8d_gemm_mma_ks {}", t.f8d_gemm_mma_ks.is_some());
    println!("f8r_gemm_mma_ks {}", t.f8r_gemm_mma_ks.is_some());
    println!("f8w_repack_lin  {}", t.f8w_repack_lin.is_some());
    println!("f8_gemm_lin     {}", t.f8_gemm_lin.is_some());
    println!("mxfp4_gemm_bs   {}", t.mxfp4_gemm_bs.is_some());
    println!("q8_0_to_mxfp4   {}", t.q8_0_to_mxfp4.is_some());
    println!("f8t_gemm        {}", t.f8t_gemm.is_some());
    println!("f8row_gemm      {}", t.f8row_gemm.is_some());
}
