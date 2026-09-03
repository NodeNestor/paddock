//! Pixel-domain forensic analyzers. Wave 0 ships ELA; later waves add noise,
//! JPEG-ghost, double-JPEG, copy-move, splice, PRNU, etc.

pub mod anti_forensics;
pub mod cfa;
pub mod channel_correlation;
pub mod chromatic;
pub mod color_consistency;
pub mod copy_move;
pub mod document_forensics;
pub mod dof_consistency;
pub mod double_jpeg;
pub mod edge_sharpness;
pub mod ela;
pub mod font_consistency;
pub mod frequency;
pub mod geometric;
pub mod histogram_analysis;
pub mod illumination;
pub mod jpeg_forensics;
pub mod jpeg_ghost;
pub mod lighting_consistency;
pub mod noise;
pub mod paste_rectangle;
pub mod prnu;
pub mod prnu_cross_region;
pub mod qtable_fingerprint;
pub mod resampling;
pub mod screenshot_detection;
pub mod shadow_consistency;
pub mod splice_boundary;
pub mod text_alignment;
pub mod texture;
pub mod thumbnail_check;
pub mod upscaling_detection;
pub mod vanishing_point;
pub mod wavelet_consistency;
