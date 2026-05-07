use sptorch_hal::{Backend, KernelProvider};
use sptorch_hal_ffi::FfiBackend;
use std::path::PathBuf;

fn target_debug_dir() -> PathBuf {
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir).join("debug");
    }

    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop(); // crates/
    dir.pop(); // project root
    dir.join("target").join("debug")
}

fn mock_npu_candidates() -> Vec<PathBuf> {
    let debug_dir = target_debug_dir();
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["mock_npu.dll", "sptorch_mock_npu.dll"]
    } else if cfg!(target_os = "macos") {
        &["libmock_npu.dylib", "libsptorch_mock_npu.dylib"]
    } else {
        &["libmock_npu.so", "libsptorch_mock_npu.so"]
    };

    names.iter().map(|name| debug_dir.join(name)).collect()
}

fn mock_npu_path() -> PathBuf {
    let candidates = mock_npu_candidates();
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

fn load_mock() -> FfiBackend {
    let path = mock_npu_path();
    assert!(
        path.exists(),
        "sptorch_mock_npu lib not found. Tried {:?}. Run `cargo build -p sptorch-mock-npu` first.",
        mock_npu_candidates()
    );
    FfiBackend::load(&path).expect("failed to load mock NPU backend")
}

#[test]
fn test_ffi_backend_name() {
    let backend = load_mock();
    assert_eq!(backend.name(), "mock_npu");
}

#[test]
fn test_ffi_add() {
    let backend = load_mock();
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let mut out = vec![0.0f32; 3];
    backend.add_f32(&a, &b, &mut out);
    assert_eq!(out, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_ffi_mul() {
    let backend = load_mock();
    let a = vec![2.0f32, 3.0];
    let b = vec![4.0f32, 5.0];
    let mut out = vec![0.0f32; 2];
    backend.mul_f32(&a, &b, &mut out);
    assert_eq!(out, vec![8.0, 15.0]);
}

#[test]
fn test_ffi_neg() {
    let backend = load_mock();
    let a = vec![1.0f32, -2.0, 3.0];
    let mut out = vec![0.0f32; 3];
    backend.neg_f32(&a, &mut out);
    assert_eq!(out, vec![-1.0, 2.0, -3.0]);
}

#[test]
fn test_ffi_scale() {
    let backend = load_mock();
    let a = vec![1.0f32, 2.0, 3.0];
    let mut out = vec![0.0f32; 3];
    backend.scale_f32(&a, 0.5, &mut out);
    assert_eq!(out, vec![0.5, 1.0, 1.5]);
}

#[test]
fn test_ffi_relu() {
    let backend = load_mock();
    let a = vec![-1.0f32, 0.0, 2.0, -0.5];
    let mut out = vec![0.0f32; 4];
    backend.relu_f32(&a, &mut out);
    assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
}

#[test]
fn test_ffi_matmul() {
    let backend = load_mock();
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3,2]
    let mut out = vec![0.0f32; 4];
    backend.matmul_f32(&a, &b, &mut out, 2, 3, 2);
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_ffi_softmax() {
    let backend = load_mock();
    let a = vec![1.0f32, 2.0, 3.0];
    let mut out = vec![0.0f32; 3];
    backend.softmax_f32(&a, &mut out, 1, 3);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(out[2] > out[1] && out[1] > out[0]);
}

#[test]
fn test_ffi_upload_download() {
    let backend = load_mock();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = backend.upload(&data).expect("upload failed");
    use sptorch_core_tensor::DeviceBuffer;
    let downloaded = buf.to_host();
    assert_eq!(downloaded, data);
}

#[test]
fn test_ffi_exp_log_roundtrip() {
    let backend = load_mock();
    let a = vec![0.0f32, 1.0, 2.0];
    let mut exp_out = vec![0.0f32; 3];
    backend.exp_f32(&a, &mut exp_out);
    let mut log_out = vec![0.0f32; 3];
    backend.log_f32(&exp_out, &mut log_out);
    for i in 0..3 {
        assert!((log_out[i] - a[i]).abs() < 1e-5, "exp->log roundtrip failed at {}", i);
    }
}

#[test]
fn test_ffi_query_runtime_optional_hook() {
    let backend = load_mock();
    let runtime = backend.query_runtime();
    assert!(runtime.is_some());
    let (queue_depth, online) = runtime.expect("runtime telemetry");
    assert_eq!(queue_depth, 0);
    assert!(online);
}
