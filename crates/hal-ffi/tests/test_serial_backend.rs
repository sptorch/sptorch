use sptorch_core_ops::matmul;
use sptorch_core_tensor::{Device, Tensor};
use sptorch_hal_ffi::serial_backend::{
    register_tang9k_serial_dry_run_backend_for, MATMUL32X32_FLAG_CLEAR_OUTPUT, MATMUL32X32_FLAG_LAST_K_TILE,
};

#[test]
fn serial_dry_run_backend_registers_and_dispatches_matmul() {
    let device = Device::Custom(901);
    let backend = register_tang9k_serial_dry_run_backend_for(device);

    let a_data: Vec<f32> = (0..32 * 32).map(|idx| (idx % 7) as f32 - 3.0).collect();
    let b_data: Vec<f32> = (0..32 * 32).map(|idx| (idx % 5) as f32 - 2.0).collect();
    let a = Tensor::new(a_data.clone(), vec![32, 32]).to_device(device);
    let b = Tensor::new(b_data.clone(), vec![32, 32]).to_device(device);

    let out = matmul(&a, &b);

    assert_eq!(out.shape(), vec![32, 32]);
    assert_eq!(out.device(), Device::CPU);
    let out_data = out.data();
    for row in 0..32 {
        for col in 0..32 {
            let mut expected = 0.0f32;
            for kk in 0..32 {
                expected += a_data[row * 32 + kk] * b_data[kk * 32 + col];
            }
            assert_eq!(out_data[row * 32 + col], expected);
        }
    }

    let trace = backend.last_trace().expect("serial trace should be recorded");
    assert_eq!(trace.plan.command_count(), 1);
    assert_eq!(trace.frames.len(), 1);
    assert_eq!(trace.frames[0].sequence, 0);
    assert_eq!(
        trace.plan.commands[0].flags,
        MATMUL32X32_FLAG_CLEAR_OUTPUT | MATMUL32X32_FLAG_LAST_K_TILE
    );
    assert_eq!(backend.queue_depth(), 0);
}

#[test]
fn serial_dry_run_backend_tracks_multi_tile_frames() {
    let device = Device::Custom(902);
    let backend = register_tang9k_serial_dry_run_backend_for(device);

    let a_data = vec![1.0f32; 64 * 64];
    let b_data = vec![1.0f32; 64 * 64];
    let a = Tensor::new(a_data, vec![64, 64]).to_device(device);
    let b = Tensor::new(b_data, vec![64, 64]).to_device(device);

    let out = matmul(&a, &b);

    assert_eq!(out.shape(), vec![64, 64]);
    assert!(out.data().iter().all(|&value| value == 64.0));
    let trace = backend.last_trace().expect("serial trace should be recorded");
    assert_eq!(trace.plan.command_count(), 8);
    assert_eq!(trace.frames.len(), 8);
    for (idx, frame) in trace.frames.iter().enumerate() {
        assert_eq!(frame.sequence, idx as u32);
    }
}
