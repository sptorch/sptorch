use sptorch_core_ops::matmul;
use sptorch_core_tensor::{Device, Tensor};
use sptorch_hal::serial::{SerialFrame, SerialOpcode, SerialProtocolError, SerialStatusCode, SerialStatusPayload};
use sptorch_hal_ffi::serial_backend::{
    register_tang9k_serial_dry_run_backend_for, register_tang9k_serial_dry_run_backend_with_transport,
    Tang9kSerialTransport, MATMUL32X32_FLAG_CLEAR_OUTPUT, MATMUL32X32_FLAG_LAST_K_TILE,
};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct CapturingTransport {
    seen: Mutex<Vec<SerialFrame>>,
}

impl Tang9kSerialTransport for CapturingTransport {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError> {
        self.seen.lock().unwrap().push(frame.clone());
        Ok(SerialFrame::ack(frame.sequence, SerialStatusPayload::ok()))
    }
}

#[derive(Debug)]
struct MismatchedTransport;

impl Tang9kSerialTransport for MismatchedTransport {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError> {
        Ok(SerialFrame::new(
            SerialOpcode::Ack,
            frame.sequence.wrapping_add(1),
            Vec::new(),
        ))
    }
}

#[derive(Debug)]
struct BusyTransport;

impl Tang9kSerialTransport for BusyTransport {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError> {
        Ok(SerialFrame::ack(
            frame.sequence,
            SerialStatusPayload {
                code: SerialStatusCode::Busy,
                detail: 0xfeed_beef,
            },
        ))
    }
}

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
    assert_eq!(trace.reports.len(), 1);
    assert_eq!(trace.frames[0].sequence, 0);
    assert_eq!(trace.reports[0].status.code, SerialStatusCode::Ok);
    assert_eq!(trace.reports[0].queue_depth_before, 0);
    assert_eq!(trace.reports[0].queue_depth_after_enqueue, 1);
    assert_eq!(trace.reports[0].queue_depth_after_submit, 0);
    assert_eq!(trace.queue_high_watermark, 1);
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
    assert_eq!(trace.reports.len(), 8);
    for (idx, frame) in trace.frames.iter().enumerate() {
        assert_eq!(frame.sequence, idx as u32);
        assert_eq!(trace.reports[idx].status.code, SerialStatusCode::Ok);
    }
    assert_eq!(trace.queue_high_watermark, 1);
}

#[test]
fn serial_dry_run_backend_accepts_custom_transport() {
    let device = Device::Custom(903);
    let transport = Arc::new(CapturingTransport::default());
    let backend = register_tang9k_serial_dry_run_backend_with_transport(device, transport.clone());

    let a = Tensor::new(vec![1.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let b = Tensor::new(vec![2.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let out = matmul(&a, &b);

    assert!(out.data().iter().all(|&value| value == 64.0));
    assert_eq!(backend.last_trace().unwrap().frames.len(), 1);
    let seen = transport.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].opcode, SerialOpcode::Matmul32x32);
}

#[test]
#[should_panic(expected = "Tang9k serial dry-run failed to submit MatMul frames")]
fn serial_dry_run_backend_rejects_mismatched_transport_echo() {
    let device = Device::Custom(904);
    let _backend = register_tang9k_serial_dry_run_backend_with_transport(device, Arc::new(MismatchedTransport));

    let a = Tensor::new(vec![1.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let b = Tensor::new(vec![1.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let _ = matmul(&a, &b);
}

#[test]
#[should_panic(expected = "Tang9k serial dry-run failed to submit MatMul frames")]
fn serial_dry_run_backend_rejects_busy_ack() {
    let device = Device::Custom(905);
    let _backend = register_tang9k_serial_dry_run_backend_with_transport(device, Arc::new(BusyTransport));

    let a = Tensor::new(vec![1.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let b = Tensor::new(vec![1.0f32; 32 * 32], vec![32, 32]).to_device(device);
    let _ = matmul(&a, &b);
}
