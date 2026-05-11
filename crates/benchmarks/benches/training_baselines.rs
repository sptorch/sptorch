use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use sptorch_core_ops::{cross_entropy_loss, matmul, relu, sum};
use sptorch_core_tensor::{no_grad, Tensor};
use sptorch_nn::{Linear, Module};

fn seq_data(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i % 97) as f32 - 48.0) * scale).collect()
}

fn bench_matmul_cpu(c: &mut Criterion) {
    let a = Tensor::new(seq_data(64 * 64, 0.01), vec![64, 64]);
    let b = Tensor::new(seq_data(64 * 64, 0.02), vec![64, 64]);

    c.bench_function("matmul_cpu_64x64", |bench| {
        bench.iter(|| {
            let out = no_grad(|| matmul(black_box(&a), black_box(&b)));
            black_box(out.data());
        });
    });
}

fn bench_forward_pass(c: &mut Criterion) {
    let layer1 = Linear::new(64, 128, true);
    let layer2 = Linear::new(128, 64, true);
    let x = Tensor::new(seq_data(16 * 64, 0.01), vec![16, 64]);

    c.bench_function("forward_pass_mlp_16x64", |bench| {
        bench.iter(|| {
            let out = no_grad(|| {
                let h = relu(&layer1.forward(black_box(&x)));
                layer2.forward(&h)
            });
            black_box(out.data());
        });
    });
}

fn bench_backward_pass(c: &mut Criterion) {
    let targets = vec![1usize, 7, 3, 11, 5, 9, 2, 13];

    c.bench_function("backward_pass_linear_ce", |bench| {
        bench.iter(|| {
            let layer = Linear::new(32, 16, true);
            let x = Tensor::with_grad(seq_data(8 * 32, 0.01), vec![8, 32], true);
            let logits = layer.forward(black_box(&x));
            let loss = cross_entropy_loss(&logits, black_box(&targets));
            loss.backward();
            black_box(sum(&logits).data());
        });
    });
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_matmul_cpu, bench_forward_pass, bench_backward_pass
}
criterion_main!(benches);
