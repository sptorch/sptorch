use sptorch_core_ops::{cross_entropy_loss, relu};
use sptorch_core_tensor::Tensor;
use sptorch_nn::{LayerNorm, Linear, Module};
use sptorch_optim::{zero_grad, Optimizer, SGD};

struct TinyClassifier {
    proj: Linear,
    norm: LayerNorm,
    head: Linear,
}

impl TinyClassifier {
    fn new() -> Self {
        let proj = Linear::new(2, 4, true);
        let norm = LayerNorm::new(4);
        let head = Linear::new(4, 2, true);
        set_linear_weight(&proj, &[0.35, -0.25, -0.15, 0.30, 0.20, 0.10, -0.10, -0.20]);
        set_linear_bias(&proj, &[0.01, -0.02, 0.03, -0.01]);
        set_linear_weight(&head, &[0.20, -0.10, 0.15, -0.05, -0.15, 0.20, -0.10, 0.10]);
        set_linear_bias(&head, &[0.0, 0.0]);
        TinyClassifier { proj, norm, head }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self.proj.forward(input);
        let hidden = self.norm.forward(&hidden);
        let hidden = relu(&hidden);
        self.head.forward(&hidden)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.proj.parameters());
        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params
    }
}

fn set_linear_weight(linear: &Linear, values: &[f32]) {
    let inner = linear.weight.0.read().unwrap();
    let mut storage = inner.storage.write().unwrap();
    let slice = storage.as_cpu_slice_mut();
    assert_eq!(slice.len(), values.len());
    slice.copy_from_slice(values);
}

fn set_linear_bias(linear: &Linear, values: &[f32]) {
    let bias = linear.bias.as_ref().expect("test linear should have bias");
    let inner = bias.0.read().unwrap();
    let mut storage = inner.storage.write().unwrap();
    let slice = storage.as_cpu_slice_mut();
    assert_eq!(slice.len(), values.len());
    slice.copy_from_slice(values);
}

fn batch_loss(model: &TinyClassifier, inputs: &[f32], targets: &[usize]) -> Tensor {
    let logits = model.forward(&Tensor::new(inputs.to_vec(), vec![targets.len(), 2]));
    cross_entropy_loss(&logits, targets)
}

#[test]
fn test_tiny_classifier_training_loop_decreases_loss() {
    let model = TinyClassifier::new();
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), 0.08, 0.0);
    let inputs = [
        1.0, 0.0, //
        0.8, 0.2, //
        0.0, 1.0, //
        0.2, 0.8,
    ];
    let targets = [0usize, 0, 1, 1];

    let initial_loss = batch_loss(&model, &inputs, &targets).data()[0];
    let initial_head = model.head.weight.data();

    for _ in 0..40 {
        zero_grad(&params);
        let loss = batch_loss(&model, &inputs, &targets);
        assert!(loss.is_scalar());
        loss.backward();

        for (idx, param) in params.iter().enumerate() {
            assert!(
                param.grad().is_some(),
                "trainable param[{idx}] should receive a gradient before optimizer step"
            );
        }

        opt.step();
        opt.zero_grad();
        for (idx, param) in params.iter().enumerate() {
            assert!(
                param.grad().is_none(),
                "optimizer zero_grad should clear param[{idx}] after step"
            );
        }
    }

    let final_loss = batch_loss(&model, &inputs, &targets).data()[0];
    let final_head = model.head.weight.data();
    let head_delta: f32 = initial_head
        .iter()
        .zip(final_head.iter())
        .map(|(before, after)| (before - after).abs())
        .sum();

    assert!(
        final_loss < initial_loss,
        "tiny classifier should reduce loss: initial={initial_loss:.6}, final={final_loss:.6}"
    );
    assert!(head_delta > 1e-5, "classifier head should be updated by SGD");
}

#[test]
fn test_tiny_classifier_logits_require_explicit_seed() {
    let model = TinyClassifier::new();
    let params = model.parameters();
    let logits = model.forward(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]));

    logits.backward_with_grad(&Tensor::new(vec![1.0, -1.0, -1.0, 1.0], vec![2, 2]));

    assert!(params.iter().any(|param| param.grad().is_some()));
}

#[test]
#[should_panic(expected = "backward() requires a scalar tensor")]
fn test_tiny_classifier_logits_reject_default_backward() {
    let model = TinyClassifier::new();
    let logits = model.forward(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]));

    logits.backward();
}
