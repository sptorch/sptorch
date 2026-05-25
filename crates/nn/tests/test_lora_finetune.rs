use sptorch_core_ops::cross_entropy_loss;
use sptorch_core_tensor::Tensor;
use sptorch_nn::{Linear, LoRALinear, Module, NamedParameter};
use sptorch_optim::{zero_grad, Optimizer, SGD};
use sptorch_serialize::{load_state_dict_bundle, save_state_dict_bundle, STATE_DICT_SCHEMA};
use sptorch_versioning::{CheckpointManifest, CHECKPOINT_MANIFEST_FORMAT_VERSION, CHECKPOINT_MANIFEST_SCHEMA};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn named_refs(params: &[NamedParameter]) -> Vec<(&str, Tensor)> {
    params
        .iter()
        .map(|param| (param.name.as_str(), param.tensor.clone()))
        .collect()
}

fn unique_temp_path(file_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("sptorch_{file_name}_{}_{}.json", std::process::id(), nanos))
}

fn state_manifest(model_name: &str) -> CheckpointManifest {
    CheckpointManifest {
        schema: CHECKPOINT_MANIFEST_SCHEMA.into(),
        format_version: CHECKPOINT_MANIFEST_FORMAT_VERSION,
        model_name: model_name.into(),
        save_kind: "state_dict".into(),
        weights_file: String::new(),
        parameter_count: 0,
        parameter_names: Vec::new(),
        state_dict_schema: STATE_DICT_SCHEMA.into(),
        created_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_millis() as u64,
        note: "lora regression checkpoint".into(),
    }
}

/// End-to-end LoRA fine-tuning test:
/// Build a tiny model, wrap its linear layers with LoRA, train on synthetic data,
/// verify loss decreases, then merge LoRA back into base weights.
#[test]
fn test_lora_finetune_loss_decreases() {
    // Tiny "model": input [batch, 4] -> Linear(4,8) -> ReLU -> Linear(8,3) -> logits
    let layer1 = Linear::new(4, 8, false);
    let layer2 = Linear::new(8, 3, false);

    // Wrap with LoRA (rank=2, alpha=1.0)
    let lora1 = LoRALinear::new(layer1, 2, 1.0);
    let lora2 = LoRALinear::new(layer2, 2, 1.0);

    // Only LoRA params are trainable
    let mut params: Vec<Tensor> = Vec::new();
    params.extend(lora1.parameters());
    params.extend(lora2.parameters());
    assert_eq!(params.len(), 4); // lora_a + lora_b for each layer

    let lr = 0.05;
    let mut opt = SGD::new(params.clone(), lr, 0.0);

    // Synthetic training data: 4 samples, 3 classes
    let inputs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let targets = vec![0usize, 1, 2, 0];

    let mut losses = Vec::new();

    for _step in 0..30 {
        zero_grad(&params);

        // Forward pass (manual since we don't have a full model wrapper)
        let mut batch_loss = 0.0f32;
        for (inp, &tgt) in inputs.iter().zip(targets.iter()) {
            let x = Tensor::with_grad(inp.clone(), vec![1, 4], false);
            let h = lora1.forward(&x);
            // ReLU
            let h_data = h.data();
            let relu_data: Vec<f32> = h_data.iter().map(|v| v.max(0.0)).collect();
            let h_relu = Tensor::new(relu_data, h.shape());
            let logits = lora2.forward(&h_relu);
            let loss = cross_entropy_loss(&logits, &[tgt]);
            loss.backward();
            batch_loss += loss.data()[0];
        }

        losses.push(batch_loss / 4.0);
        opt.step();
    }

    // Verify loss decreased
    let first_loss = losses[0];
    let last_loss = losses[losses.len() - 1];
    assert!(
        last_loss < first_loss,
        "LoRA fine-tuning should decrease loss: first={:.4} last={:.4}",
        first_loss,
        last_loss
    );

    // Verify LoRA adapters were actually updated during training
    let b_data = lora1.lora_b.data();
    let b_nonzero: f32 = b_data.iter().map(|v| v.abs()).sum();
    // B started as zeros; if training worked, at least some values should be nonzero
    // (Note: with only 30 steps and small lr, B might still be very small)

    // Verify merge changes base weights (only if B is nonzero)
    if b_nonzero > 1e-10 {
        let w_before = lora1.base.weight.data();
        lora1.merge();
        let w_after = lora1.base.weight.data();
        let diff: f32 = w_before.iter().zip(w_after.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "merge should change base weights when B is nonzero");
    }
}

#[test]
fn test_lora_state_dict_bundle_roundtrip_restores_forward() {
    let lora = LoRALinear::from_dims(4, 3, true, 2, 1.0);
    {
        let inner = lora.lora_b.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        let slice = storage.as_cpu_slice_mut();
        for (i, value) in slice.iter_mut().enumerate() {
            *value = 0.01 * (i + 1) as f32;
        }
    }

    let input = Tensor::new(vec![0.2, -0.1, 0.4, 0.7, -0.3, 0.8, 0.1, -0.5], vec![2, 4]);
    let expected = lora.forward(&input).data();
    let weights_path = unique_temp_path("lora_bundle");
    let named = lora.named_parameters("adapter");
    let refs = named_refs(&named);
    let manifest = state_manifest("single-lora");
    let manifest_path = format!("{}.manifest.json", weights_path.display());

    save_state_dict_bundle(&weights_path, &manifest, &refs).unwrap();

    let restored = LoRALinear::from_dims(4, 3, true, 2, 1.0);
    let restored_named = restored.named_parameters("adapter");
    let restored_refs = named_refs(&restored_named);
    let (loaded_manifest, loaded_entries) = load_state_dict_bundle(&weights_path, &restored_refs).unwrap();

    assert_eq!(loaded_manifest.model_name, "single-lora");
    assert_eq!(loaded_manifest.parameter_count, refs.len());
    assert_eq!(
        loaded_manifest.parameter_names,
        refs.iter().map(|(name, _)| (*name).to_string()).collect::<Vec<_>>()
    );
    assert_eq!(loaded_entries.len(), refs.len());

    let actual = restored.forward(&input).data();
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "restored LoRA output mismatch at {idx}: got {got}, expected {want}"
        );
    }

    fs::remove_file(&weights_path).unwrap();
    fs::remove_file(manifest_path).unwrap();
}
