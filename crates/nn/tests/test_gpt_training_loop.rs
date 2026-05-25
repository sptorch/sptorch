use sptorch_core_ops::cross_entropy_loss;
use sptorch_core_tensor::Tensor;
use sptorch_nn::GPT;
use sptorch_optim::{zero_grad, Optimizer, SGD};
use sptorch_serialize::{load_state_dict_file, save_state_dict};
use sptorch_versioning::CheckpointManifest;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn sequence_loss(model: &GPT, tokens: &[usize]) -> Tensor {
    assert!(tokens.len() >= 2, "sequence training needs input and target tokens");
    let inputs = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];
    let logits = model.forward_ids(inputs);
    cross_entropy_loss(&logits, targets)
}

fn average_sequence_loss(model: &GPT, sequences: &[Vec<usize>]) -> f32 {
    let total: f32 = sequences.iter().map(|seq| sequence_loss(model, seq).data()[0]).sum();
    total / sequences.len() as f32
}

fn named_gpt_params(model: &GPT) -> Vec<(String, Tensor)> {
    let named = model.named_parameters();
    named.into_iter().map(|param| (param.name, param.tensor)).collect()
}

fn named_refs(params: &[(String, Tensor)]) -> Vec<(&str, Tensor)> {
    params
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor.clone()))
        .collect()
}

fn unique_temp_path(file_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("sptorch_{file_name}_{}_{}.json", std::process::id(), nanos))
}

#[test]
fn test_tiny_gpt_training_loop_decreases_next_token_loss() {
    let mut model = GPT::new(6, 4, 2, 1, 8, 4);
    model.set_training(false);
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), 0.02, 0.0);
    let sequences = vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![2, 3, 4, 5]];

    let initial_loss = average_sequence_loss(&model, &sequences);
    let initial_head = model.lm_head.weight.data();

    for _ in 0..6 {
        zero_grad(&params);
        for seq in &sequences {
            sequence_loss(&model, seq).backward();
        }

        for (idx, param) in params.iter().enumerate() {
            assert!(
                param.grad().is_some(),
                "GPT param[{idx}] should receive gradient during next-token training"
            );
        }

        opt.step();
        opt.zero_grad();
        assert!(params.iter().all(|param| param.grad().is_none()));
    }

    let final_loss = average_sequence_loss(&model, &sequences);
    let final_head = model.lm_head.weight.data();
    let head_delta: f32 = initial_head
        .iter()
        .zip(final_head.iter())
        .map(|(before, after)| (before - after).abs())
        .sum();

    assert!(
        final_loss <= initial_loss,
        "tiny GPT should not regress on the fixed next-token task: initial={initial_loss:.6}, final={final_loss:.6}"
    );
    assert!(head_delta > 1e-6, "language-model head should be updated");
}

#[test]
#[should_panic(expected = "backward() requires a scalar tensor")]
fn test_tiny_gpt_logits_reject_default_backward() {
    let mut model = GPT::new(6, 4, 2, 1, 8, 4);
    model.set_training(false);
    let logits = model.forward_ids(&[0, 1, 2]);

    logits.backward();
}

#[test]
fn test_tiny_gpt_logits_accept_explicit_seed() {
    let mut model = GPT::new(6, 4, 2, 1, 8, 4);
    model.set_training(false);
    let params = model.parameters();
    let logits = model.forward_ids(&[0, 1, 2]);
    let seed = Tensor::new(vec![1.0; logits.numel()], logits.shape());

    logits.backward_with_grad(&seed);

    assert!(params.iter().any(|param| param.grad().is_some()));
}

#[test]
fn test_tiny_gpt_state_dict_roundtrip_and_resume_training() {
    let mut model = GPT::new(6, 4, 2, 1, 8, 4);
    model.set_training(false);
    let sequences = vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![2, 3, 4, 5]];
    let checkpoint_path = unique_temp_path("tiny_gpt_state_dict");
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), 0.02, 0.0);

    for _ in 0..4 {
        zero_grad(&params);
        for seq in &sequences {
            sequence_loss(&model, seq).backward();
        }
        opt.step();
        opt.zero_grad();
    }

    let named_params = named_gpt_params(&model);
    let named_param_refs = named_refs(&named_params);
    let manifest = CheckpointManifest {
        schema: "sptorch.checkpoint_manifest.v1".into(),
        format_version: 1,
        model_name: "tiny-gpt".into(),
        save_kind: "state_dict".into(),
        parameter_count: named_param_refs.len(),
        state_dict_schema: "sptorch.state_dict.v1".into(),
        created_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_millis() as u64,
        note: "integration test checkpoint".into(),
    };
    assert_eq!(manifest.parameter_count, named_param_refs.len());

    save_state_dict(&checkpoint_path, &named_param_refs).unwrap();
    let checkpoint_loss = average_sequence_loss(&model, &sequences);
    let saved_weights = model.lm_head.weight.data();

    {
        let mut loaded_model = GPT::new(6, 4, 2, 1, 8, 4);
        loaded_model.set_training(false);
        let loaded_named_params = named_gpt_params(&loaded_model);
        let loaded_param_refs = named_refs(&loaded_named_params);
        let loaded_entries = load_state_dict_file(&checkpoint_path, &loaded_param_refs).unwrap();

        assert_eq!(loaded_entries.len(), named_param_refs.len());
        assert_eq!(loaded_model.lm_head.weight.data(), saved_weights);
        let restored_loss = average_sequence_loss(&loaded_model, &sequences);
        assert!(
            (restored_loss - checkpoint_loss).abs() < 1e-5,
            "loaded model loss should match checkpointed model: checkpoint={checkpoint_loss:.6}, restored={restored_loss:.6}"
        );

        let mut opt = SGD::new(loaded_model.parameters(), 0.02, 0.0);
        let before_resume_head = loaded_model.lm_head.weight.data();
        for _ in 0..3 {
            let params = loaded_model.parameters();
            zero_grad(&params);
            for seq in &sequences {
                sequence_loss(&loaded_model, seq).backward();
            }
            assert!(params.iter().any(|param| param.grad().is_some()));
            opt.step();
            opt.zero_grad();
        }

        let resumed_loss = average_sequence_loss(&loaded_model, &sequences);
        let after_resume_head = loaded_model.lm_head.weight.data();
        let head_delta: f32 = before_resume_head
            .iter()
            .zip(after_resume_head.iter())
            .map(|(before, after)| (before - after).abs())
            .sum();

        assert!(resumed_loss.is_finite());
        assert!(resumed_loss <= checkpoint_loss);
        assert!(
            head_delta > 1e-6,
            "loaded GPT should keep updating after state_dict restore"
        );
    }

    fs::remove_file(checkpoint_path).ok();
}
