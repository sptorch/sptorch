use sptorch_core_ops::cross_entropy_loss;
use sptorch_core_tensor::Tensor;
use sptorch_nn::GPT;
use sptorch_optim::{zero_grad, Optimizer, SGD};

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
