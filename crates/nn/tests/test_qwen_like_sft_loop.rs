use sptorch_core_ops::cross_entropy_loss_ignore_index;
use sptorch_core_tensor::Tensor;
use sptorch_data::{BpeTokenizer, Tokenizer};
use sptorch_nn::QwenLikeGPT;
use sptorch_optim::{zero_grad, Optimizer, SGD};

const IGNORE_INDEX: usize = usize::MAX;

fn masked_sft_sequence(
    tokenizer: &BpeTokenizer,
    prompt: &str,
    answer: &str,
    seq_len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut prompt_ids = tokenizer.encode(prompt);
    let mut answer_ids = tokenizer.encode(answer);
    assert!(
        !prompt_ids.is_empty() && !answer_ids.is_empty(),
        "test tokenizer should represent prompt and answer"
    );

    let max_tokens = seq_len + 1;
    while prompt_ids.len() + answer_ids.len() > max_tokens {
        if prompt_ids.len() > 1 {
            prompt_ids.remove(0);
        } else {
            answer_ids.pop();
        }
    }

    let mut tokens = prompt_ids.clone();
    tokens.append(&mut answer_ids);
    assert!(tokens.len() >= 2);

    let input = tokens[..tokens.len() - 1].to_vec();
    let mut targets = tokens[1..].to_vec();
    let prompt_target_len = prompt_ids.len().saturating_sub(1);
    for target in targets.iter_mut().take(prompt_target_len) {
        *target = IGNORE_INDEX;
    }

    (input, targets)
}

fn sequence_loss(model: &QwenLikeGPT, input: &[usize], targets: &[usize]) -> Tensor {
    let logits = model.forward_ids(input);
    cross_entropy_loss_ignore_index(&logits, targets, IGNORE_INDEX)
}

fn average_loss(model: &QwenLikeGPT, samples: &[(Vec<usize>, Vec<usize>)]) -> f32 {
    samples
        .iter()
        .map(|(input, targets)| sequence_loss(model, input, targets).data()[0])
        .sum::<f32>()
        / samples.len() as f32
}

#[test]
fn test_qwen_like_sft_loop_masks_prompt_and_updates_sql_region() {
    let corpus = [
        "Question: How many employees are there?\nSQL: SELECT COUNT(*) FROM employees;\n",
        "Question: List department names.\nSQL: SELECT name FROM departments;\n",
        "Question: How many orders are paid?\nSQL: SELECT COUNT(*) FROM orders WHERE status = paid;\n",
    ]
    .join("");
    let tokenizer = BpeTokenizer::train(&corpus, 96);
    let seq_len = 18;
    let samples = vec![
        masked_sft_sequence(
            &tokenizer,
            "Question: How many employees are there?\nSQL: ",
            "SELECT COUNT(*) FROM employees;\n",
            seq_len,
        ),
        masked_sft_sequence(
            &tokenizer,
            "Question: List department names.\nSQL: ",
            "SELECT name FROM departments;\n",
            seq_len,
        ),
    ];

    assert!(
        samples
            .iter()
            .all(|(_, targets)| targets.iter().any(|&target| target == IGNORE_INDEX)
                && targets.iter().any(|&target| target != IGNORE_INDEX)),
        "SFT samples should contain both ignored prompt targets and supervised SQL targets"
    );

    let mut model = QwenLikeGPT::new(tokenizer.vocab_size(), 8, 2, 1, 16, seq_len);
    model.set_training(false);
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), 0.01, 0.0);
    let initial_loss = average_loss(&model, &samples);
    let initial_head = model.lm_head.weight.data();

    for _ in 0..5 {
        zero_grad(&params);
        for (input, targets) in &samples {
            sequence_loss(&model, input, targets).backward();
        }
        assert!(
            params.iter().any(|param| param.grad().is_some()),
            "Qwen-like SFT loop should propagate gradients through RoPE attention and SwiGLU"
        );
        opt.step();
        opt.zero_grad();
    }

    let final_loss = average_loss(&model, &samples);
    let final_head = model.lm_head.weight.data();
    let head_delta: f32 = initial_head
        .iter()
        .zip(final_head.iter())
        .map(|(before, after)| (before - after).abs())
        .sum();

    assert!(
        final_loss.is_finite() && final_loss <= initial_loss,
        "SFT loop should not regress on masked SQL targets: initial={initial_loss:.6}, final={final_loss:.6}"
    );
    assert!(head_delta > 1e-6, "language-model head should be updated");
}
