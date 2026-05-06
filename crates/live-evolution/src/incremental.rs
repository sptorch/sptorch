use sptorch_core_tensor::Tensor;
use sptorch_optim::Optimizer;

/// Incremental training scheduler: triggers micro-batch updates
/// when new data arrives, instead of full epoch-based training.
pub struct IncrementalTrainer<O: Optimizer> {
    optimizer: O,
    _params: Vec<Tensor>,
    micro_batch_size: usize,
    buffer: Vec<(Vec<usize>, Vec<usize>)>, // (input_ids, target_ids)
    total_steps: u64,
}

impl<O: Optimizer> IncrementalTrainer<O> {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(optimizer: O, params: Vec<Tensor>, micro_batch_size: usize) -> Self {
        IncrementalTrainer {
            optimizer,
            _params: params,
            micro_batch_size,
            buffer: Vec::new(),
            total_steps: 0,
        }
    }
    /// `push_sample`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn push_sample(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>) -> bool {
        self.buffer.push((input_ids, target_ids));
        self.buffer.len() >= self.micro_batch_size
    }
    /// `drain_batch`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn drain_batch(&mut self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let batch: Vec<_> = self.buffer.drain(..).collect();
        batch
    }
    /// `step_completed`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn step_completed(&mut self) {
        self.total_steps += 1;
    }
    /// `total_steps`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
    /// `buffer_len`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
    /// `optimizer_mut`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_optim::SGD;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_incremental_trainer_buffering() {
        let params = vec![Tensor::with_grad(vec![1.0, 2.0], vec![2], true)];
        let opt = SGD::new(params.clone(), 0.01, 0.0);
        let mut trainer = IncrementalTrainer::new(opt, params, 3);

        assert!(!trainer.push_sample(vec![0, 1], vec![1, 2]));
        assert!(!trainer.push_sample(vec![2, 3], vec![3, 4]));
        assert_eq!(trainer.buffer_len(), 2);

        // Third sample triggers batch
        assert!(trainer.push_sample(vec![4, 5], vec![5, 6]));
        assert_eq!(trainer.buffer_len(), 3);

        let batch = trainer.drain_batch();
        assert_eq!(batch.len(), 3);
        assert_eq!(trainer.buffer_len(), 0);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_incremental_trainer_step_count() {
        let params = vec![Tensor::with_grad(vec![1.0], vec![1], true)];
        let opt = SGD::new(params.clone(), 0.01, 0.0);
        let mut trainer = IncrementalTrainer::new(opt, params, 1);

        assert_eq!(trainer.total_steps(), 0);
        trainer.step_completed();
        assert_eq!(trainer.total_steps(), 1);
        trainer.step_completed();
        assert_eq!(trainer.total_steps(), 2);
    }
}
