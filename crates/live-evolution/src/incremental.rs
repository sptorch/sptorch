use sptorch_core_tensor::Tensor;
use sptorch_optim::Optimizer;

/// 增量训练调度器。
///
/// 在线系统不会像离线训练那样等完整 epoch 准备好再更新，而是持续接收样本，
/// 凑够一个 micro-batch 就触发一次训练步骤。这个结构只负责样本缓冲和步数
/// 记录，不直接计算 loss；模型前向、反向和 optimizer step 由调用者编排。
pub struct IncrementalTrainer<O: Optimizer> {
    optimizer: O,
    _params: Vec<Tensor>,
    micro_batch_size: usize,
    buffer: Vec<(Vec<usize>, Vec<usize>)>, // (input_ids, target_ids)
    total_steps: u64,
}

impl<O: Optimizer> IncrementalTrainer<O> {
    /// 创建增量训练器。
    ///
    /// `micro_batch_size` 是触发训练的样本阈值。当前不主动拒绝 0，但实际使用
    /// 时应传入大于 0 的值；否则每次 `push_sample` 都会被视为可训练状态。
    pub fn new(optimizer: O, params: Vec<Tensor>, micro_batch_size: usize) -> Self {
        IncrementalTrainer {
            optimizer,
            _params: params,
            micro_batch_size,
            buffer: Vec::new(),
            total_steps: 0,
        }
    }

    /// 放入一条样本，并返回是否已经凑够 micro-batch。
    ///
    /// 返回 `true` 只表示“可以训练”，不会自动清空缓冲；调用者应在完成
    /// 前向/反向准备后调用 [`Self::drain_batch`]，这样可以在训练前记录指标或做
    /// 额外过滤。
    pub fn push_sample(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>) -> bool {
        self.buffer.push((input_ids, target_ids));
        self.buffer.len() >= self.micro_batch_size
    }

    /// 取出当前缓冲中的所有样本并清空缓冲。
    ///
    /// 这里一次性 drain 全部样本，而不是只取 `micro_batch_size` 条，是为了
    /// 让高吞吐输入在一次训练步骤里消化积压，减少在线系统的延迟尾巴。
    pub fn drain_batch(&mut self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let batch: Vec<_> = self.buffer.drain(..).collect();
        batch
    }

    /// 标记一次 optimizer step 已完成。
    ///
    /// 步数只在训练真正提交后增加，不在 `push_sample` 时增加；这让版本提交、
    /// 梯度累积和监控窗口都能以真实更新次数为准。
    pub fn step_completed(&mut self) {
        self.total_steps += 1;
    }

    /// 返回已完成的训练步数。
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// 返回当前尚未训练的样本数量。
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// 返回 optimizer 的可变引用。
    ///
    /// 调用者通过它执行 `zero_grad`、`step` 等操作；调度器不隐藏 optimizer，
    /// 是为了保持框架层组合灵活，而不是把训练循环写死在这里。
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_optim::SGD;

    // 验证样本缓冲只在达到 micro-batch 阈值后触发训练信号。
    #[test]
    fn test_incremental_trainer_buffering() {
        let params = vec![Tensor::with_grad(vec![1.0, 2.0], vec![2], true)];
        let opt = SGD::new(params.clone(), 0.01, 0.0);
        let mut trainer = IncrementalTrainer::new(opt, params, 3);

        assert!(!trainer.push_sample(vec![0, 1], vec![1, 2]));
        assert!(!trainer.push_sample(vec![2, 3], vec![3, 4]));
        assert_eq!(trainer.buffer_len(), 2);

        assert!(trainer.push_sample(vec![4, 5], vec![5, 6]));
        assert_eq!(trainer.buffer_len(), 3);

        let batch = trainer.drain_batch();
        assert_eq!(batch.len(), 3);
        assert_eq!(trainer.buffer_len(), 0);
    }

    // 步数统计代表真实提交的训练更新，不能随样本进入缓冲而提前增加。
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
