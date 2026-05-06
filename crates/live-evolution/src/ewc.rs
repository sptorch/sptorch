use sptorch_core_tensor::Tensor;

/// Elastic Weight Consolidation（EWC）正则器。
///
/// 在线学习最容易破坏旧任务能力。EWC 用上一阶段参数快照 `theta_star` 和
/// Fisher 对角近似 `F_i` 衡量“哪些参数不能乱动”，并把如下惩罚项加到新任务
/// loss 中：
///
/// `Loss_total = Loss_new + (lambda / 2) * sum_i F_i * (theta_i - theta_star_i)^2`
///
/// 当前实现用平方梯度近似 Fisher，对小模型和增量训练足够直观；后续如果
/// 引入更精确的 Fisher 估计，也应保持这里的外部语义不变。
pub struct EWC {
    /// 上一个稳定版本的参数快照。
    param_snapshot: Vec<Vec<f32>>,
    /// Fisher 对角近似，值越大表示对应参数对旧任务越重要。
    fisher_diag: Vec<Vec<f32>>,
    /// 正则强度；越大越抗遗忘，但也越限制新任务适应能力。
    pub lambda: f32,
}

impl EWC {
    /// 从当前参数和对应梯度构建 EWC 正则器。
    ///
    /// `params` 与 `grads` 必须一一对应，且每个梯度向量长度应等于参数
    /// 展平后的元素数。当前实现信任调用者，这是为了避免训练热路径重复做
    /// shape 检查。
    pub fn new(params: &[Tensor], grads: &[Vec<f32>], lambda: f32) -> Self {
        let param_snapshot: Vec<Vec<f32>> = params.iter().map(|p| p.contiguous_data()).collect();

        let fisher_diag: Vec<Vec<f32>> = grads.iter().map(|g| g.iter().map(|v| v * v).collect()).collect();

        EWC {
            param_snapshot,
            fisher_diag,
            lambda,
        }
    }

    /// 计算当前参数相对快照的 EWC 惩罚值。
    ///
    /// 参数没有漂移时返回 0；重要参数（Fisher 大）漂移同样距离会产生更高
    /// penalty，从而在训练目标里“拉回”旧能力。
    pub fn penalty(&self, current_params: &[Tensor]) -> f32 {
        let mut total = 0.0f32;
        for (i, param) in current_params.iter().enumerate() {
            let current = param.contiguous_data();
            let snapshot = &self.param_snapshot[i];
            let fisher = &self.fisher_diag[i];
            for j in 0..current.len() {
                let diff = current[j] - snapshot[j];
                total += fisher[j] * diff * diff;
            }
        }
        self.lambda * 0.5 * total
    }

    /// 计算 EWC 惩罚项对每个参数的梯度。
    ///
    /// 对 `lambda / 2 * F * (theta - theta_star)^2` 求导得到
    /// `lambda * F * (theta - theta_star)`。调用者可将这些梯度加到正常
    /// backprop 梯度上。
    pub fn penalty_grads(&self, current_params: &[Tensor]) -> Vec<Vec<f32>> {
        current_params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let current = param.contiguous_data();
                let snapshot = &self.param_snapshot[i];
                let fisher = &self.fisher_diag[i];
                current
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| self.lambda * fisher[j] * (c - snapshot[j]))
                    .collect()
            })
            .collect()
    }

    /// 将 EWC 梯度原地叠加到参数已有梯度上。
    ///
    /// 如果某个参数还没有 grad tensor，会跳过它。这样 EWC 可以安全用于只
    /// 训练部分层的场景，不会强行给冻结层创建梯度。
    pub fn apply_penalty(&self, params: &[Tensor]) {
        let penalty_grads = self.penalty_grads(params);
        for (param, pg) in params.iter().zip(penalty_grads.iter()) {
            let inner = param.0.read().unwrap();
            if let Some(ref grad_tensor) = inner.grad {
                let g_inner = grad_tensor.0.write().unwrap();
                let mut g_storage = g_inner.storage.write().unwrap();
                let g_slice = g_storage.as_cpu_slice_mut();
                for (g, &p) in g_slice.iter_mut().zip(pg.iter()) {
                    *g += p;
                }
            }
        }
    }

    /// 返回 EWC 管理的参数张量数量。
    pub fn num_params(&self) -> usize {
        self.param_snapshot.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 快照点 penalty 必须为 0，这是 EWC 不干扰已稳定版本的基本不变量。
    #[test]
    fn test_ewc_zero_penalty_at_snapshot() {
        let params = vec![
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
            Tensor::new(vec![4.0, 5.0], vec![2]),
        ];
        let grads = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5]];
        let ewc = EWC::new(&params, &grads, 1.0);

        let penalty = ewc.penalty(&params);
        assert!(penalty.abs() < 1e-10);
    }

    // 漂移越大 penalty 越大，这直接决定在线更新是否会被监控/优化器压回。
    #[test]
    fn test_ewc_penalty_increases_with_drift() {
        let params = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let grads = vec![vec![1.0, 1.0]]; // uniform Fisher
        let ewc = EWC::new(&params, &grads, 2.0);

        let drifted = vec![Tensor::new(vec![1.5, 2.5], vec![2])];
        let penalty = ewc.penalty(&drifted);
        // lambda/2 * sum(F * diff^2) = 2.0/2 * (1.0*0.25 + 1.0*0.25) = 0.5
        assert!((penalty - 0.5).abs() < 1e-6);
    }

    // 梯度公式是 EWC 能否真正参与优化器更新的核心语义。
    #[test]
    fn test_ewc_penalty_grads() {
        let params = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let grads = vec![vec![1.0, 2.0]]; // non-uniform Fisher
        let ewc = EWC::new(&params, &grads, 1.0);

        let drifted = vec![Tensor::new(vec![2.0, 3.0], vec![2])];
        let pg = ewc.penalty_grads(&drifted);
        // lambda * F * (theta - theta_star)
        // [1.0 * 1.0 * (2.0-1.0), 1.0 * 4.0 * (3.0-2.0)] = [1.0, 4.0]
        assert!((pg[0][0] - 1.0).abs() < 1e-6);
        assert!((pg[0][1] - 4.0).abs() < 1e-6);
    }

    // Fisher 大的参数应该更“抗改”，这是防灾难性遗忘的直觉验收。
    #[test]
    fn test_ewc_high_fisher_resists_change() {
        let params = vec![Tensor::new(vec![1.0, 1.0], vec![2])];
        let grads = vec![vec![10.0, 0.1]];
        let ewc = EWC::new(&params, &grads, 1.0);

        let drifted = vec![Tensor::new(vec![2.0, 2.0], vec![2])];
        let pg = ewc.penalty_grads(&drifted);
        assert!(pg[0][0] > pg[0][1] * 100.0);
    }
}
