use sptorch_core_tensor::Tensor;

/// Elastic Weight Consolidation (EWC) — prevents catastrophic forgetting
/// by penalizing changes to parameters that were important for previous tasks.
///
/// Loss_total = Loss_new + (lambda/2) * sum_i F_i * (theta_i - theta_star_i)^2
///
/// where F_i is the Fisher information (approximated by squared gradients)
/// and theta_star is the parameter snapshot from the previous task.
pub struct EWC {
    /// Snapshot of parameters after previous task training
    param_snapshot: Vec<Vec<f32>>,
    /// Fisher information diagonal (importance of each parameter)
    fisher_diag: Vec<Vec<f32>>,
    /// Regularization strength
    pub lambda: f32,
}

impl EWC {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(params: &[Tensor], grads: &[Vec<f32>], lambda: f32) -> Self {
        let param_snapshot: Vec<Vec<f32>> = params.iter().map(|p| p.contiguous_data()).collect();

        let fisher_diag: Vec<Vec<f32>> = grads.iter().map(|g| g.iter().map(|v| v * v).collect()).collect();

        EWC {
            param_snapshot,
            fisher_diag,
            lambda,
        }
    }
    /// `penalty`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `penalty_grads`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `apply_penalty`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `num_params`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn num_params(&self) -> usize {
        self.param_snapshot.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_ewc_zero_penalty_at_snapshot() {
        let params = vec![
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
            Tensor::new(vec![4.0, 5.0], vec![2]),
        ];
        let grads = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5]];
        let ewc = EWC::new(&params, &grads, 1.0);

        // At the snapshot point, penalty should be zero
        let penalty = ewc.penalty(&params);
        assert!(penalty.abs() < 1e-10);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_ewc_penalty_increases_with_drift() {
        let params = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let grads = vec![vec![1.0, 1.0]]; // uniform Fisher
        let ewc = EWC::new(&params, &grads, 2.0);

        // Drift params by 0.5
        let drifted = vec![Tensor::new(vec![1.5, 2.5], vec![2])];
        let penalty = ewc.penalty(&drifted);
        // lambda/2 * sum(F * diff^2) = 2.0/2 * (1.0*0.25 + 1.0*0.25) = 0.5
        assert!((penalty - 0.5).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_ewc_high_fisher_resists_change() {
        let params = vec![Tensor::new(vec![1.0, 1.0], vec![2])];
        // param[0] has high Fisher (important), param[1] has low Fisher
        let grads = vec![vec![10.0, 0.1]];
        let ewc = EWC::new(&params, &grads, 1.0);

        let drifted = vec![Tensor::new(vec![2.0, 2.0], vec![2])];
        let pg = ewc.penalty_grads(&drifted);
        // param[0] penalty grad = 1.0 * 100.0 * 1.0 = 100.0 (strong resistance)
        // param[1] penalty grad = 1.0 * 0.01 * 1.0 = 0.01 (weak resistance)
        assert!(pg[0][0] > pg[0][1] * 100.0);
    }
}
