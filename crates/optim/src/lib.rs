use sptorch_core_tensor::Tensor;

// ============ Optimizer Trait ============

pub trait Optimizer {
    // 中文注释：关键逻辑说明。
    fn step(&mut self);
    // 中文注释：关键逻辑说明。
    fn zero_grad(&self);
}

// ============ zero_grad ============
/// `zero_grad`：中文注释，说明函数用途、输入约束与输出语义。
pub fn zero_grad(params: &[Tensor]) {
    for p in params {
        let mut inner = p.0.write().unwrap();
        inner.grad = None;
    }
}

// ============ clip_grad_norm ============
/// `clip_grad_norm`：中文注释，说明函数用途、输入约束与输出语义。
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    let mut total_norm_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            total_norm_sq += g.iter().map(|x| x * x).sum::<f32>();
        }
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in params {
            let inner = p.0.read().unwrap();
            if let Some(ref grad_tensor) = inner.grad {
                let g_inner = grad_tensor.0.write().unwrap();
                let mut g_storage = g_inner.storage.write().unwrap();
                let g_slice = g_storage.as_cpu_slice_mut();
                for v in g_slice.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }

    total_norm
}

// ============ NaN/Inf guard ============

// ============ Gradient Accumulation ============
/// `scale_gradients`：中文注释，说明函数用途、输入约束与输出语义。
pub fn scale_gradients(params: &[Tensor], factor: f32) {
    for p in params {
        let inner = p.0.read().unwrap();
        if let Some(ref grad_tensor) = inner.grad {
            let g_inner = grad_tensor.0.write().unwrap();
            let mut g_storage = g_inner.storage.write().unwrap();
            let g_slice = g_storage.as_cpu_slice_mut();
            for v in g_slice.iter_mut() {
                *v *= factor;
            }
        }
    }
}

// ============ NaN/Inf guard ============

fn has_nan_inf(data: &[f32]) -> bool {
    data.iter().any(|x| x.is_nan() || x.is_infinite())
}

// ============ SGD ============

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<Vec<f32>>>,
}

impl SGD {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let n = params.len();
        SGD {
            params,
            lr,
            momentum,
            velocities: vec![None; n],
        }
    }
}

impl Optimizer for SGD {
    // 中文注释：关键逻辑说明。
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            if has_nan_inf(&grad) {
                eprintln!("SGD: skipping param[{}] due to NaN/Inf in gradient", i);
                continue;
            }

            let update = if self.momentum != 0.0 {
                let vel = self.velocities[i].get_or_insert_with(|| vec![0.0; grad.len()]);
                for (v, g) in vel.iter_mut().zip(grad.iter()) {
                    *v = self.momentum * *v + *g;
                }
                vel.clone()
            } else {
                grad
            };

            let inner = param.0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            let w_slice = storage.as_cpu_slice_mut();
            for (w, u) in w_slice.iter_mut().zip(update.iter()) {
                *w -= self.lr * u;
            }
        }
    }

    // 中文注释：关键逻辑说明。
    fn zero_grad(&self) {
        zero_grad(&self.params);
    }
}

// ============ AdamW ============

pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>, // first moment
    v: Vec<Vec<f32>>, // second moment
    t: u64,           // step count
}

impl AdamW {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(params: Vec<Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        AdamW {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }
    /// `default`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn default(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    // 中文注释：关键逻辑说明。
    fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            if has_nan_inf(&grad) {
                eprintln!("AdamW: skipping param[{}] due to NaN/Inf in gradient", i);
                continue;
            }

            let inner = param.0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            let w_slice = storage.as_cpu_slice_mut();

            // Decoupled weight decay
            if self.weight_decay != 0.0 {
                for w in w_slice.iter_mut() {
                    *w *= 1.0 - self.lr * self.weight_decay;
                }
            }

            // Update moments
            for (j, g) in grad.iter().enumerate() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected update
            for (j, w) in w_slice.iter_mut().enumerate() {
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;
                *w -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    // 中文注释：关键逻辑说明。
    fn zero_grad(&self) {
        zero_grad(&self.params);
    }
}

// ============ Learning Rate Schedulers ============

pub trait LrScheduler {
    // 中文注释：关键逻辑说明。
    fn get_lr(&self, step: u64) -> f32;
}

/// Warmup + Cosine Decay
pub struct CosineScheduler {
    pub base_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_lr: f32,
}

impl CosineScheduler {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(base_lr: f32, warmup_steps: u64, total_steps: u64) -> Self {
        CosineScheduler {
            base_lr,
            warmup_steps,
            total_steps,
            min_lr: base_lr * 0.1,
        }
    }
}

impl LrScheduler for CosineScheduler {
    // 中文注释：关键逻辑说明。
    fn get_lr(&self, step: u64) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32 / self.warmup_steps.max(1) as f32)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps).max(1) as f32;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

impl AdamW {
    /// `set_lr`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl SGD {
    /// `set_lr`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_core_tensor::Tensor;

    // 中文注释：关键逻辑说明。
    fn make_param_with_grad(data: Vec<f32>, grad_data: Vec<f32>) -> Tensor {
        let shape = vec![data.len()];
        let t = Tensor::with_grad(data, shape.clone(), true);
        let grad = Tensor::new(grad_data, shape);
        t.accum_grad(&grad);
        t
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_sgd_basic() {
        let p = make_param_with_grad(vec![1.0, 2.0], vec![0.1, 0.2]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0);
        opt.step();
        let d = p.data();
        assert!((d[0] - 0.99).abs() < 1e-6); // 1.0 - 0.1*0.1
        assert!((d[1] - 1.98).abs() < 1e-6); // 2.0 - 0.1*0.2
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_sgd_momentum() {
        let p = make_param_with_grad(vec![1.0], vec![1.0]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.9);
        opt.step();
        // v = 0.9*0 + 1.0 = 1.0, w = 1.0 - 0.1*1.0 = 0.9
        let d = p.data();
        assert!((d[0] - 0.9).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_adamw_basic() {
        let p = make_param_with_grad(vec![1.0, 2.0], vec![0.1, 0.2]);
        let mut opt = AdamW::new(vec![p.clone()], 0.001, 0.9, 0.999, 1e-8, 0.0);
        opt.step();
        let d = p.data();
        // Params should have decreased
        assert!(d[0] < 1.0);
        assert!(d[1] < 2.0);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_adamw_weight_decay() {
        let p = make_param_with_grad(vec![1.0], vec![0.0]);
        let mut opt = AdamW::new(vec![p.clone()], 0.1, 0.9, 0.999, 1e-8, 0.1);
        opt.step();
        let d = p.data();
        // With zero grad, only weight decay applies: w *= (1 - lr*wd) = 1.0 * 0.99 = 0.99
        assert!((d[0] - 0.99).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_zero_grad() {
        let p = make_param_with_grad(vec![1.0], vec![0.5]);
        assert!(p.grad().is_some());
        zero_grad(&[p.clone()]);
        assert!(p.grad().is_none());
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_clip_grad_norm() {
        let p = make_param_with_grad(vec![1.0], vec![3.0, 4.0]);
        let norm = clip_grad_norm(&[p.clone()], 1.0);
        assert!((norm - 5.0).abs() < 1e-5); // sqrt(9+16) = 5
        let g = p.grad().unwrap();
        let clipped_norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-4);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_clip_grad_norm_no_clip() {
        let p = make_param_with_grad(vec![1.0], vec![0.3, 0.4]);
        let norm = clip_grad_norm(&[p.clone()], 10.0);
        assert!((norm - 0.5).abs() < 1e-5);
        // Should not be clipped
        let g = p.grad().unwrap();
        assert!((g[0] - 0.3).abs() < 1e-6);
        assert!((g[1] - 0.4).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_nan_guard() {
        let p = make_param_with_grad(vec![1.0], vec![f32::NAN]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0);
        opt.step();
        // Should skip update, param unchanged
        assert_eq!(p.data(), vec![1.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cosine_scheduler_warmup() {
        let sched = CosineScheduler::new(0.001, 100, 1000);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-8);
        assert!((sched.get_lr(50) - 0.0005).abs() < 1e-6);
        assert!((sched.get_lr(100) - 0.001).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cosine_scheduler_decay() {
        let sched = CosineScheduler::new(0.001, 0, 1000);
        let lr_start = sched.get_lr(0);
        let lr_mid = sched.get_lr(500);
        let lr_end = sched.get_lr(1000);
        assert!((lr_start - 0.001).abs() < 1e-6);
        assert!(lr_mid < lr_start && lr_mid > lr_end);
        assert!((lr_end - 0.0001).abs() < 1e-6); // min_lr = 0.1 * base
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gradient_accumulation_basic() {
        let p = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        let params = vec![p.clone()];
        let mut opt = SGD::new(params.clone(), 0.1, 0.0);

        // Simulate 3 micro-steps of gradient accumulation
        // Each micro-step adds grad [1.0, 1.0]
        for _ in 0..3 {
            // Manually set grad (simulating backward)
            p.accum_grad(&Tensor::new(vec![1.0, 1.0], vec![2]));
        }
        // Accumulated grad = [3.0, 3.0], scale by 1/3
        scale_gradients(&params, 1.0 / 3.0);
        opt.step();

        let w = p.data();
        // w = [1.0 - 0.1*1.0, 2.0 - 0.1*1.0] = [0.9, 1.9]
        assert!((w[0] - 0.9).abs() < 1e-6);
        assert!((w[1] - 1.9).abs() < 1e-6);
    }
}
