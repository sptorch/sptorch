//! 训练优化器与梯度处理工具。
//!
//! 这里的实现服务于框架核心训练闭环：参数来自 `core-tensor::Tensor`，优化器只读
//! 已累积梯度并原地更新 CPU storage。复杂分布式同步、混合精度缩放和版本化参数
//! 切换由更高层 crate 编排，本模块保持数学语义清晰、易测试。

use sptorch_core_tensor::Tensor;

// ============ Optimizer Trait ============

/// 优化器的最小行为契约。
///
/// `step` 消费当前梯度并更新参数；`zero_grad` 清空梯度缓存。调用方负责在合适的
/// 训练边界调用二者，例如梯度累积时只在多个 micro-step 后执行一次 `step`。
pub trait Optimizer {
    /// 根据当前参数梯度执行一次更新。
    fn step(&mut self);
    /// 清空优化器持有参数的梯度。
    fn zero_grad(&self);
}

// ============ zero_grad ============
/// 清空一组参数上的梯度。
///
/// 该函数不会改变 `requires_grad`，也不会断开计算图；它只把叶子参数上已累积的
/// `grad` 置空，通常在 optimizer step 后或新一轮梯度累积前调用。
pub fn zero_grad(params: &[Tensor]) {
    for p in params {
        let mut inner = p.0.write().unwrap();
        inner.grad = None;
    }
}

// ============ clip_grad_norm ============
/// 按全局 L2 范数裁剪梯度，并返回裁剪前的范数。
///
/// 所有参数的梯度会被视作一个展开后的大向量。如果总范数超过 `max_norm`，每个
/// 梯度元素乘以同一个缩放系数，从而保持梯度方向不变。
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

// ============ Gradient Accumulation ============
/// 对已有梯度统一乘以 `factor`。
///
/// 典型用途是梯度累积：多个 micro-step 累加梯度后，用 `1 / accum_steps` 把梯度
/// 还原为平均梯度。该函数只处理已存在梯度的参数，缺失梯度的参数会被跳过。
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

/// 随机梯度下降，可选 momentum。
///
/// 当前实现采用经典 velocity 累积：`v = momentum * v + grad`，参数更新为
/// `w -= lr * v`。若某个参数梯度包含 NaN/Inf，该参数会被跳过，避免一次异常
/// backward 污染全部权重。
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<Vec<f32>>>,
}

impl SGD {
    /// 创建 SGD 优化器。
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

    fn zero_grad(&self) {
        zero_grad(&self.params);
    }
}

// ============ AdamW ============

/// AdamW 优化器，使用解耦 weight decay。
///
/// 一阶、二阶矩按参数展平存储，长度在构造时由 `Tensor::numel` 决定。因此调用方
/// 不应在优化器生命周期内替换参数 shape。
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
    /// 创建 AdamW 优化器。
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

    /// 使用常见 Transformer 训练默认超参创建 AdamW。
    pub fn default(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
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

            // AdamW 的 weight decay 与梯度矩估计解耦，避免把 L2 正则混进一阶/二阶矩。
            if self.weight_decay != 0.0 {
                for w in w_slice.iter_mut() {
                    *w *= 1.0 - self.lr * self.weight_decay;
                }
            }

            for (j, g) in grad.iter().enumerate() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
            }

            for (j, w) in w_slice.iter_mut().enumerate() {
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;
                *w -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    fn zero_grad(&self) {
        zero_grad(&self.params);
    }
}

// ============ Learning Rate Schedulers ============

/// 学习率调度器接口。
pub trait LrScheduler {
    /// 返回指定 step 应使用的学习率。
    fn get_lr(&self, step: u64) -> f32;
}

/// 线性 warmup 后接余弦衰减的学习率计划。
pub struct CosineScheduler {
    pub base_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_lr: f32,
}

impl CosineScheduler {
    /// 创建 scheduler，默认最小学习率为 `base_lr * 0.1`。
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
    /// 动态调整 AdamW 学习率，供 scheduler 或手动调参使用。
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl SGD {
    /// 动态调整 SGD 学习率，供 scheduler 或手动调参使用。
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_core_tensor::Tensor;

    // 构造带梯度参数，避免每个优化器测试重复手写锁内部结构。
    fn make_param_with_grad(data: Vec<f32>, grad_data: Vec<f32>) -> Tensor {
        let shape = vec![data.len()];
        let t = Tensor::with_grad(data, shape.clone(), true);
        let grad = Tensor::new(grad_data, shape);
        t.accum_grad(&grad);
        t
    }

    // SGD 基础更新必须匹配 `w -= lr * grad`。
    #[test]
    fn test_sgd_basic() {
        let p = make_param_with_grad(vec![1.0, 2.0], vec![0.1, 0.2]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0);
        opt.step();
        let d = p.data();
        assert!((d[0] - 0.99).abs() < 1e-6); // 1.0 - 0.1*0.1
        assert!((d[1] - 1.98).abs() < 1e-6); // 2.0 - 0.1*0.2
    }

    // momentum 的第一步退化为当前梯度，后续才体现历史累积。
    #[test]
    fn test_sgd_momentum() {
        let p = make_param_with_grad(vec![1.0], vec![1.0]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.9);
        opt.step();
        // v = 0.9*0 + 1.0 = 1.0, w = 1.0 - 0.1*1.0 = 0.9
        let d = p.data();
        assert!((d[0] - 0.9).abs() < 1e-6);
    }

    // AdamW 一步更新至少应沿负梯度方向移动。
    #[test]
    fn test_adamw_basic() {
        let p = make_param_with_grad(vec![1.0, 2.0], vec![0.1, 0.2]);
        let mut opt = AdamW::new(vec![p.clone()], 0.001, 0.9, 0.999, 1e-8, 0.0);
        opt.step();
        let d = p.data();
        assert!(d[0] < 1.0);
        assert!(d[1] < 2.0);
    }

    // 解耦 weight decay 在零梯度时仍会独立收缩参数。
    #[test]
    fn test_adamw_weight_decay() {
        let p = make_param_with_grad(vec![1.0], vec![0.0]);
        let mut opt = AdamW::new(vec![p.clone()], 0.1, 0.9, 0.999, 1e-8, 0.1);
        opt.step();
        let d = p.data();
        assert!((d[0] - 0.99).abs() < 1e-6);
    }

    // zero_grad 只清空梯度，不影响参数本身。
    #[test]
    fn test_zero_grad() {
        let p = make_param_with_grad(vec![1.0], vec![0.5]);
        assert!(p.grad().is_some());
        zero_grad(&[p.clone()]);
        assert!(p.grad().is_none());
    }

    // 裁剪返回裁剪前范数，同时把实际梯度压到 max_norm 附近。
    #[test]
    fn test_clip_grad_norm() {
        let p = make_param_with_grad(vec![1.0], vec![3.0, 4.0]);
        let norm = clip_grad_norm(&[p.clone()], 1.0);
        assert!((norm - 5.0).abs() < 1e-5); // sqrt(9+16) = 5
        let g = p.grad().unwrap();
        let clipped_norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-4);
    }

    // 小于阈值的梯度不能被意外缩放。
    #[test]
    fn test_clip_grad_norm_no_clip() {
        let p = make_param_with_grad(vec![1.0], vec![0.3, 0.4]);
        let norm = clip_grad_norm(&[p.clone()], 10.0);
        assert!((norm - 0.5).abs() < 1e-5);
        let g = p.grad().unwrap();
        assert!((g[0] - 0.3).abs() < 1e-6);
        assert!((g[1] - 0.4).abs() < 1e-6);
    }

    // NaN/Inf guard 是训练稳定性底线：异常梯度不能污染参数。
    #[test]
    fn test_nan_guard() {
        let p = make_param_with_grad(vec![1.0], vec![f32::NAN]);
        let mut opt = SGD::new(vec![p.clone()], 0.1, 0.0);
        opt.step();
        assert_eq!(p.data(), vec![1.0]);
    }

    // warmup 阶段应从 0 线性爬升到 base_lr。
    #[test]
    fn test_cosine_scheduler_warmup() {
        let sched = CosineScheduler::new(0.001, 100, 1000);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-8);
        assert!((sched.get_lr(50) - 0.0005).abs() < 1e-6);
        assert!((sched.get_lr(100) - 0.001).abs() < 1e-6);
    }

    // 衰减阶段应单调进入较低学习率区间，并在 total_steps 后固定到 min_lr。
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

    // 梯度累积缩放验证多 micro-step 后的平均梯度更新。
    #[test]
    fn test_gradient_accumulation_basic() {
        let p = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        let params = vec![p.clone()];
        let mut opt = SGD::new(params.clone(), 0.1, 0.0);

        for _ in 0..3 {
            p.accum_grad(&Tensor::new(vec![1.0, 1.0], vec![2]));
        }
        scale_gradients(&params, 1.0 / 3.0);
        opt.step();

        let w = p.data();
        assert!((w[0] - 0.9).abs() < 1e-6);
        assert!((w[1] - 1.9).abs() < 1e-6);
    }
}
