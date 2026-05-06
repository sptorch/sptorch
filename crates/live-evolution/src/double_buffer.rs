use sptorch_core_tensor::Tensor;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};

/// 双缓冲参数仓。
///
/// active 缓冲服务推理请求，shadow 缓冲承接在线训练更新。训练完成并通过
/// 监控后，只翻转一个原子标志即可让 shadow 成为新的 active，避免在推理
/// 热路径复制整套参数。这个结构是版本化张量协议的本地内存基础。
pub struct DoubleBufferParams {
    buf_a: Vec<Tensor>,
    buf_b: Vec<Tensor>,
    /// true 表示 buf_a 当前给推理读，buf_b 当前给训练写。
    a_is_active: Arc<AtomicBool>,
    /// swap 与快照读取共享同一把锁，防止快照期间看到一半旧版本、一半新版本。
    swap_lock: Arc<RwLock<()>>,
}

impl DoubleBufferParams {
    /// 从一组参数创建 active/shadow 两份独立快照。
    ///
    /// 两个缓冲初始内容完全一致，但 Tensor 对象互不共享存储。调用者可以
    /// 直接把 shadow 交给训练过程修改，而不会污染正在服务推理的 active。
    pub fn new(params: &[Tensor]) -> Self {
        let buf_a: Vec<Tensor> = params.iter().map(clone_tensor).collect();
        let buf_b: Vec<Tensor> = params.iter().map(clone_tensor).collect();
        DoubleBufferParams {
            buf_a,
            buf_b,
            a_is_active: Arc::new(AtomicBool::new(true)),
            swap_lock: Arc::new(RwLock::new(())),
        }
    }

    /// 返回当前 active 参数视图。
    ///
    /// 这个方法只读取原子标志，不加锁，适合推理热路径。若调用者需要一个
    /// 跨所有参数一致的版本快照，应使用 [`Self::active_params_snapshot`]。
    pub fn active_params(&self) -> &[Tensor] {
        if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_a
        } else {
            &self.buf_b
        }
    }

    /// 返回当前 shadow 参数视图。
    ///
    /// shadow 是训练写入目标；调用者必须保证训练更新和 swap 的时序由外层
    /// fence/监控流程约束，不能在 swap 过程中继续写同一份参数。
    pub fn shadow_params(&self) -> &[Tensor] {
        if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_b
        } else {
            &self.buf_a
        }
    }

    /// 复制当前 active 参数的连续数据，得到一致快照。
    ///
    /// 读锁保证整个快照期间不会发生 swap，适合 Studio、checkpoint 或调试
    /// 工具读取版本内容。返回值是普通 `Vec<Vec<f32>>`，不会把内部锁或 Tensor
    /// 引用泄漏给异步调用者。
    pub fn active_params_snapshot(&self) -> Vec<Vec<f32>> {
        let _guard = self.swap_lock.read().unwrap();
        let params = if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_a
        } else {
            &self.buf_b
        };
        params.iter().map(|p| p.contiguous_data()).collect()
    }

    /// 原子切换 active 与 shadow 的角色。
    ///
    /// 写锁让切换与快照读取互斥；Acquire/Release 让训练写入在新 active 被
    /// 推理线程读取前具备可见性。真实硬件接入后，调用本方法前必须先确认
    /// 设备侧 fence 已经完成。
    pub fn swap(&self) {
        let _guard = self.swap_lock.write().unwrap();
        let prev = self.a_is_active.load(Ordering::Acquire);
        self.a_is_active.store(!prev, Ordering::Release);
    }

    /// 用 active 内容覆盖 shadow。
    ///
    /// 回滚或放弃一次在线训练更新后，需要把 shadow 拉回稳定版本，否则下一轮
    /// 训练会从已判定不可靠的参数继续漂移。
    pub fn sync_shadow_from_active(&self) {
        let _guard = self.swap_lock.write().unwrap();
        let (src, dst) = if self.a_is_active.load(Ordering::Acquire) {
            (&self.buf_a, &self.buf_b)
        } else {
            (&self.buf_b, &self.buf_a)
        };
        for (s, d) in src.iter().zip(dst.iter()) {
            let src_data = s.contiguous_data();
            let d_inner = d.0.read().unwrap();
            let mut d_storage = d_inner.storage.write().unwrap();
            d_storage.as_cpu_slice_mut().copy_from_slice(&src_data);
        }
    }

    /// 返回参数张量数量；active 与 shadow 数量始终相同。
    pub fn num_params(&self) -> usize {
        self.buf_a.len()
    }
}

/// 克隆 Tensor 的值、shape 和 requires_grad 标志，但不共享底层 Storage。
fn clone_tensor(t: &Tensor) -> Tensor {
    let data = t.contiguous_data();
    let shape = t.shape();
    Tensor::with_grad(data, shape, t.requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    // 初始状态必须保证 active/shadow 完全一致，否则在线训练会从不确定版本开始。
    #[test]
    fn test_double_buffer_initial_state() {
        let p = vec![Tensor::new(vec![1.0, 2.0, 3.0], vec![3])];
        let db = DoubleBufferParams::new(&p);
        assert_eq!(db.num_params(), 1);
        assert_eq!(db.active_params()[0].data(), vec![1.0, 2.0, 3.0]);
        assert_eq!(db.shadow_params()[0].data(), vec![1.0, 2.0, 3.0]);
    }

    // swap 只改变角色，不复制数据；这是实时服务场景可接受延迟的关键。
    #[test]
    fn test_double_buffer_swap() {
        let p = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let db = DoubleBufferParams::new(&p);

        // 模拟训练过程只写 shadow，active 在 swap 前必须保持旧值。
        {
            let shadow = db.shadow_params();
            let inner = shadow[0].0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            let s = storage.as_cpu_slice_mut();
            s[0] = 10.0;
            s[1] = 20.0;
        }

        assert_eq!(db.active_params()[0].data(), vec![1.0, 2.0]);

        db.swap();

        assert_eq!(db.active_params()[0].data(), vec![10.0, 20.0]);
        assert_eq!(db.shadow_params()[0].data(), vec![1.0, 2.0]);
    }

    // 回滚后 shadow 必须回到稳定 active，否则下一次提交会夹带失败更新。
    #[test]
    fn test_sync_shadow_from_active() {
        let p = vec![Tensor::new(vec![5.0, 6.0], vec![2])];
        let db = DoubleBufferParams::new(&p);

        {
            let shadow = db.shadow_params();
            let inner = shadow[0].0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            storage.as_cpu_slice_mut()[0] = 99.0;
        }
        assert_eq!(db.shadow_params()[0].data()[0], 99.0);

        db.sync_shadow_from_active();
        assert_eq!(db.shadow_params()[0].data(), vec![5.0, 6.0]);
    }
}
