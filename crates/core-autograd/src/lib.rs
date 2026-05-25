//! 最小 autograd 扩展层。
//!
//! 这个 crate 目前保留一个轻量 `add` 示例，用来验证 `core-tensor` 暴露的
//! [`Node`] / [`Op`] / [`Tensor`] 计算图协议是否足够被外部算子 crate 复用。
//! 真正完整的算子集合位于 `sptorch-core-ops`，这里更像 autograd 协议的
//! “冒烟测试”和教学入口。

use sptorch_core_tensor::{Node, Op, Tensor};
use std::sync::Arc;

/// 加法算子的反向传播实现。
#[derive(Debug)]
pub struct AddOp;

impl Op for AddOp {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // z = a + b，所以上游梯度会原样分发给两个输入。
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// 逐元素加法，并在需要时挂接 autograd 节点。
///
/// 当前示例要求 `a` 与 `b` 形状一致；前向结果沿用 `a` 的 shape。如果任一
/// 输入开启 `requires_grad`，结果张量会记录 [`AddOp`] 与两个输入，后续
/// 调用 [`Tensor::backward`] 时梯度会沿这条边继续传播。
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let shape = a.shape();

    let res_data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
    let res = Tensor::new(res_data, shape);

    let a_req = a.requires_grad();
    let b_req = b.requires_grad();

    if a_req || b_req {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(AddOp),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    // 基础用例：`a + b` 对两个输入的局部导数都为 1。
    #[test]
    fn test_autograd_basic() {
        let x = Tensor::with_grad(vec![2.0], vec![1], true);
        let y = Tensor::with_grad(vec![3.0], vec![1], true);

        let z = add(&x, &y);
        assert_eq!(z.data(), vec![5.0]);

        z.backward();

        assert_eq!(x.grad().unwrap(), vec![1.0]);
        assert_eq!(y.grad().unwrap(), vec![1.0]);
    }

    // 链式加法验证梯度能沿计算图逐层传播到更早的叶子节点。
    #[test]
    fn test_chain_add() {
        // a + b = c, c + d = e => da=1, db=1, dc=1, dd=1
        let a = Tensor::with_grad(vec![1.0], vec![1], true);
        let b = Tensor::with_grad(vec![2.0], vec![1], true);
        let d = Tensor::with_grad(vec![4.0], vec![1], true);

        let c = add(&a, &b);
        let e = add(&c, &d);
        assert_eq!(e.data(), vec![7.0]);

        e.backward();

        assert_eq!(a.grad().unwrap(), vec![1.0]);
        assert_eq!(b.grad().unwrap(), vec![1.0]);
        assert_eq!(d.grad().unwrap(), vec![1.0]);
    }

    // 同一个叶子张量在图中被多次使用时，梯度必须累加而不是覆盖。
    #[test]
    fn test_diamond_graph() {
        // x 被使用两次：z = (x + y) + x => dx = 2.0, dy = 1.0
        let x = Tensor::with_grad(vec![3.0], vec![1], true);
        let y = Tensor::with_grad(vec![1.0], vec![1], true);

        let t = add(&x, &y); // t = x + y = 4
        let z = add(&t, &x); // z = t + x = 7
        assert_eq!(z.data(), vec![7.0]);

        z.backward();

        assert_eq!(x.grad().unwrap(), vec![2.0]);
        assert_eq!(y.grad().unwrap(), vec![1.0]);
    }

    // 非标量输出必须显式给出上游梯度；这里验证外部算子 crate 也能复用
    // `Tensor::backward_with_grad` 的 VJP 入口。
    #[test]
    fn test_backward_with_explicit_seed() {
        let x = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        let y = Tensor::with_grad(vec![3.0, 4.0], vec![2], true);
        let z = add(&x, &y);

        z.backward_with_grad(&Tensor::new(vec![0.5, 2.0], vec![2]));

        assert_eq!(x.grad().unwrap(), vec![0.5, 2.0]);
        assert_eq!(y.grad().unwrap(), vec![0.5, 2.0]);
    }
}
