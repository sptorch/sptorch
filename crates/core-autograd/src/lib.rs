use sptorch_core_tensor::{Node, Op, Tensor};
use std::sync::Arc;

#[derive(Debug)]
pub struct AddOp;

impl Op for AddOp {
    // 加法的梯度会原样分发给两个输入。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // z = a + b => dL/da = dL/dz, dL/db = dL/dz
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}
/// `add`：中文注释，说明函数用途、输入约束与输出语义。
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

    // 基础用例：a+b 的反向梯度都应为 1。
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

    // 链式加法：验证梯度沿计算图逐层传播。
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

    // 菱形计算图：同一变量被多次使用时梯度应累加。
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
}
