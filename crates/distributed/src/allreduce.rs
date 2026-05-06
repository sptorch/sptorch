/// CPU-local allreduce utilities for single-machine multi-thread simulation.
/// These are building blocks for the distributed allreduce over gRPC.
/// 对多份同形状梯度做逐元素平均。
pub fn average_gradients(grads: &[Vec<f32>]) -> Vec<f32> {
    if grads.is_empty() {
        return Vec::new();
    }
    let n = grads[0].len();
    let world = grads.len() as f32;
    let mut result = vec![0.0f32; n];
    for g in grads {
        for (r, v) in result.iter_mut().zip(g.iter()) {
            *r += v;
        }
    }
    for r in result.iter_mut() {
        *r /= world;
    }
    result
}
/// 在本地模拟 ring-allreduce（scatter-reduce + allgather）。
pub fn ring_allreduce(local_grads: &mut [Vec<f32>]) {
    let world = local_grads.len();
    if world <= 1 {
        return;
    }
    let n = local_grads[0].len();
    let chunk = n.div_ceil(world);

    // Scatter-reduce: each rank accumulates one chunk from all peers
    for step in 0..world - 1 {
        for rank in 0..world {
            let send_chunk = (rank + world - step) % world;
            let recv_chunk = (rank + world - step - 1) % world;
            let src_start = send_chunk * chunk;
            let src_end = (src_start + chunk).min(n);
            let dst_start = recv_chunk * chunk;
            let dst_end = (dst_start + chunk).min(n);

            if src_start < n && dst_start < n {
                let next = (rank + 1) % world;
                for i in 0..(dst_end - dst_start) {
                    let val = local_grads[rank][src_start + i.min(src_end - src_start - 1)];
                    local_grads[next][dst_start + i] += val;
                }
            }
        }
    }

    // Allgather: broadcast the reduced chunks
    for step in 0..world - 1 {
        for rank in 0..world {
            let send_chunk = (rank + world - step + 1) % world;
            let start = send_chunk * chunk;
            let end = (start + chunk).min(n);
            if start < n {
                let next = (rank + 1) % world;
                let src = local_grads[rank][start..end].to_vec();
                for (dst, value) in local_grads[next][start..end].iter_mut().zip(src) {
                    *dst = value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 测试：验证 average_gradients 的行为与数值正确性。
    #[test]
    fn test_average_gradients() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![3.0, 4.0, 5.0];
        let avg = average_gradients(&[g1, g2]);
        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    // 测试：验证 average_gradients_single 的行为与数值正确性。
    #[test]
    fn test_average_gradients_single() {
        let g = vec![1.0, 2.0];
        let avg = average_gradients(&[g.clone()]);
        assert_eq!(avg, g);
    }

    // 测试：验证 average_gradients_empty 的行为与数值正确性。
    #[test]
    fn test_average_gradients_empty() {
        let avg = average_gradients(&[]);
        assert!(avg.is_empty());
    }
}
