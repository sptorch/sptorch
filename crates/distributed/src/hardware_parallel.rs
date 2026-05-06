//! 面向硬件拓扑的分布式验证计划。
//!
//! `distributed` crate 负责把 `sptorch-hal` 给出的设备图转成可执行的并行
//! 方案。这里的计划刻意保持 dry-run 友好：即使真实 Tank9k 串口/PCIe
//! 后端还没接入，也能先验证多板拓扑、Ring AllReduce 顺序和 MatMul 分片
//! 是否满足工程预期。

use sptorch_hal::topology::{AllReduceEstimate, HardwareTopology, MatmulPartitionPlan, TopologyValidation};
use sptorch_hal::DeviceId;

/// 分布式层当前认识的硬件并行原语。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelCollective {
    RingAllReduce,
    Matmul2DShard,
}

/// 一次硬件并行验证的完整规划结果。
///
/// `validation` 保存拓扑诊断，`world` 保存参与设备顺序，`collectives` 描述
/// 本计划会用到的通信/计算原语。具体执行器可以先检查 [`Self::ready`]，再决定
/// 是否把计划下发给真实硬件或 mock 后端。
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareParallelPlan {
    pub topology_name: String,
    pub world: Vec<DeviceId>,
    pub validation: TopologyValidation,
    pub collectives: Vec<ParallelCollective>,
    pub allreduce: Option<AllReduceEstimate>,
    pub matmul: Option<MatmulPartitionPlan>,
}

impl HardwareParallelPlan {
    /// 判断计划是否具备最基本的执行条件。
    ///
    /// 这里不要求所有估算字段都存在，因为某些验证只关心拓扑连通性；但
    /// 如果拓扑不可达或没有节点，就必须阻止执行。
    pub fn ready(&self) -> bool {
        self.validation.connected && self.validation.total_nodes > 0
    }

    /// 生成适合日志和 CI 失败信息的短摘要。
    pub fn summary(&self) -> String {
        format!(
            "topology={} world={} connected={} collectives={}",
            self.topology_name,
            self.world.len(),
            self.validation.connected,
            self.collectives.len()
        )
    }
}

/// 基于拓扑和梯度 payload 大小规划 Ring AllReduce。
///
/// 该函数只生成计划，不执行通信。真实运行时仍需要 coordinator/worker 或
/// HAL 后端根据 `world` 顺序提交梯度分片、等待 fence 并合并结果。
pub fn plan_ring_allreduce(topology: &HardwareTopology, payload_bytes: usize) -> HardwareParallelPlan {
    let validation = topology.validate_connectivity();
    let world = topology.ring_plan();
    let allreduce = topology.allreduce_cost_estimate(payload_bytes);
    HardwareParallelPlan {
        topology_name: topology.name.clone(),
        world,
        validation,
        collectives: vec![ParallelCollective::RingAllReduce],
        allreduce,
        matmul: None,
    }
}

/// 为 Tank9k MatMul 验证生成“计算分片 + 梯度归约”组合计划。
///
/// 这对应 roadmap 里的多板并行验证主线：先把 32x32 MatMul 拆到多块板，
/// 再用 Ring AllReduce 估算梯度同步路径。它不是最终训练调度器，但能把
/// 硬件连通性、分片边界和通信成本放到一个可测试对象里。
pub fn plan_tank9k_matmul_validation(
    topology: &HardwareTopology,
    m: usize,
    k: usize,
    n: usize,
    gradient_payload_bytes: usize,
) -> HardwareParallelPlan {
    let validation = topology.validate_connectivity();
    let world = topology.ring_plan();
    let allreduce = topology.allreduce_cost_estimate(gradient_payload_bytes);
    let matmul = Some(topology.matmul_partition_plan(m, k, n));
    HardwareParallelPlan {
        topology_name: topology.name.clone(),
        world,
        validation,
        collectives: vec![ParallelCollective::Matmul2DShard, ParallelCollective::RingAllReduce],
        allreduce,
        matmul,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_hal::topology::{HardwareLink, HardwareNode, LinkRole, TransportKind};

    // 构造一个最小可用的 Tank9k 环形拓扑，后续多板测试都以它作为稳定夹具。
    fn tank9k_ring(n: usize) -> HardwareTopology {
        let mut topo = HardwareTopology::new("tank9k-validation-ring");
        for i in 0..n {
            topo.add_node(HardwareNode::new(
                DeviceId::tank9k(i),
                format!("tank9k-{i}"),
                "tank9k",
                4096,
            ));
        }
        for i in 0..n {
            topo.add_link(HardwareLink::new(
                DeviceId::tank9k(i),
                DeviceId::tank9k((i + 1) % n),
                TransportKind::Serial,
                LinkRole::Data,
                32,
                150,
            ));
        }
        topo
    }

    // 验证多板 MatMul 计划同时包含计算分片和梯度同步，避免只测拓扑却漏掉训练闭环。
    #[test]
    fn builds_multi_board_tank9k_validation_plan() {
        let topo = tank9k_ring(4);
        let plan = plan_tank9k_matmul_validation(&topo, 32, 32, 32, 32 * 32 * 4);

        assert!(plan.ready(), "{}", plan.summary());
        assert_eq!(plan.world.len(), 4);
        assert_eq!(
            plan.collectives,
            vec![ParallelCollective::Matmul2DShard, ParallelCollective::RingAllReduce]
        );
        assert_eq!(plan.matmul.as_ref().unwrap().shards.len(), 4);
        assert_eq!(plan.allreduce.as_ref().unwrap().ring_hops, 4);
    }

    // 缺链路场景必须生成 not-ready 计划，这样 CLI/CI 能早于真实硬件执行阶段失败。
    #[test]
    fn broken_topology_produces_not_ready_plan() {
        let mut topo = HardwareTopology::new("broken-tank9k");
        topo.add_node(HardwareNode::new(DeviceId::tank9k(0), "a", "tank9k", 4096));
        topo.add_node(HardwareNode::new(DeviceId::tank9k(1), "b", "tank9k", 4096));

        let plan = plan_ring_allreduce(&topo, 4096);
        assert!(!plan.ready());
        assert!(plan.allreduce.is_none());
        assert!(!plan.validation.diagnostics.is_empty());
    }
}
