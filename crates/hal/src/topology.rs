//! 多板卡与异构设备拓扑描述。
//!
//! 这个模块服务于 Tank9k 和未来异构集群的“先规划、再点亮”流程：在真实
//! 串口、PCIe、DMA 后端稳定之前，框架仍然可以用拓扑模型验证连通性、
//! 生成 Ring AllReduce 顺序，并估算 MatMul 分片是否合理。
//!
//! 拓扑是有向图。若一条链路在物理上双向可用，调用者需要显式加入两个
//! 方向的 [`HardwareLink`]，这样可以表达半双工、控制链路和数据链路不对称
//! 的硬件现实。

use super::DeviceId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// 设备之间的传输介质。
///
/// 它描述的是链路的工程类型，而不是 Rust 侧的具体实现。比如 `Serial`
/// 可以先代表 mock 串口协议，后续再接真实 UART/DMA；规划层只关心带宽、
/// 延迟和可达性。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportKind {
    Serial,
    Pcie,
    Ethernet,
    SharedMemory,
    Mock,
}

impl fmt::Display for TransportKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TransportKind::Serial => "serial",
            TransportKind::Pcie => "pcie",
            TransportKind::Ethernet => "ethernet",
            TransportKind::SharedMemory => "shared-memory",
            TransportKind::Mock => "mock",
        })
    }
}

/// 链路在系统中的用途。
///
/// 同一对板卡可能同时存在控制、数据和遥测链路。把 role 放进拓扑模型，
/// 是为了后续调度器能区分“发指令”“搬梯度”“读队列深度”这些不同路径，
/// 而不是把所有连接都粗暴看成一条数据线。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkRole {
    Control,
    Data,
    Telemetry,
    Synchronization,
}

impl fmt::Display for LinkRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LinkRole::Control => "control",
            LinkRole::Data => "data",
            LinkRole::Telemetry => "telemetry",
            LinkRole::Synchronization => "sync",
        })
    }
}

/// 拓扑中的一个逻辑硬件节点。
///
/// `memory_mb` 和 `queue_depth_hint` 都是规划提示，不代表实时硬件状态。
/// 实时状态应由 HAL/FFI 遥测层上报；拓扑层只保存足够稳定的信息，保证
/// dry-run、CI 和文档中的示例可以复现。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareNode {
    pub id: DeviceId,
    pub label: String,
    pub board_class: String,
    pub memory_mb: u32,
    pub queue_depth_hint: u32,
    pub online: bool,
}

impl HardwareNode {
    /// 创建一个默认在线的硬件节点。
    ///
    /// `label` 面向人类排障，`board_class` 面向调度和能力识别。比如多块
    /// Tank9k 可以分别叫 `board-a`、`board-b`，但共享同一个 `tank9k`
    /// board class。
    pub fn new(id: DeviceId, label: impl Into<String>, board_class: impl Into<String>, memory_mb: u32) -> Self {
        Self {
            id,
            label: label.into(),
            board_class: board_class.into(),
            memory_mb,
            queue_depth_hint: 0,
            online: true,
        }
    }
}

/// 拓扑中的一条有向硬件链路。
///
/// `bandwidth_mb_s` 和 `latency_us` 是粗粒度规划参数，用于判断方案是否
/// 值得进入真实硬件验证；它们不是 benchmark 结果，也不应该被展示成
/// 性能承诺。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareLink {
    pub from: DeviceId,
    pub to: DeviceId,
    pub transport: TransportKind,
    pub role: LinkRole,
    pub hops: u32,
    pub bandwidth_mb_s: u32,
    pub latency_us: u32,
    pub full_duplex: bool,
}

impl HardwareLink {
    /// 创建一条单跳、默认全双工的有向链路。
    ///
    /// 即使 `full_duplex` 为 true，这个对象仍只表示 `from -> to`。如果算法
    /// 需要反向发送梯度或 fence，需要额外添加 `to -> from` 链路，避免规划
    /// 层隐式猜测硬件能力。
    pub fn new(
        from: DeviceId,
        to: DeviceId,
        transport: TransportKind,
        role: LinkRole,
        bandwidth_mb_s: u32,
        latency_us: u32,
    ) -> Self {
        Self {
            from,
            to,
            transport,
            role,
            hops: 1,
            bandwidth_mb_s,
            latency_us,
            full_duplex: true,
        }
    }
}

/// 一组硬件节点和有向链路组成的拓扑。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareTopology {
    pub name: String,
    pub nodes: Vec<HardwareNode>,
    pub links: Vec<HardwareLink>,
}

impl HardwareTopology {
    /// 创建空拓扑。
    ///
    /// 空拓扑是合法的中间状态，但 [`Self::validate_connectivity`] 会把它标记为
    /// 不可用，防止上层误把“还没发现设备”当成“单机可运行”。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }

    /// 添加一个节点；当前不做去重，调用者应保持 `DeviceId` 唯一。
    pub fn add_node(&mut self, node: HardwareNode) {
        self.nodes.push(node);
    }

    /// 添加一条链路。
    ///
    /// 链路端点是否存在会在 [`Self::validate_connectivity`] 中统一诊断，便于批量
    /// 构造拓扑时一次性返回所有问题。
    pub fn add_link(&mut self, link: HardwareLink) {
        self.links.push(link);
    }

    /// 按逻辑设备 ID 查找节点。
    pub fn node(&self, id: &DeviceId) -> Option<&HardwareNode> {
        self.nodes.iter().find(|node| &node.id == id)
    }

    /// 返回从指定设备出发的所有有向链路。
    pub fn neighbors(&self, id: &DeviceId) -> Vec<&HardwareLink> {
        self.links.iter().filter(|link| &link.from == id).collect()
    }

    /// 返回当前标记为在线的节点数量。
    ///
    /// 这是规划输入中的状态，不会主动探测硬件；真实在线探测应由 HAL FFI
    /// 或遥测服务更新节点字段后再重新验证。
    pub fn online_node_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.online).count()
    }

    /// 验证拓扑是否从第一个节点出发可达所有节点。
    ///
    /// 这里使用有向 BFS，因此会抓出“环少了一条反向边”“某块板只接了控制
    /// 线没有数据线”等早期硬件联调常见问题。离线节点会进入 diagnostics，
    /// 但当前连通性仍按图结构计算，方便区分“拓扑设计错误”和“运行时掉线”。
    pub fn validate_connectivity(&self) -> TopologyValidation {
        let mut diagnostics = Vec::new();
        let mut graph: HashMap<&DeviceId, Vec<&DeviceId>> = HashMap::new();
        let mut all_nodes: HashSet<&DeviceId> = HashSet::new();

        for node in &self.nodes {
            all_nodes.insert(&node.id);
            if !node.online {
                diagnostics.push(format!("node {} is offline", node.id));
            }
        }

        for link in &self.links {
            if self.node(&link.from).is_none() {
                diagnostics.push(format!("link from {} points to unknown node", link.from));
                continue;
            }
            if self.node(&link.to).is_none() {
                diagnostics.push(format!("link to {} points to unknown node", link.to));
                continue;
            }
            graph.entry(&link.from).or_default().push(&link.to);
        }

        if self.nodes.is_empty() {
            diagnostics.push("topology has no nodes".to_string());
            return TopologyValidation {
                connected: false,
                connected_nodes: 0,
                total_nodes: 0,
                diagnostics,
            };
        }

        let start = &self.nodes[0].id;
        let mut visited: HashSet<&DeviceId> = HashSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(next) = graph.get(current) {
                for peer in next {
                    queue.push_back(peer);
                }
            }
        }

        let connected = visited.len() == all_nodes.len();
        if !connected {
            diagnostics.push(format!(
                "topology is not fully connected: {} / {} nodes reachable",
                visited.len(),
                all_nodes.len()
            ));
        }

        TopologyValidation {
            connected,
            connected_nodes: visited.len(),
            total_nodes: all_nodes.len(),
            diagnostics,
        }
    }

    /// 生成稳定的 Ring 顺序。
    ///
    /// 排序只依赖 `backend` 和 `ordinal`，不依赖插入顺序。这样 CI、文档样例
    /// 和硬件 dry-run 会得到同一条 ring，方便比较 AllReduce 估算结果。
    pub fn ring_plan(&self) -> Vec<DeviceId> {
        let mut nodes = self.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
        nodes.sort_by(|a, b| a.backend.cmp(&b.backend).then(a.ordinal.cmp(&b.ordinal)));
        nodes
    }

    /// 给定梯度 payload 字节数，估算 Ring AllReduce 的粗略耗时。
    ///
    /// 返回 `None` 表示拓扑不具备执行条件。估算采用最窄带宽和累计延迟，
    /// 只用于方案筛选，不替代真实 benchmark；真实硬件点亮后仍需用遥测和
    /// benchmark 校准。
    pub fn allreduce_cost_estimate(&self, payload_bytes: usize) -> Option<AllReduceEstimate> {
        if self.nodes.len() < 2 {
            return None;
        }
        let validation = self.validate_connectivity();
        if !validation.connected {
            return None;
        }

        let ring = self.ring_plan();
        let mut bandwidth_mb_s = u32::MAX;
        let mut latency_us = 0u32;
        for window in ring.windows(2) {
            let from = &window[0];
            let to = &window[1];
            if let Some(link) = self.links.iter().find(|link| &link.from == from && &link.to == to) {
                bandwidth_mb_s = bandwidth_mb_s.min(link.bandwidth_mb_s);
                latency_us += link.latency_us;
            }
        }
        if let (Some(first), Some(last)) = (ring.first(), ring.last()) {
            if let Some(link) = self.links.iter().find(|link| &link.from == last && &link.to == first) {
                bandwidth_mb_s = bandwidth_mb_s.min(link.bandwidth_mb_s);
                latency_us += link.latency_us;
            }
        }

        let hops = ring.len() as u32;
        let per_round_bytes = payload_bytes as f64 / hops as f64;
        let bandwidth_seconds = (per_round_bytes / (bandwidth_mb_s.max(1) as f64 * 1024.0 * 1024.0)) * hops as f64;
        let latency_seconds = (latency_us as f64 * hops as f64) / 1_000_000.0;
        Some(AllReduceEstimate {
            nodes: ring,
            payload_bytes,
            estimated_seconds: bandwidth_seconds + latency_seconds,
            ring_hops: hops,
            min_bandwidth_mb_s: bandwidth_mb_s,
            latency_us,
        })
    }

    /// 生成 MatMul 验证分片计划。
    ///
    /// 当前策略按节点数量粗略切分行列，目标是为 Tank9k 多板 32x32 MatMul
    /// 验证提供可解释计划，而不是追求最优 tiling。后续真实 kernel 上线后，
    /// 可以在保持返回结构不变的前提下替换为更细的调度器。
    pub fn matmul_partition_plan(&self, m: usize, k: usize, n: usize) -> MatmulPartitionPlan {
        let nodes = self.ring_plan();
        let tile_rows = std::cmp::max(1, m / nodes.len().max(1));
        let tile_cols = std::cmp::max(1, n / nodes.len().max(1));
        let shards = nodes
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let row_start = index * tile_rows;
                let row_end = if index + 1 == nodes.len() {
                    m
                } else {
                    (row_start + tile_rows).min(m)
                };
                let col_start = index * tile_cols;
                let col_end = if index + 1 == nodes.len() {
                    n
                } else {
                    (col_start + tile_cols).min(n)
                };
                MatmulShard {
                    device: device.clone(),
                    row_range: row_start..row_end,
                    col_range: col_start..col_end,
                    k,
                }
            })
            .collect();

        MatmulPartitionPlan { m, k, n, shards }
    }
}

/// 拓扑连通性诊断结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyValidation {
    pub connected: bool,
    pub connected_nodes: usize,
    pub total_nodes: usize,
    pub diagnostics: Vec<String>,
}

/// Ring AllReduce 粗略成本估算。
#[derive(Debug, Clone, PartialEq)]
pub struct AllReduceEstimate {
    pub nodes: Vec<DeviceId>,
    pub payload_bytes: usize,
    pub estimated_seconds: f64,
    pub ring_hops: u32,
    pub min_bandwidth_mb_s: u32,
    pub latency_us: u32,
}

/// 单个 MatMul 分片在某个设备上的负责范围。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulShard {
    pub device: DeviceId,
    pub row_range: std::ops::Range<usize>,
    pub col_range: std::ops::Range<usize>,
    pub k: usize,
}

/// 一次 MatMul dry-run 的完整分片计划。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulPartitionPlan {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub shards: Vec<MatmulShard>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // 这个测试把“三块 Tank9k 串口成环”作为最小多板验收样板：拓扑可达、
    // ring 顺序稳定、AllReduce 有成本估算、MatMul 能分配到每块板。
    #[test]
    fn tank9k_ring_topology_validates_and_plans() {
        let a = DeviceId::tank9k(0);
        let b = DeviceId::tank9k(1);
        let c = DeviceId::tank9k(2);

        let mut topo = HardwareTopology::new("tank9k-ring");
        topo.add_node(HardwareNode::new(a.clone(), "board-a", "tank9k", 4096));
        topo.add_node(HardwareNode::new(b.clone(), "board-b", "tank9k", 4096));
        topo.add_node(HardwareNode::new(c.clone(), "board-c", "tank9k", 4096));
        topo.add_link(HardwareLink::new(
            a.clone(),
            b.clone(),
            TransportKind::Serial,
            LinkRole::Data,
            40,
            120,
        ));
        topo.add_link(HardwareLink::new(
            b.clone(),
            c.clone(),
            TransportKind::Serial,
            LinkRole::Data,
            40,
            120,
        ));
        topo.add_link(HardwareLink::new(
            c.clone(),
            a.clone(),
            TransportKind::Serial,
            LinkRole::Data,
            40,
            120,
        ));

        let validation = topo.validate_connectivity();
        assert!(validation.connected);
        assert_eq!(validation.connected_nodes, 3);
        assert_eq!(topo.online_node_count(), 3);

        let ring = topo.ring_plan();
        assert_eq!(ring.len(), 3);
        assert_eq!(ring[0].backend, "tank9k");

        let estimate = topo.allreduce_cost_estimate(12 * 1024 * 1024).expect("estimate");
        assert_eq!(estimate.ring_hops, 3);
        assert!(estimate.estimated_seconds > 0.0);

        let partition = topo.matmul_partition_plan(32, 32, 32);
        assert_eq!(partition.shards.len(), 3);
        assert_eq!(partition.shards[0].device, a);
        assert_eq!(partition.shards[1].device, b);
        assert_eq!(partition.shards[2].device, c);
    }

    // 缺链路时必须拒绝 AllReduce 估算，否则上层可能把不可执行计划推给真实硬件。
    #[test]
    fn detects_missing_links() {
        let mut topo = HardwareTopology::new("broken");
        topo.add_node(HardwareNode::new(DeviceId::tank9k(0), "board-a", "tank9k", 4096));
        topo.add_node(HardwareNode::new(DeviceId::tank9k(1), "board-b", "tank9k", 4096));
        let validation = topo.validate_connectivity();
        assert!(!validation.connected);
        assert!(!validation.diagnostics.is_empty());
        assert!(topo.allreduce_cost_estimate(1024).is_none());
    }
}
