//! Hardware topology descriptions for multi-board and heterogeneous deployments.
//!
//! This module is intentionally framework-level: it models devices, links, and
//! executable validation plans without binding to any single board or transport.

use super::DeviceId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportKind {
    Serial,
    Pcie,
    Ethernet,
    SharedMemory,
    Mock,
}

impl fmt::Display for TransportKind {
    // 中文注释：关键逻辑说明。
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkRole {
    Control,
    Data,
    Telemetry,
    Synchronization,
}

impl fmt::Display for LinkRole {
    // 中文注释：关键逻辑说明。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LinkRole::Control => "control",
            LinkRole::Data => "data",
            LinkRole::Telemetry => "telemetry",
            LinkRole::Synchronization => "sync",
        })
    }
}

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
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareTopology {
    pub name: String,
    pub nodes: Vec<HardwareNode>,
    pub links: Vec<HardwareLink>,
}

impl HardwareTopology {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }
    /// `add_node`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn add_node(&mut self, node: HardwareNode) {
        self.nodes.push(node);
    }
    /// `add_link`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn add_link(&mut self, link: HardwareLink) {
        self.links.push(link);
    }
    /// `node`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn node(&self, id: &DeviceId) -> Option<&HardwareNode> {
        self.nodes.iter().find(|node| &node.id == id)
    }
    /// `neighbors`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn neighbors(&self, id: &DeviceId) -> Vec<&HardwareLink> {
        self.links.iter().filter(|link| &link.from == id).collect()
    }
    /// `online_node_count`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn online_node_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.online).count()
    }
    /// `validate_connectivity`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `ring_plan`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn ring_plan(&self) -> Vec<DeviceId> {
        let mut nodes = self.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
        nodes.sort_by(|a, b| a.backend.cmp(&b.backend).then(a.ordinal.cmp(&b.ordinal)));
        nodes
    }
    /// `allreduce_cost_estimate`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `matmul_partition_plan`：中文注释，说明函数用途、输入约束与输出语义。
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyValidation {
    pub connected: bool,
    pub connected_nodes: usize,
    pub total_nodes: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AllReduceEstimate {
    pub nodes: Vec<DeviceId>,
    pub payload_bytes: usize,
    pub estimated_seconds: f64,
    pub ring_hops: u32,
    pub min_bandwidth_mb_s: u32,
    pub latency_us: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulShard {
    pub device: DeviceId,
    pub row_range: std::ops::Range<usize>,
    pub col_range: std::ops::Range<usize>,
    pub k: usize,
}

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

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
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
