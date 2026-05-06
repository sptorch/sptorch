use crate::proto::node_service_client::NodeServiceClient;
use crate::proto::*;
use tonic::transport::Channel;

/// A worker node that connects to the coordinator for distributed training.
pub struct Worker {
    pub rank: u32,
    pub world_size: u32,
    pub node_id: String,
    client: NodeServiceClient<Channel>,
}

impl Worker {
    /// Connect to the coordinator and register this worker.
    pub async fn connect(
        coordinator_addr: &str,
        node_id: &str,
        world_size: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client = NodeServiceClient::connect(format!("http://{}", coordinator_addr)).await?;

        let resp = client
            .register(RegisterRequest {
                node_id: node_id.to_string(),
                address: String::new(),
                world_size,
            })
            .await?
            .into_inner();

        eprintln!(
            "[sptorch] worker '{}' registered as rank {}/{}",
            node_id, resp.rank, resp.world_size
        );

        Ok(Worker {
            rank: resp.rank,
            world_size: resp.world_size,
            node_id: node_id.to_string(),
            client,
        })
    }

    /// Send local gradients for a parameter and receive the globally averaged result.
    pub async fn allreduce(
        &mut self,
        param_index: u32,
        local_grad: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let data: Vec<u8> = local_grad.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();

        let resp = self
            .client
            .all_reduce_gradients(GradientChunk {
                param_index,
                rank: self.rank,
                world_size: self.world_size,
                data,
                shape: shape_u32,
            })
            .await?
            .into_inner();

        let averaged: Vec<f32> = resp
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(averaged)
    }

    /// 分布式屏障同步：阻塞直到所有 worker 都到达同一 step。
    pub async fn barrier(&mut self, step: u64) -> Result<(), Box<dyn std::error::Error>> {
        self.client
            .barrier(BarrierRequest {
                node_id: self.node_id.clone(),
                step,
            })
            .await?;
        Ok(())
    }

    /// Heartbeat check.
    pub async fn heartbeat(&mut self, step: u64) -> Result<u64, Box<dyn std::error::Error>> {
        let resp = self
            .client
            .heartbeat(HeartbeatRequest {
                node_id: self.node_id.clone(),
                step,
            })
            .await?
            .into_inner();
        Ok(resp.global_step)
    }
}
