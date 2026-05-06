/// Online training monitor: tracks loss, detects degradation, triggers rollback.
pub struct TrainingMonitor {
    /// Rolling window of recent loss values
    loss_history: Vec<f32>,
    window_size: usize,
    /// Best average loss seen so far
    best_avg_loss: f32,
    /// Threshold: if current avg exceeds best by this ratio, trigger rollback
    degradation_threshold: f32,
    /// Total samples processed
    total_samples: u64,
    /// Number of rollbacks triggered
    rollback_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MonitorAction {
    Continue,
    Rollback { current_avg: f32, best_avg: f32 },
}

impl TrainingMonitor {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(window_size: usize, degradation_threshold: f32) -> Self {
        TrainingMonitor {
            loss_history: Vec::new(),
            window_size,
            best_avg_loss: f32::MAX,
            degradation_threshold,
            total_samples: 0,
            rollback_count: 0,
        }
    }
    /// `record_loss`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn record_loss(&mut self, loss: f32) -> MonitorAction {
        self.loss_history.push(loss);
        self.total_samples += 1;

        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
        }

        if self.loss_history.len() < self.window_size {
            return MonitorAction::Continue;
        }

        let current_avg = self.rolling_avg();

        if current_avg < self.best_avg_loss {
            self.best_avg_loss = current_avg;
            return MonitorAction::Continue;
        }

        if current_avg > self.best_avg_loss * (1.0 + self.degradation_threshold) {
            self.rollback_count += 1;
            return MonitorAction::Rollback {
                current_avg,
                best_avg: self.best_avg_loss,
            };
        }

        MonitorAction::Continue
    }
    /// `reset_after_rollback`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn reset_after_rollback(&mut self) {
        self.loss_history.clear();
    }
    /// `rolling_avg`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn rolling_avg(&self) -> f32 {
        if self.loss_history.is_empty() {
            return f32::MAX;
        }
        self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
    }
    /// `best_avg_loss`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn best_avg_loss(&self) -> f32 {
        self.best_avg_loss
    }
    /// `total_samples`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
    /// `rollback_count`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn rollback_count(&self) -> u32 {
        self.rollback_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_monitor_improving_loss() {
        let mut mon = TrainingMonitor::new(3, 0.2);
        assert_eq!(mon.record_loss(3.0), MonitorAction::Continue);
        assert_eq!(mon.record_loss(2.5), MonitorAction::Continue);
        assert_eq!(mon.record_loss(2.0), MonitorAction::Continue); // window full, avg=2.5
        assert_eq!(mon.record_loss(1.5), MonitorAction::Continue); // avg=2.0, improving
        assert!(mon.best_avg_loss() < 2.5);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_monitor_triggers_rollback() {
        let mut mon = TrainingMonitor::new(3, 0.1); // 10% threshold
                                                    // Establish a good baseline
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(1.0); // avg=1.0, best=1.0

        // Sudden degradation — fill the window with bad values
        mon.record_loss(1.5);
        mon.record_loss(1.5);
        let action = mon.record_loss(1.5); // avg=1.5, > 1.0*1.1=1.1

        match action {
            MonitorAction::Rollback { current_avg, best_avg } => {
                assert!((current_avg - 1.5).abs() < 1e-6);
                assert!((best_avg - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected rollback"),
        }
        assert!(mon.rollback_count() >= 1);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_monitor_no_rollback_within_threshold() {
        let mut mon = TrainingMonitor::new(3, 0.5); // 50% threshold (lenient)
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(1.0); // best=1.0

        // Slight increase, within threshold
        mon.record_loss(1.2);
        mon.record_loss(1.2);
        let action = mon.record_loss(1.2); // avg=1.2, < 1.0*1.5=1.5
        assert_eq!(action, MonitorAction::Continue);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_monitor_reset_after_rollback() {
        let mut mon = TrainingMonitor::new(2, 0.1);
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(5.0);
        mon.record_loss(5.0); // triggers rollback

        mon.reset_after_rollback();
        assert_eq!(mon.rolling_avg(), f32::MAX); // history cleared
        assert!((mon.best_avg_loss() - 1.0).abs() < 1e-6); // best preserved
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_monitor_sample_count() {
        let mut mon = TrainingMonitor::new(3, 0.2);
        mon.record_loss(1.0);
        mon.record_loss(2.0);
        mon.record_loss(3.0);
        assert_eq!(mon.total_samples(), 3);
    }
}
