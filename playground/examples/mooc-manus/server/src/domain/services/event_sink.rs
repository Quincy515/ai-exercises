use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::Event;

/// 事件交付完成后，决定是否继续推进当前轮次。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventControl {
    Continue,
    Stop,
}

/// 逐事件接收者；生产者等待交付完成后，再执行下一项模型或工具动作。
#[async_trait]
pub trait EventSink: Send {
    async fn emit(&mut self, event: Event) -> Result<EventControl>;
}
