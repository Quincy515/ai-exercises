use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{
    models::Message,
    services::event_sink::{EventControl, EventSink},
};

/// 流状态类型枚举。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlowStatus {
    /// 空闲中
    #[default]
    Idle,
    /// 规划中
    Planning,
    /// 执行中
    Executing,
    /// 更新中
    Updating,
    /// 汇总中
    Summarizing,
    /// 已完成
    Completed,
}

/// 基础流协议，统一流的调用入口和结束状态判断。
#[async_trait]
pub trait BaseFlow: Send + Sync {
    /// 传递用户消息运行流，等待每条事件交付后再推进后续动作。
    async fn invoke(&mut self, message: Message, sink: &mut dyn EventSink) -> Result<EventControl>;

    /// 返回流是否已经回到空闲状态。
    fn done(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::FlowStatus;

    #[test]
    fn flow_status_serializes_to_stable_values() {
        assert_eq!(
            serde_json::to_string(&FlowStatus::Idle).unwrap(),
            "\"idle\""
        );
        assert_eq!(
            serde_json::to_string(&FlowStatus::Summarizing).unwrap(),
            "\"summarizing\""
        );
        assert_eq!(
            serde_json::from_str::<FlowStatus>("\"completed\"").unwrap(),
            FlowStatus::Completed
        );
    }
}
