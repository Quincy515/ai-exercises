use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

pub type MessageQueuePayload = Value;
pub type SharedMessageQueue = Arc<dyn MessageQueue>;

/// 任务消息队列，承载 Agent 任务的输入消息和输出事件。
/// Message queue for Agent task input messages and output events.
#[async_trait]
pub trait MessageQueue: Send + Sync {
    /// 写入一条任务消息。
    /// Push one task message into the queue.
    async fn put(&self, message: MessageQueuePayload) -> Result<()>;

    /// 根据传递的开始 id + 阻塞时间，获取一条数据
    async fn get(
        &self,
        start_id: Option<&str>,
        block_ms: Option<u64>,
    ) -> Result<Option<(&str, MessageQueuePayload)>>;

    /// 获取并移除消息队列中的第一条消息
    async fn pop(&self) -> Result<Option<(&str, MessageQueuePayload)>>;

    /// 清空消息队列中的所有消息
    async fn clear(&self) -> Result<()>;

    /// 判断消息队列是否为空
    async fn is_empty(&self) -> Result<bool>;

    /// 获取消息队列的长度
    async fn size(&self) -> Result<usize>;

    /// 根据传递的消息id删除队列中指定的消息
    async fn delete_message(&self, message_id: &str) -> Result<bool>;
}
