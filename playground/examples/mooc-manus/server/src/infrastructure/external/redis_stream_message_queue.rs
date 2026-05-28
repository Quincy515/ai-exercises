use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use redis::streams::StreamMaxlen;
use redis::AsyncCommands;
use redis::{
    aio::MultiplexedConnection,
    streams::{StreamRangeReply, StreamReadOptions, StreamReadReply},
};
use redis::{from_redis_value, ExistenceCheck, Script, SetExpiry, SetOptions};
use tracing::{debug, error, info};

use crate::domain::external::{MessageQueue, MessageQueuePayload};

/// 基于 RedisStream 的消息队列
pub struct RedisStreamMessageQueue {
    stream_name: String,
    redis: MultiplexedConnection,
    lock_expire_seconds: u64,
}

impl RedisStreamMessageQueue {
    /// 构造函数，完成 Redis-stream 的初始化，涵盖名字、锁的时间
    pub fn new(
        stream_name: String,
        redis: MultiplexedConnection,
        lock_expire_seconds: Option<u64>,
    ) -> Self {
        Self {
            stream_name,
            redis,
            lock_expire_seconds: lock_expire_seconds.unwrap_or(10),
        }
    }

    /// 根据传递的 lock 键构建一个分布式锁
    async fn acquire_lock(
        &self,
        lock_key: &str,
        timeout_seconds: Option<usize>,
    ) -> Result<Option<String>> {
        // 1. 创建锁对应的值
        let lock_value = uuid::Uuid::new_v4().to_string();
        // 最多等待 end_time 秒，来完成抢夺锁
        let end_time = timeout_seconds.unwrap_or(5);
        let mut remaining_ms = end_time.saturating_mul(1_000);
        let mut conn = self.redis.clone();

        // 2. 构建一个循环，在剩余的时间内，每 100ms 重新尝试获取锁
        while remaining_ms > 0 {
            let options = SetOptions::default()
                .conditional_set(ExistenceCheck::NX) // 仅在键不存在时设置
                .with_expiration(SetExpiry::PX(self.lock_expire_seconds * 1_000)); // 设置锁的过期时间

            // 3. 使用 redis 的 set 方法，将 lock_key 和 lock_value 存储到 redis 中
            // SET NX PX 是单条原子命令，同时完成抢锁和设置过期时间。
            // SET NX PX is one atomic command that acquires the lock and sets its expiry together.
            let result: Option<String> = conn.set_options(lock_key, &lock_value, options).await?;

            // 4. 如果设置成功，返回锁的值
            if result.is_some() {
                return Ok(Some(lock_value));
            }

            // 5. 等待 100ms 后重试并将剩余时间递减
            tokio::time::sleep(Duration::from_millis(100)).await;
            remaining_ms = remaining_ms.saturating_sub(100);
        }

        Ok(None)
    }

    /// 根据传递的 lock_key 和 lock_value 释放分布式锁
    async fn release_lock(&self, lock_key: &str, lock_value: &str) -> Result<bool> {
        let mut conn = self.redis.clone();
        // 构建一段 redis 脚本用于释放分布式锁
        let deleted: usize = Script::new(
            r#"
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
            "#,
        )
        .key(lock_key)
        .arg(lock_value)
        .invoke_async(&mut conn)
        .await?;

        Ok(deleted == 1)
    }
}

#[async_trait]
impl MessageQueue for RedisStreamMessageQueue {
    /// 往 Redis-stream 中添加一条消息并返回 id
    async fn put(&self, message: MessageQueuePayload) -> Result<String> {
        info!(
            "往消息队列中添加一条消息： [{:?}]: {:?}",
            self.stream_name, message
        );
        let mut conn = self.redis.clone();
        let data = serde_json::to_string(&message)?;
        let id: String = conn.xadd(&self.stream_name, "*", &[("data", data)]).await?;
        Ok(id)
    }

    /// 从 Redis-stream 获取一条数据
    async fn get(
        &self,
        start_id: Option<&str>,
        block_ms: Option<usize>,
    ) -> Result<Option<(String, MessageQueuePayload)>> {
        info!(
            "从消息队列中获取一条消息： [{:?}] start_id={:?} block_ms={:?}",
            self.stream_name, start_id, block_ms
        );
        let mut conn = self.redis.clone();

        // count(1) 表示一次只取一条消息。
        // count(1) reads one message at a time.
        let options = match block_ms {
            Some(ms) => StreamReadOptions::default().count(1).block(ms),
            None => StreamReadOptions::default().count(1),
        };

        let messages: StreamReadReply = conn
            .xread_options(&[&self.stream_name], &[start_id.unwrap_or("0")], &options)
            .await?;

        let Some(message) = messages
            .keys
            .into_iter()
            .find(|stream| stream.key == self.stream_name)
            .and_then(|stream| stream.ids.into_iter().next())
        else {
            return Ok(None);
        };

        let Some(Ok(data)) = message
            .map
            .get("data")
            .cloned()
            .map(from_redis_value::<String>)
        else {
            error!("从消息队列 [{:?}] 获取 data 字段失败", self.stream_name);
            return Ok(None);
        };

        let Ok(payload) = serde_json::from_str::<MessageQueuePayload>(&data) else {
            error!("解析消息队列 [{:?}] 的 data 字段失败", self.stream_name);
            return Ok(None);
        };

        Ok(Some((message.id, payload)))
    }

    /// 从消息队列中获取第一条消息并删除
    async fn pop(&self) -> Result<Option<(String, MessageQueuePayload)>> {
        // 1. 记录日志
        debug!("从消息队列 [{:?}] 中获取第一条消息并删除", self.stream_name);
        let lock_key = format!("lock:{}:pop", self.stream_name);

        // 2. 构建分布式锁，如果分布式锁创建失败则返回 None
        let Some(lock_value) = self.acquire_lock(&lock_key, None).await? else {
            return Ok(None);
        };

        let mut conn = self.redis.clone();
        let result: Result<Option<(String, MessageQueuePayload)>> = async {
            // 3. 分布式锁创建成功，获取消息队列中的第一条消息
            let messages: StreamRangeReply =
                conn.xrange_count(&self.stream_name, "-", "+", 1).await?;

            let Some(message) = messages.ids.into_iter().next() else {
                return Ok(None);
            };

            // 4. 取出消息 id 和消息 data 字段。
            // 4. Extract the message id and the data field.
            let message_id = message.id;
            let data = message.map.get("data").cloned();

            // 5. 删除消息队列中的 message 数据。
            // 5. Delete the message entry from the queue.
            let deleted: usize = conn.xdel(&self.stream_name, &[message_id.as_str()]).await?;
            if deleted == 0 {
                return Ok(None);
            }

            let Some(Ok(data)) = data.map(from_redis_value::<String>) else {
                error!("从消息队列 [{:?}] 获取 data 字段失败", self.stream_name);
                return Ok(None);
            };

            let Ok(payload) = serde_json::from_str::<MessageQueuePayload>(&data) else {
                error!("解析消息队列 [{:?}] 的 data 字段失败", self.stream_name);
                return Ok(None);
            };

            Ok(Some((message_id, payload)))
        }
        .await;

        // 6. 释放分布式锁
        let release_result = self.release_lock(&lock_key, &lock_value).await;
        if let Err(err) = release_result {
            error!("释放消息队列 [{:?}] 的 pop 锁失败: {err}", self.stream_name);
            if result.is_ok() {
                return Err(err);
            }
        }

        result
    }

    /// 清除 Redis-stream 中的所有消息
    async fn clear(&self) -> Result<()> {
        let mut conn = self.redis.clone();
        // MAXLEN 0 表示保留 0 条消息，也就是清空 Stream。
        // MAXLEN 0 keeps zero entries, effectively clearing the stream.
        conn.xtrim::<_, ()>(self.stream_name.clone(), StreamMaxlen::Equals(0))
            .await?;
        Ok(())
    }

    /// 检查 Redis-stream 是否为空
    async fn is_empty(&self) -> Result<bool> {
        self.size().await.map(|count| count == 0)
    }

    /// 获取 Redis-stream 中消息的数量
    async fn size(&self) -> Result<usize> {
        let mut conn = self.redis.clone();
        let count = conn.xlen(self.stream_name.clone()).await?;
        Ok(count)
    }

    /// 根据传递的消息 id 从 Redis-stream 删除数据
    async fn delete_message(&self, message_id: &str) -> Result<bool> {
        let mut conn = self.redis.clone();
        let result: usize = conn.xdel(self.stream_name.clone(), &[message_id]).await?;
        Ok(result > 0)
    }
}
