use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::domain::external::SharedMessageQueue;
use crate::domain::external::SharedTask;
use crate::domain::external::SharedTaskRunner;
use crate::domain::external::Task;
use crate::infrastructure::external::RedisStreamMessageQueue;

/// 定义一个全局变量用于存储所有已注册的任务。
/// Define a global variable for storing all registered tasks.
static TASK_REGISTRY: OnceLock<Mutex<HashMap<String, RedisStreamTask>>> = OnceLock::new();

/// 基于 Redis Stream 的任务。
/// Task backed by Redis Stream.
#[derive(Clone)]
pub struct RedisStreamTask {
    task_runner: SharedTaskRunner,
    id: String,
    // 定义在后台执行的任务。
    // Store the background task handle.
    execution_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    input_stream: SharedMessageQueue,
    output_stream: SharedMessageQueue,
}

impl RedisStreamTask {
    /// 构造函数：传递任务运行器完成 Task 初始化。
    /// Initialize the task with the provided task runner.
    pub fn new(task_runner: SharedTaskRunner, redis: MultiplexedConnection) -> Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let input_stream_name = format!("task:input:{id}");
        let output_stream_name = format!("task:output:{id}");

        let task = Self {
            task_runner,
            id: id.clone(),
            execution_task: Arc::new(Mutex::new(None)),
            input_stream: Arc::new(RedisStreamMessageQueue::new(
                input_stream_name,
                redis.clone(),
                None,
            )),
            output_stream: Arc::new(RedisStreamMessageQueue::new(
                output_stream_name,
                redis,
                None,
            )),
        };

        // 将任务注册到全局任务注册表中。
        // Register the task in the global task registry.
        let mut registry = TASK_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|err| anyhow!("任务注册表锁定失败: {err}"))?;
        registry.insert(id, task.clone());

        Ok(task)
    }

    /// 清除全局任务注册表中的当前任务。
    /// Remove the current task from the global task registry.
    fn cleanup_registry(&self) -> Result<()> {
        Self::cleanup_registry_by_id(&self.id)
    }

    /// 根据任务 id 清理注册表。
    /// Remove a task from the registry by task id.
    fn cleanup_registry_by_id(task_id: &str) -> Result<()> {
        let mut registry = TASK_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|err| anyhow!("任务注册表锁定失败: {err}"))?;

        if registry.remove(task_id).is_some() {
            info!("任务 [{task_id}] 从注册中心移除");
        }

        Ok(())
    }

    /// 任务结束时的回调函数。
    /// Callback executed after the task finishes.
    fn on_task_done(&self) {
        let task_runner = self.task_runner.clone();
        let task = Arc::new(self.clone()) as SharedTask;
        let task_id = self.id.clone();

        tokio::spawn(async move {
            if let Err(err) = task_runner.on_done(task).await {
                error!("任务 [{task_id}] 完成回调执行失败: {err}");
            }
        });

        if let Err(err) = self.cleanup_registry() {
            error!("任务 [{:?}] 清理注册表失败: {err}", self.id);
        }
    }

    /// 使用 TaskRunner 执行任务。
    /// Execute the task with its TaskRunner.
    async fn execute_task(task: RedisStreamTask) {
        let task_id = task.id.clone();
        let task_runner = task.task_runner.clone();
        let shared_task = Arc::new(task.clone()) as SharedTask;

        if let Err(err) = task_runner.invoke(shared_task).await {
            error!("任务 [{task_id}] 执行出现异常: {err}");
        }

        task.on_task_done();
    }
}

#[async_trait]
impl Task for RedisStreamTask {
    /// 使用提交的 task_runner 来运行任务。
    /// Run the task with the provided task runner.
    async fn invoke(&self) -> Result<()> {
        let mut execution_task = self
            .execution_task
            .lock()
            .map_err(|err| anyhow!("任务执行句柄锁定失败: {err}"))?;

        if execution_task
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
        {
            return Ok(());
        }

        let task = self.clone();
        let task_id = self.id.clone();
        let handle = tokio::spawn(async move {
            Self::execute_task(task).await;
        });
        *execution_task = Some(handle);

        info!("任务 [{task_id}] 开始执行");
        Ok(())
    }

    /// 取消当前执行的任务。
    /// Cancel the current running task.
    fn cancel(&self) -> bool {
        let mut aborted = false;

        let lock_result = self.execution_task.lock();
        let Ok(mut execution_task) = lock_result else {
            error!("任务 [{:?}] 执行句柄锁定失败", self.id);
            return false;
        };

        if let Some(handle) = execution_task.take() {
            if !handle.is_finished() {
                handle.abort();
                aborted = true;
                info!("任务 [{:?}] 已取消", self.id);
            }
        }

        drop(execution_task);

        if aborted {
            self.on_task_done();
        } else if let Err(err) = self.cleanup_registry() {
            error!("任务 [{:?}] 清理注册表失败: {err}", self.id);
            return false;
        }

        true
    }

    fn input_stream(&self) -> SharedMessageQueue {
        self.input_stream.clone()
    }

    fn output_stream(&self) -> SharedMessageQueue {
        self.output_stream.clone()
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn done(&self) -> bool {
        match self.execution_task.lock() {
            Ok(execution_task) => execution_task
                .as_ref()
                .is_none_or(|handle| handle.is_finished()),
            Err(err) => {
                error!("任务 [{:?}] 执行句柄锁定失败: {err}", self.id);
                true
            }
        }
    }

    fn get(task_id: &str) -> Result<Option<SharedTask>> {
        let registry = TASK_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|err| anyhow!("任务注册表锁定失败: {err}"))?;

        Ok(registry
            .get(task_id)
            .cloned()
            .map(|task| Arc::new(task) as SharedTask))
    }

    /// 关联函数：销毁所有任务实例。
    /// Destroy all task instances.
    async fn destroy() -> Result<()> {
        // 1. 先把任务 clone 出来并清空注册表。
        // Clone tasks out first, then clear the registry.
        let tasks = {
            let mut registry = TASK_REGISTRY
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .map_err(|err| anyhow!("任务注册表锁定失败: {err}"))?;

            let tasks = registry.values().cloned().collect::<Vec<_>>();
            registry.clear();
            tasks
        };

        // 2. 遍历任务列表，取消执行并销毁任务运行器。
        // Cancel each task and destroy its task runner.
        for task in tasks {
            task.cancel();
            task.task_runner.destroy().await?;
        }

        Ok(())
    }
}
