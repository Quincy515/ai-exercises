use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use tokio::task::{AbortHandle, JoinHandle};
use tracing::{error, info};

use crate::domain::external::SharedMessageQueue;
use crate::domain::external::SharedTask;
use crate::domain::external::SharedTaskRunner;
use crate::domain::external::Task;
use crate::infrastructure::external::RedisStreamMessageQueue;

/// 定义一个全局变量用于存储所有已注册的任务。
/// Define a global variable for storing all registered tasks.
static TASK_REGISTRY: OnceLock<Mutex<HashMap<String, RedisStreamTask>>> = OnceLock::new();

/// 执行任务可取消；监督任务负责等待执行结束并完成异步收尾。
struct RunningTask {
    execution_abort: AbortHandle,
    completion: Option<JoinHandle<()>>,
    // 销毁时取走等待句柄后，仍可查询监督任务是否结束。
    completion_status: AbortHandle,
    cancel_requested: bool,
}

#[derive(Default)]
struct ExecutionState {
    running: Option<RunningTask>,
    // 销毁期间关闭启动入口，避免旧任务收尾时启动新的执行任务。
    destroying: bool,
}

/// 基于 Redis Stream 的任务。
/// Task backed by Redis Stream.
#[derive(Clone)]
pub struct RedisStreamTask {
    task_runner: SharedTaskRunner,
    id: String,
    // 定义在后台执行的任务。
    // Store the background task handle.
    execution_task: Arc<Mutex<ExecutionState>>,
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
            execution_task: Arc::new(Mutex::new(ExecutionState::default())),
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
    async fn on_task_done(&self) {
        let task = Arc::new(self.clone()) as SharedTask;

        // 完成回调结束后再移除注册表，确保异步收尾期间仍能访问任务。
        if let Err(err) = self.task_runner.on_done(task).await {
            error!("任务 [{}] 完成回调执行失败: {err}", self.id);
        }

        if let Err(err) = self.cleanup_registry() {
            error!("任务 [{:?}] 清理注册表失败: {err}", self.id);
        }
    }

    /// 使用 TaskRunner 执行任务。
    /// Execute the task with its TaskRunner.
    async fn execute_task(task: RedisStreamTask, execution: JoinHandle<Result<()>>) {
        let task_id = task.id.clone();
        let task_runner = task.task_runner.clone();
        let shared_task = Arc::new(task.clone()) as SharedTask;

        // 独立监督任务等待真实执行结果，覆盖首次 poll 前取消与正常完成的竞争。
        match execution.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => error!("任务 [{task_id}] 执行出现异常: {err}"),
            Err(err) if err.is_cancelled() => {
                if let Err(err) = task_runner.on_cancel(shared_task).await {
                    error!("任务 [{task_id}] 取消回调执行失败: {err}");
                }
            }
            Err(err) => error!("任务 [{task_id}] 执行任务异常终止: {err}"),
        }

        task.on_task_done().await;
    }

    /// 关闭启动入口并取消执行，交出监督任务句柄供销毁过程等待。
    fn begin_destroy(&self) -> Result<Option<JoinHandle<()>>> {
        let mut state = self
            .execution_task
            .lock()
            .map_err(|err| anyhow!("任务执行句柄锁定失败: {err}"))?;
        state.destroying = true;
        Ok(state.running.as_mut().and_then(|running| {
            running.execution_abort.abort();
            running.cancel_requested = true;
            running.completion.take()
        }))
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

        if execution_task.destroying {
            bail!("任务 [{}] 正在销毁或已销毁", self.id);
        }

        if execution_task
            .running
            .as_ref()
            .is_some_and(|running| !running.completion_status.is_finished())
        {
            return Ok(());
        }

        let task = self.clone();
        let task_id = self.id.clone();
        let task_runner = self.task_runner.clone();
        let shared_task = Arc::new(task.clone()) as SharedTask;
        let execution = tokio::spawn(async move { task_runner.invoke(shared_task).await });
        let execution_abort = execution.abort_handle();
        let completion = tokio::spawn(async move {
            Self::execute_task(task, execution).await;
        });
        let completion_status = completion.abort_handle();
        execution_task.running = Some(RunningTask {
            execution_abort,
            completion: Some(completion),
            completion_status,
            cancel_requested: false,
        });

        info!("任务 [{task_id}] 开始执行");
        Ok(())
    }

    /// 取消当前执行的任务。
    /// Cancel the current running task.
    fn cancel(&self) -> bool {
        let lock_result = self.execution_task.lock();
        let Ok(mut execution_task) = lock_result else {
            error!("任务 [{:?}] 执行句柄锁定失败", self.id);
            return false;
        };

        let Some(running) = execution_task.running.as_mut() else {
            drop(execution_task);
            if let Err(err) = self.cleanup_registry() {
                error!("任务 [{:?}] 清理注册表失败: {err}", self.id);
            }
            return false;
        };
        if running.cancel_requested || running.execution_abort.is_finished() {
            return false;
        }

        // 只取消执行任务；监督任务继续完成取消回调、完成回调和注册表清理。
        running.cancel_requested = true;
        running.execution_abort.abort();
        info!("任务 [{:?}] 已发出取消请求", self.id);
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
                .running
                .as_ref()
                .is_none_or(|running| running.completion_status.is_finished()),
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
        // 先关闭全部任务的启动入口并发出取消；等待前释放标准互斥锁。
        let pending = tasks
            .into_iter()
            .map(|task| task.begin_destroy().map(|completion| (task, completion)))
            .collect::<Result<Vec<_>>>()?;
        for (task, completion) in pending {
            if let Some(completion) = completion {
                if let Err(err) = completion.await {
                    error!("任务 [{}] 等待收尾失败: {err}", task.id);
                }
            }
            task.task_runner.destroy().await?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "redis_stream_task_tests.rs"]
mod tests;
