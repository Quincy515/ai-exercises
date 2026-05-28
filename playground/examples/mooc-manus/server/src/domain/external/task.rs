use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use super::message_queue::SharedMessageQueue;

pub type SharedTask = Arc<dyn Task>;
pub type SharedTaskRunner = Arc<dyn TaskRunner>;

/// 任务运行器，负责任务执行、任务销毁和完成回调。
/// Task runner responsible for execution, teardown, and completion callbacks.
#[async_trait]
pub trait TaskRunner: Send + Sync {
    /// 调用任务并执行。
    /// Invoke and execute a task.
    async fn invoke(&self, task: SharedTask) -> Result<()>;

    /// 销毁任务运行器并释放资源，包括关闭连接、清理临时数据和后台进程。
    /// Destroy the runner and release resources such as connections, temporary data, and background processes.
    async fn destroy(&self) -> Result<()>;

    /// 执行任务完成时触发的回调函数。
    /// Callback triggered after task execution completes.
    async fn on_done(&self, task: SharedTask) -> Result<()>;
}

/// 任务实例协议，描述单个 Agent 任务的运行、取消和消息流访问能力。
/// Task instance protocol for running, cancelling, and accessing message streams.
#[async_trait]
pub trait Task: Send + Sync {
    /// 运行当前任务。
    /// Run the current task.
    async fn invoke(&self) -> Result<()>;

    /// 取消当前任务，返回本次调用是否成功发出取消请求。
    /// Cancel the current task and report whether the cancellation request was accepted.
    fn cancel(&self) -> bool;

    /// 返回任务的输入流。
    /// Return the task input stream.
    fn input_stream(&self) -> SharedMessageQueue;

    /// 返回任务的输出流。
    /// Return the task output stream.
    fn output_stream(&self) -> SharedMessageQueue;

    /// 返回任务 id。
    /// Return the task id.
    fn id(&self) -> &str;

    /// 返回任务是否结束。
    /// Return whether the task has finished.
    fn done(&self) -> bool;

    /// 关联函数：根据任务 id 获取对应任务。
    /// Get a task by task id.
    fn get(task_id: &str) -> Result<Option<SharedTask>>
    where
        Self: Sized;

    /// 关联函数：销毁所有任务实例。
    /// Destroy all task instances.
    async fn destroy() -> Result<()>
    where
        Self: Sized;
}
