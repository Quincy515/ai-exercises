use std::{future::pending, sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use serial_test::serial;
use tokio::{sync::Notify, time::timeout};

use super::{ExecutionState, RedisStreamTask, TASK_REGISTRY};
use crate::domain::external::{MessageQueue, SharedTask, Task, TaskRunner};

// 生命周期测试直接提供内存替身，验证任务调度时无需启动 Redis。
struct UnusedQueue;

#[async_trait]
impl MessageQueue for UnusedQueue {
    async fn put(&self, _message: Value) -> Result<String> {
        panic!("生命周期测试不读写队列")
    }

    async fn get(
        &self,
        _start_id: Option<&str>,
        _block_ms: Option<usize>,
    ) -> Result<Option<(String, Value)>> {
        panic!("生命周期测试不读写队列")
    }

    async fn pop(&self) -> Result<Option<(String, Value)>> {
        panic!("生命周期测试不读写队列")
    }

    async fn clear(&self) -> Result<()> {
        panic!("生命周期测试不读写队列")
    }

    async fn is_empty(&self) -> Result<bool> {
        panic!("生命周期测试不读写队列")
    }

    async fn size(&self) -> Result<usize> {
        panic!("生命周期测试不读写队列")
    }

    async fn delete_message(&self, _message_id: &str) -> Result<bool> {
        panic!("生命周期测试不读写队列")
    }
}

#[derive(Default)]
struct RecordingRunner {
    keep_running: bool,
    block_cancel: bool,
    block_done: bool,
    calls: std::sync::Mutex<Vec<&'static str>>,
    started: Notify,
    cancel_started: Notify,
    release_cancel: Notify,
    done_started: Notify,
    release_done: Notify,
}

impl RecordingRunner {
    fn record(&self, call: &'static str) {
        self.calls.lock().unwrap().push(call);
    }

    fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl TaskRunner for RecordingRunner {
    async fn invoke(&self, _task: SharedTask) -> Result<()> {
        self.record("invoke");
        self.started.notify_one();
        if self.keep_running {
            pending::<()>().await;
        }
        Ok(())
    }

    async fn destroy(&self) -> Result<()> {
        self.record("destroy");
        Ok(())
    }

    async fn on_cancel(&self, _task: SharedTask) -> Result<()> {
        self.record("cancel");
        self.cancel_started.notify_one();
        if self.block_cancel {
            self.release_cancel.notified().await;
        }
        self.record("cancel_finished");
        Ok(())
    }

    async fn on_done(&self, _task: SharedTask) -> Result<()> {
        self.record("done");
        self.done_started.notify_one();
        if self.block_done {
            self.release_done.notified().await;
        }
        self.record("done_finished");
        Ok(())
    }
}

fn fixture(runner: Arc<RecordingRunner>) -> RedisStreamTask {
    let task = RedisStreamTask {
        task_runner: runner,
        id: uuid::Uuid::new_v4().to_string(),
        execution_task: Arc::new(std::sync::Mutex::new(ExecutionState::default())),
        input_stream: Arc::new(UnusedQueue),
        output_stream: Arc::new(UnusedQueue),
    };
    TASK_REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(task.id.clone(), task.clone());
    task
}

async fn wait_for(signal: &Notify) {
    timeout(Duration::from_secs(2), signal.notified())
        .await
        .expect("任务应到达指定生命周期阶段");
}

async fn wait_finished(task: &RedisStreamTask) {
    timeout(Duration::from_secs(2), async {
        while !task.done() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("任务应完成全部异步收尾");
}

#[tokio::test(flavor = "current_thread")]
#[serial(redis_stream_task)]
async fn cancellation_before_first_poll_still_runs_callbacks_once() {
    let runner = Arc::new(RecordingRunner::default());
    let task = fixture(runner.clone());

    // 当前线程尚未让出执行权，刚创建的执行任务还没有首次 poll。
    task.invoke().await.unwrap();
    assert!(task.cancel());
    assert!(!task.cancel());
    wait_finished(&task).await;

    assert_eq!(
        runner.calls(),
        vec!["cancel", "cancel_finished", "done", "done_finished"]
    );
    assert!(RedisStreamTask::get(task.id()).unwrap().is_none());
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn running_task_accepts_one_cancellation_and_finishes_once() {
    let runner = Arc::new(RecordingRunner {
        keep_running: true,
        ..RecordingRunner::default()
    });
    let task = fixture(runner.clone());
    task.invoke().await.unwrap();
    wait_for(&runner.started).await;

    assert!(task.cancel());
    assert!(!task.cancel());
    wait_finished(&task).await;
    assert!(!task.cancel());

    assert_eq!(
        runner.calls(),
        vec![
            "invoke",
            "cancel",
            "cancel_finished",
            "done",
            "done_finished"
        ]
    );
    assert!(RedisStreamTask::get(task.id()).unwrap().is_none());
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn normal_completion_does_not_call_cancellation_callback() {
    let runner = Arc::new(RecordingRunner::default());
    let task = fixture(runner.clone());

    task.invoke().await.unwrap();
    wait_finished(&task).await;

    assert!(!task.cancel());
    assert_eq!(runner.calls(), vec!["invoke", "done", "done_finished"]);
    assert!(RedisStreamTask::get(task.id()).unwrap().is_none());
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn cancellation_callback_keeps_task_registered_and_prevents_restart() {
    let runner = Arc::new(RecordingRunner {
        keep_running: true,
        block_cancel: true,
        ..RecordingRunner::default()
    });
    let task = fixture(runner.clone());
    task.invoke().await.unwrap();
    wait_for(&runner.started).await;
    assert!(task.cancel());
    wait_for(&runner.cancel_started).await;

    assert!(!task.done());
    assert!(RedisStreamTask::get(task.id()).unwrap().is_some());
    task.invoke().await.unwrap();
    assert!(!task.cancel());
    assert_eq!(runner.calls(), vec!["invoke", "cancel"]);

    runner.release_cancel.notify_one();
    wait_finished(&task).await;
    assert_eq!(
        runner.calls(),
        vec![
            "invoke",
            "cancel",
            "cancel_finished",
            "done",
            "done_finished"
        ]
    );
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn late_cancellation_during_completion_does_not_change_normal_exit() {
    let runner = Arc::new(RecordingRunner {
        block_done: true,
        ..RecordingRunner::default()
    });
    let task = fixture(runner.clone());
    task.invoke().await.unwrap();
    wait_for(&runner.done_started).await;

    assert!(!task.done());
    assert!(!task.cancel());
    assert!(RedisStreamTask::get(task.id()).unwrap().is_some());
    task.invoke().await.unwrap();
    assert_eq!(runner.calls(), vec!["invoke", "done"]);

    runner.release_done.notify_one();
    wait_finished(&task).await;
    assert_eq!(runner.calls(), vec!["invoke", "done", "done_finished"]);
}

#[tokio::test]
#[serial(redis_stream_task)]
async fn destroy_waits_for_callbacks_and_blocks_new_execution() {
    let runner = Arc::new(RecordingRunner {
        keep_running: true,
        block_cancel: true,
        block_done: true,
        ..RecordingRunner::default()
    });
    let task = fixture(runner.clone());
    task.invoke().await.unwrap();
    wait_for(&runner.started).await;

    let destroying = tokio::spawn(RedisStreamTask::destroy());
    wait_for(&runner.cancel_started).await;
    assert!(!destroying.is_finished());
    assert!(!task.done());
    assert!(task.invoke().await.is_err());
    assert_eq!(runner.calls(), vec!["invoke", "cancel"]);

    runner.release_cancel.notify_one();
    wait_for(&runner.done_started).await;
    assert!(!destroying.is_finished());
    assert!(!task.done());
    assert!(task.invoke().await.is_err());
    assert_eq!(
        runner.calls(),
        vec!["invoke", "cancel", "cancel_finished", "done"]
    );

    runner.release_done.notify_one();
    timeout(Duration::from_secs(2), destroying)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(task.done());
    assert!(task.invoke().await.is_err());
    assert_eq!(
        runner.calls(),
        vec![
            "invoke",
            "cancel",
            "cancel_finished",
            "done",
            "done_finished",
            "destroy"
        ]
    );
    assert!(RedisStreamTask::get(task.id()).unwrap().is_none());
}
