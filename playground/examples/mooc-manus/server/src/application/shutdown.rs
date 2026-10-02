use std::{future::Future, sync::Arc};

use tokio::sync::{watch, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// 应用关闭标记，供长连接结束等待；watch 保留状态，覆盖晚到的订阅。
#[derive(Clone)]
pub struct ShutdownSignal {
    closing: watch::Sender<bool>,
    preparations: Arc<RwLock<()>>,
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self {
            closing: watch::channel(false).0,
            preparations: Arc::new(RwLock::new(())),
        }
    }
}

impl ShutdownSignal {
    pub fn notify(&self) {
        self.closing.send_replace(true);
    }

    pub fn cancelled(&self) -> impl Future<Output = ()> + Send + 'static {
        let mut receiver = self.closing.subscribe();
        async move {
            let _ = receiver.wait_for(|closing| *closing).await;
        }
    }

    /// 准备会话任务时持读锁；关闭标记发布后拒绝新的准备过程。
    pub async fn begin_preparation(&self) -> Option<RwLockReadGuard<'_, ()>> {
        let guard = self.preparations.read().await;
        if *self.closing.borrow() {
            None
        } else {
            Some(guard)
        }
    }

    /// 等待已进入的准备过程完成或被取消，再独占执行任务销毁快照。
    pub async fn finish_preparations(&self) -> RwLockWriteGuard<'_, ()> {
        self.preparations.write().await
    }
}
