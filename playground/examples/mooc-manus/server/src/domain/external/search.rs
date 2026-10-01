use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::{SearchResults, ToolResult};

/// 搜索引擎 API 接口协议
#[async_trait]
pub trait SearchEngine: Send + Sync {
    /// 根据传递的 query+date_range(时间筛选)调用搜索引擎获取工具
    async fn invoke(
        &self,
        query: String,
        date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>>;
}

/// 共享搜索引擎，保留工具构造函数接收 Box 的现有接口。
#[async_trait]
impl<T: SearchEngine + ?Sized> SearchEngine for Arc<T> {
    async fn invoke(
        &self,
        query: String,
        date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>> {
        (**self).invoke(query, date_range).await
    }
}
