use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::File;

/// 文件模型数据仓库；只保存元数据，文件内容由 FileStorage 管理。
#[async_trait]
pub trait FileRepository: Send + Sync {
    /// 新增或更新文件信息；相同领域文件 id 已存在时更新。
    async fn save(&self, file: File) -> Result<()>;

    /// 根据传递的文件 id 获取文件信息；不存在时返回 None，读取失败时返回 Err。
    async fn get_by_id(&self, file_id: &str) -> Result<Option<File>>;
}
