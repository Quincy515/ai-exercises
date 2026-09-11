use anyhow::Result;
use async_trait::async_trait;
use sea_orm::{
    sea_query::{Expr, OnConflict},
    DatabaseConnection, EntityTrait,
};

use crate::{
    domain::{models::File, repositories::FileRepository},
    models::files::{ActiveModel, Column, Entity as Files, Model},
};

/// 基于数据库的文件数据仓库。
pub struct SeaOrmFileRepository {
    db: DatabaseConnection,
}

impl SeaOrmFileRepository {
    /// 构造函数，完成数据仓库初始化；复用应用已经建立的数据库连接池。
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

#[async_trait]
impl FileRepository for SeaOrmFileRepository {
    /// 根据传递的文件模型存储或更新数据。
    async fn save(&self, file: File) -> Result<()> {
        // 1.从领域模型创建 ORM 模型，同时检查 UUID 和文件大小范围。
        let record = ActiveModel::from_file(file)?;

        // 2.文件不存在则新建文件，存在则直接更新文件，由 uuid 唯一索引原子判断。
        // 只更新业务字段，保留数据库的主键、创建时间、用户、删除标记和备注。
        let on_conflict = OnConflict::column(Column::Uuid)
            .update_columns([
                Column::Filename,
                Column::Filepath,
                Column::Key,
                Column::Extension,
                Column::MimeType,
                Column::Size,
            ])
            // insert 的冲突更新不经过 ActiveModel 保存钩子，需要显式更新时间。
            .value(Column::UpdatedAt, Expr::current_timestamp())
            .to_owned();

        // 3.一条 INSERT ... ON CONFLICT 完成保存，避免先查询再插入时的并发竞争。
        Files::insert(record)
            .on_conflict(on_conflict)
            .exec(&self.db)
            .await?;
        Ok(())
    }

    /// 根据传递的文件 id 获取文件信息。
    async fn get_by_id(&self, file_id: &str) -> Result<Option<File>> {
        // 1.根据领域 id 查询记录是否存在。
        let record = Files::find_by_uuid(file_id)?.one(&self.db).await?;

        // 2.判断文件记录是否存在返回不同的值，转换失败则继续向上传播错误。
        record.map(Model::into_file).transpose()
    }
}
