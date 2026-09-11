//! 文件的 Loco/SeaORM 模型门面。

pub use super::_entities::files::{ActiveModel, Column, Entity, Model};
use anyhow::{Context, Result};
use sea_orm::{entity::prelude::*, ActiveValue::Set, QueryFilter, Select};

use crate::domain::models::File;

pub type Files = Entity;

#[async_trait::async_trait]
impl ActiveModelBehavior for ActiveModel {
    async fn before_save<C>(self, _db: &C, insert: bool) -> std::result::Result<Self, DbErr>
    where
        C: ConnectionTrait,
    {
        if !insert && self.updated_at.is_unchanged() {
            let mut this = self;
            this.updated_at = sea_orm::ActiveValue::Set(chrono::Utc::now().into());
            Ok(this)
        } else {
            Ok(self)
        }
    }
}

// implement your read-oriented logic here
impl Model {
    /// 将 ORM 模型转换为领域模型。
    pub fn into_file(self) -> Result<File> {
        let uuid = self.uuid.context("files.uuid must not be null")?;
        let size = usize::try_from(self.size.unwrap_or_default())
            .context("files.size must be greater than or equal to 0")?;

        Ok(File {
            id: uuid.to_string(),
            filename: self.filename.unwrap_or_default(),
            filepath: self.filepath.unwrap_or_default(),
            key: self.key.unwrap_or_default(),
            extension: self.extension.unwrap_or_default(),
            mime_type: self.mime_type.unwrap_or_default(),
            size,
        })
    }
}

// implement your write-oriented logic here
impl ActiveModel {
    /// 从领域模型创建 ORM 模型。
    pub fn from_file(file: File) -> Result<Self> {
        let uuid =
            Uuid::parse_str(&file.id).with_context(|| format!("invalid file id: {}", file.id))?;
        let size = i32::try_from(file.size).with_context(|| {
            format!("file.size is larger than database i32 range: {}", file.size)
        })?;

        Ok(Self {
            uuid: Set(Some(uuid)),
            filename: Set(Some(file.filename)),
            filepath: Set(Some(file.filepath)),
            key: Set(Some(file.key)),
            extension: Set(Some(file.extension)),
            mime_type: Set(Some(file.mime_type)),
            size: Set(Some(size)),
            user_id: Set(None),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            ..Default::default()
        })
    }

    /// 从领域模型更新数据，保留数据库的主键、时间、用户、删除标记和备注。
    pub fn apply_file(&mut self, file: File) -> Result<()> {
        // 先完成全部校验，避免转换失败后只更新部分字段。
        let source = Self::from_file(file)?;
        self.uuid = source.uuid;
        self.filename = source.filename;
        self.filepath = source.filepath;
        self.key = source.key;
        self.extension = source.extension;
        self.mime_type = source.mime_type;
        self.size = source.size;
        Ok(())
    }
}

// implement your custom finders, selectors oriented logic here
impl Entity {
    /// 根据领域文件 id 构建查询；它对应 uuid 列，而不是数据库自增主键。
    pub fn find_by_uuid(file_id: &str) -> Result<Select<Self>> {
        let uuid =
            Uuid::parse_str(file_id).with_context(|| format!("invalid file id: {file_id}"))?;
        Ok(Self::find().filter(Column::Uuid.eq(uuid)))
    }
}
