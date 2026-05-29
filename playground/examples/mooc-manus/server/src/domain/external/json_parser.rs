use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// JSON 解析器，用于解析 JSON 字符串并尝试修复。
/// JSON parser that parses and repairs JSON text.
#[async_trait]
pub trait JsonParser: Send + Sync {
    /// 解析文本；解析/修复失败且提供默认值时返回默认值。
    /// Parse text; return the default value when parsing/repairing fails and a default is provided.
    async fn invoke(&self, text: &str, default_value: Option<Value>) -> Result<Value>;
}
