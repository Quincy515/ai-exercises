use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::domain::models::ToolResult;

use super::{arguments::required_str, tool, BaseTool, ToolArguments, ToolDefinition};

/// 消息工具，用于完成消息工具包初始化。
pub struct MessageTool {
    name: String,
    definitions: Vec<ToolDefinition>,
}

impl MessageTool {
    /// 构造函数，完成消息工具包初始化。
    pub fn new() -> Self {
        Self {
            name: "message".to_string(),
            definitions: vec![
                tool(
                    "message_notify_user",
                    "向用户发送消息，且无需用户回复。用于确认收到消息、提供进度更新、报告任务完成情况，或解释处理方式的变更。",
                    ToolArguments::from_iter([(
                        "text".to_string(),
                        json!({
                            "type": "string",
                            "description": "要显示给用户的消息文本"
                        }),
                    )]),
                    vec!["text".to_string()],
                ),
                tool(
                    "message_ask_user",
                    "向用户提问并等待回复。用于：请求澄清、寻求确认、或收集额外信息。",
                    ToolArguments::from_iter([
                        (
                            "text".to_string(),
                            json!({
                                "type": "string",
                                "description": "要展示给用户的问题文本"
                            }),
                        ),
                        (
                            "attachments".to_string(),
                            json!({
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                ],
                                "description": "(可选)与问题相关的文件或参考资料"
                            }),
                        ),
                        (
                            "suggest_user_takeover".to_string(),
                            json!({
                                "type": "string",
                                "enum": ["none", "browser"],
                                "description": "(可选)建议用户接管的操作（例如由用户在浏览器中手动完成某些事）。"
                            }),
                        ),
                    ]),
                    vec!["text".to_string()],
                ),
            ],
        }
    }

    /// 发送通知消息给用户，不需要用户响应。
    async fn message_notify_user(&self, _text: &str) -> ToolResult<Value> {
        ToolResult {
            data: Some(Value::String("Continue".to_string())),
            ..ToolResult::default()
        }
    }

    /// 提问用户并等待响应。
    async fn message_ask_user(
        &self,
        _text: &str,
        _attachments: Option<&Value>,
        _suggest_user_takeover: Option<&str>,
    ) -> ToolResult<Value> {
        ToolResult::default()
    }
}

impl Default for MessageTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseTool for MessageTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        match tool_name {
            "message_notify_user" => {
                let text = required_str(&kwargs, "text")?;
                Ok(self.message_notify_user(text).await)
            }
            "message_ask_user" => {
                let text = required_str(&kwargs, "text")?;
                let attachments = kwargs.get("attachments");
                validate_attachments(attachments)?;
                let suggest_user_takeover = optional_user_takeover(&kwargs)?;

                Ok(self
                    .message_ask_user(text, attachments, suggest_user_takeover)
                    .await)
            }
            _ => Err(anyhow!("工具[{tool_name}]未找到")),
        }
    }
}

fn validate_attachments(attachments: Option<&Value>) -> Result<()> {
    match attachments {
        None | Some(Value::Null | Value::String(_)) => Ok(()),
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => Ok(()),
        Some(_) => Err(anyhow!("工具参数[attachments]必须是字符串或字符串数组")),
    }
}

fn optional_user_takeover(kwargs: &ToolArguments) -> Result<Option<&str>> {
    match kwargs.get("suggest_user_takeover") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if matches!(value.as_str(), "none" | "browser") => {
            Ok(Some(value))
        }
        Some(Value::String(_)) => Err(anyhow!(
            "工具参数[suggest_user_takeover]必须是 none 或 browser"
        )),
        Some(_) => Err(anyhow!("工具参数[suggest_user_takeover]必须是字符串")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;

    #[test]
    fn exposes_message_tool_schemas() {
        let tool = MessageTool::new();

        assert_eq!(tool.name(), "message");
        assert!(tool.has_tool("message_notify_user"));
        assert!(tool.has_tool("message_ask_user"));

        let schemas = tool.get_tools();
        assert_eq!(schemas.len(), 2);

        let notify = &schemas[0]["function"];
        assert_eq!(notify["name"], "message_notify_user");
        assert_eq!(notify["parameters"]["required"], json!(["text"]));

        let ask = &schemas[1]["function"];
        assert_eq!(ask["name"], "message_ask_user");
        assert_eq!(ask["parameters"]["required"], json!(["text"]));
        assert_eq!(
            ask["parameters"]["properties"]["attachments"]["anyOf"],
            json!([
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ])
        );
        assert_eq!(
            ask["parameters"]["properties"]["attachments"]["description"],
            "(可选)与问题相关的文件或参考资料"
        );
        assert_eq!(
            ask["parameters"]["properties"]["suggest_user_takeover"]["enum"],
            json!(["none", "browser"])
        );
    }

    #[tokio::test]
    async fn notifies_user_and_filters_extra_arguments() {
        let result = MessageTool::new()
            .invoke(
                "message_notify_user",
                Map::from_iter([
                    ("text".to_string(), json!("正在整理数据")),
                    ("extra".to_string(), json!("ignored")),
                ]),
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.data, Some(json!("Continue")));
    }

    #[tokio::test]
    async fn asks_user_with_optional_arguments() {
        let tool = MessageTool::new();

        for (attachments, suggest_user_takeover) in [
            (json!("/tmp/report.md"), json!("none")),
            (json!(["/tmp/report.md", "/tmp/data.csv"]), json!("browser")),
        ] {
            let result = tool
                .invoke(
                    "message_ask_user",
                    Map::from_iter([
                        ("text".to_string(), json!("请确认生成的文件")),
                        ("attachments".to_string(), attachments),
                        ("suggest_user_takeover".to_string(), suggest_user_takeover),
                    ]),
                )
                .await
                .unwrap();

            assert!(result.success);
            assert_eq!(result.data, None);
        }

        let result = tool
            .invoke(
                "message_ask_user",
                Map::from_iter([("text".to_string(), json!("是否继续？"))]),
            )
            .await
            .unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn rejects_invalid_message_arguments() {
        let tool = MessageTool::new();

        let missing_text = tool
            .invoke("message_notify_user", Map::new())
            .await
            .unwrap_err();
        assert_eq!(missing_text.to_string(), "工具参数[text]缺失");

        let invalid_attachments = tool
            .invoke(
                "message_ask_user",
                Map::from_iter([
                    ("text".to_string(), json!("请确认")),
                    ("attachments".to_string(), json!(["report.md", 1])),
                ]),
            )
            .await
            .unwrap_err();
        assert_eq!(
            invalid_attachments.to_string(),
            "工具参数[attachments]必须是字符串或字符串数组"
        );

        let invalid_takeover = tool
            .invoke(
                "message_ask_user",
                Map::from_iter([
                    ("text".to_string(), json!("请接管操作")),
                    ("suggest_user_takeover".to_string(), json!("desktop")),
                ]),
            )
            .await
            .unwrap_err();
        assert_eq!(
            invalid_takeover.to_string(),
            "工具参数[suggest_user_takeover]必须是 none 或 browser"
        );
    }
}
