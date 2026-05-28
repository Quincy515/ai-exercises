use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type Message = Map<String, Value>;

/// 记忆类，定义 Agent 的记忆基础信息
/// Memory stores the base message list for an Agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    pub messages: Vec<Message>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    /// 根据传递的消息获取消息的角色信息
    /// Returns the role from a message.
    pub fn get_message_role(message: &Message) -> Option<&str> {
        message.get("role").and_then(Value::as_str)
    }

    /// 往记忆中添加一条消息
    /// Adds one message into memory.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// 往记忆中添加多条消息
    /// Adds multiple messages into memory.
    pub fn add_messages(&mut self, messages: Vec<Message>) {
        self.messages.extend(messages);
    }

    /// 获取记忆中的所有消息列表
    /// Returns all messages in memory.
    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    /// 获取记忆中的最后一条消息
    /// Returns the last message in memory.
    pub fn get_last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// 回滚记忆，删除最后一条消息
    /// Rolls memory back by removing the last message.
    pub fn roll_back(&mut self) {
        self.messages.pop();
    }

    /// 记忆压缩，移除已经执行工具的大内容和推理内容
    /// Compacts memory by removing executed browser tool results and reasoning content.
    pub fn compact(&mut self) {
        for message in &mut self.messages {
            if Self::get_message_role(message) == Some("tool") {
                // 移除浏览器工具的大内容和推理内容
                // remove browser tool results and reasoning content
                let removable_tool = message
                    .get("function_name")
                    .and_then(Value::as_str)
                    .filter(|function_name| {
                        matches!(*function_name, "browser_view" | "browser_navigate")
                    })
                    .map(str::to_owned);

                if let Some(function_name) = removable_tool {
                    message.insert(
                        "content".to_string(),
                        Value::String("(removed)".to_string()),
                    );
                    tracing::debug!(
                        function_name = %function_name,
                        "从记忆中移除对应工具的结果 / removed tool result from memory"
                    );
                }
            }

            if let Some(reasoning_content) = message.remove("reasoning_content") {
                let preview = reasoning_preview(&reasoning_content);
                tracing::debug!(
                    reasoning_content = %preview,
                    "从记忆中移除工具思考结果 / removed reasoning content from memory"
                );
            }
        }
    }

    /// 只读属性，检查记忆是否为空
    /// Returns whether memory has no messages.
    pub fn empty(&self) -> bool {
        self.messages.is_empty()
    }
}

fn reasoning_preview(reasoning_content: &Value) -> String {
    reasoning_content
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| reasoning_content.to_string())
        .chars()
        .take(50)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Memory, Message};
    use serde_json::{json, Value};

    fn message(value: Value) -> Message {
        match value {
            Value::Object(message) => message,
            _ => unreachable!("test messages must be JSON objects"),
        }
    }

    #[test]
    fn manages_messages() {
        let mut memory = Memory::new();
        assert!(memory.empty());

        let first = message(json!({
            "role": "user",
            "content": "hello"
        }));
        assert_eq!(Memory::get_message_role(&first), Some("user"));

        memory.add_message(first);
        memory.add_messages(vec![
            message(json!({
                "role": "tool",
                "content": "tool result"
            })),
            message(json!({
                "role": "assistant",
                "content": "answer"
            })),
        ]);

        assert_eq!(memory.get_messages().len(), 3);
        assert_eq!(
            memory.get_last_message().and_then(Memory::get_message_role),
            Some("assistant")
        );

        memory.roll_back();

        assert_eq!(memory.get_messages().len(), 2);
        assert_eq!(
            memory.get_last_message().and_then(Memory::get_message_role),
            Some("tool")
        );
    }

    #[test]
    fn compact_removes_browser_results_and_reasoning_content() {
        let mut memory = Memory {
            messages: vec![
                message(json!({
                    "role": "tool",
                    "function_name": "browser_view",
                    "content": "large browser result",
                    "reasoning_content": "hidden reasoning"
                })),
                message(json!({
                    "role": "tool",
                    "function_name": "search",
                    "content": "kept result",
                    "reasoning_content": {
                        "hidden": true
                    }
                })),
                message(json!({
                    "role": "assistant",
                    "content": "kept answer",
                    "reasoning_content": "hidden assistant reasoning"
                })),
            ],
        };

        memory.compact();

        let messages = memory.get_messages();
        assert_eq!(
            messages[0].get("content"),
            Some(&Value::String("(removed)".to_string()))
        );
        assert!(!messages[0].contains_key("reasoning_content"));

        assert_eq!(
            messages[1].get("content"),
            Some(&Value::String("kept result".to_string()))
        );
        assert!(!messages[1].contains_key("reasoning_content"));

        assert_eq!(
            messages[2].get("content"),
            Some(&Value::String("kept answer".to_string()))
        );
        assert!(!messages[2].contains_key("reasoning_content"));
    }
}
