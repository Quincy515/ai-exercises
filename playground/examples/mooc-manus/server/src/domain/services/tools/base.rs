use std::collections::HashSet;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::domain::models::ToolResult;

// Manus 工具设计思路：
// Manus tool design:
// 1. 所有工具集都实现 BaseTool，用统一的 invoke 方法调用工具。
//    Every tool collection implements BaseTool and uses one invoke method.
// 2. Rust 用 ToolDefinition 承载 Python 装饰器写入方法上的工具元数据。
//    ToolDefinition stores the metadata attached by the Python decorator.
// 3. 工具集可以通过 get_tools 获取 schema 参数信息，方便 LLM 调用。
//    Tool collections expose schemas through get_tools for LLM binding.
// 4. 调用工具前过滤模型幻觉出的多余参数。
//    Extra hallucinated model arguments are filtered before dispatch.
//
// Python 使用示例：
// Python usage example:
// @tool("demo", "这是一个 demo", {"name": {"type": "string"}}, ["name"])
// async def demo(name: str) -> ToolResult[str]:
//     return ToolResult(data=f"hello {name}")
//
// Rust 使用示例：
// Rust usage example:
// struct DemoTool {
//     name: String,
//     tools: Vec<ToolDefinition>,
// }
//
// impl DemoTool {
//     fn new() -> Self {
//         let mut parameters = Map::new();
//         parameters.insert("name".to_string(), json!({"type": "string"}));
//
//         Self {
//             name: "demo_tool".to_string(),
//             tools: vec![tool(
//                 "demo",
//                 "这是一个 demo",
//                 parameters,
//                 vec!["name".to_string()],
//             )],
//         }
//     }
//
//     async fn demo(&self, name: String) -> Result<ToolResult<Value>> {
//         Ok(ToolResult {
//             data: Some(Value::String(format!("hello {name}"))),
//             ..ToolResult::default()
//         })
//     }
// }
//
// #[async_trait]
// impl BaseTool for DemoTool {
//     fn name(&self) -> &str {
//         &self.name
//     }
//
//     fn tool_definitions(&self) -> &[ToolDefinition] {
//         &self.tools
//     }
//
//     async fn call_tool(
//         &self,
//         tool_name: &str,
//         kwargs: ToolArguments,
//     ) -> Result<ToolResult<Value>> {
//         match tool_name {
//             "demo" => {
//                 let name = kwargs
//                     .get("name")
//                     .and_then(Value::as_str)
//                     .ok_or_else(|| anyhow!("工具参数[name]缺失"))?;
//                 self.demo(name.to_string()).await
//             }
//             _ => Err(anyhow!("工具[{tool_name}]未找到")),
//         }
//     }
// }

pub type ToolParameters = Map<String, Value>;
pub type ToolArguments = Map<String, Value>;
pub type ToolSchema = Map<String, Value>;

/// 单个工具的声明信息，等价于 Python 装饰器写入方法上的元数据。
/// Metadata for one tool, equivalent to attributes attached by the Python decorator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    /// 工具名字：调用工具时用来匹配工具名
    pub name: String,
    /// 工具描述：给 LLM 看，告诉模型这个工具做什么
    pub description: String,
    /// 工具参数：定义工具的输入参数
    pub parameters: ToolParameters,
    /// 必填参数列表：工具调用时必须提供的参数
    pub required: Vec<String>,
    /// 传给 OpenAI tools 参数的完整工具声明
    pub schema: ToolSchema,
}

impl ToolDefinition {
    /// 创建工具定义，并提前生成 OpenAI tool schema。
    /// Create a tool definition and pre-build its OpenAI tool schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: ToolParameters,
        required: Vec<String>,
    ) -> Self {
        // 1. 接收工具名字和描述，对齐 Python 装饰器参数 name / description。
        //    Receive name and description, matching the Python decorator arguments.
        let name = name.into();
        let description = description.into();
        // 2. 创建工具声明数据结构，对应 Python 里的 tool_schema。
        //    Build the tool_schema object used by OpenAI.
        let schema = build_tool_schema(&name, &description, parameters.clone(), required.clone());

        // 3. Rust 用结构体字段保存元数据，对应 Python 的 func._tool_name / _tool_description / _tool_schema。
        //    Store metadata in struct fields instead of Python runtime function attributes.
        Self {
            name,
            description,
            parameters,
            required,
            schema,
        }
    }

    /// 过滤模型幻觉出的多余参数。
    /// Filter arguments hallucinated by the model.
    pub fn filter_parameters(&self, kwargs: ToolArguments) -> ToolArguments {
        // 1. 获取工具允许的参数名；Python 这里使用 inspect.signature(method)。
        //    Collect allowed parameter names; Python uses inspect.signature(method).
        let allowed_parameters = self.parameters.keys().collect::<HashSet<_>>();

        // 2. 遍历 LLM 传入的 kwargs，只保留工具 schema 中声明过的参数。
        //    Iterate through model-provided kwargs and keep only schema-declared parameters.
        kwargs
            .into_iter()
            .filter(|(key, _)| allowed_parameters.contains(key))
            .collect()
    }
}

/// 定义 OpenAI 工具声明(装饰器)，用于将一个函数 / 方法，添加上对应的工具声明。
/// Define an OpenAI tool declaration, replacing the Python decorator.
pub fn tool(
    name: impl Into<String>,
    description: impl Into<String>,
    parameters: ToolParameters,
    required: Vec<String>,
) -> ToolDefinition {
    // 1. Python 的 tool(...) 返回 decorator；Rust 这里直接返回工具元数据。
    //    Python tool(...) returns a decorator; Rust returns metadata directly.
    // 2. 具体工具集把返回的 ToolDefinition 放进 Vec<ToolDefinition>。
    //    Concrete tool collections store the returned ToolDefinition in Vec<ToolDefinition>.
    ToolDefinition::new(name, description, parameters, required)
}

/// 创建工具声明 tool_schema，生成 OpenAI 需要的工具格式。
/// Build the OpenAI-compatible tool_schema object.
fn build_tool_schema(
    name: &str,
    description: &str,
    parameters: ToolParameters,
    required: Vec<String>,
) -> ToolSchema {
    // 1. 创建 function 对象，对应 Python tool_schema["function"]。
    //    Create the function object, matching Python tool_schema["function"].
    let mut function = Map::new();
    // 2. 写入工具名字和描述。
    //    Fill the tool name and description.
    function.insert("name".to_string(), Value::String(name.to_string()));
    function.insert(
        "description".to_string(),
        Value::String(description.to_string()),
    );
    // 3. 写入 parameters，包含 type / properties / required。
    //    Fill parameters with type / properties / required.
    function.insert(
        "parameters".to_string(),
        json!({
            "type": "object",
            "properties": parameters,
            "required": required,
        }),
    );

    // 4. 外层包一层 {"type": "function", "function": ...}，这是 OpenAI tools 需要的格式。
    //    Wrap with {"type": "function", "function": ...}, the shape expected by OpenAI tools.
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String("function".to_string()));
    schema.insert("function".to_string(), Value::Object(function));
    schema
}

/// 基础工具协议，用于管理一组可被 LLM 调用的工具。
/// Base tool protocol for managing a set of LLM-callable tools.
#[async_trait]
pub trait BaseTool: Send + Sync {
    /// 工具集名字。
    /// Tool collection name.
    fn name(&self) -> &str {
        ""
    }

    /// 返回当前工具集的工具声明。
    /// Return tool declarations for this tool collection.
    fn tool_definitions(&self) -> &[ToolDefinition];

    /// 调用已过滤参数后的具体工具。
    /// Dispatch the concrete tool with filtered arguments.
    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>>;

    /// 获取所有已注册工具的 schema 信息，用于绑定给 LLM。
    /// Return registered tool schemas for LLM binding.
    fn get_tools(&self) -> Vec<ToolSchema> {
        // 1. Rust 的 ToolDefinition 已经是预先注册好的工具缓存，对应 Python 的 _tools_cache。
        //    Rust ToolDefinition values are pre-registered tool cache, matching Python _tools_cache.
        // 2. 遍历所有工具定义。
        //    Iterate through all tool definitions.
        // 3. 取出每个工具的 schema，返回给 LLM 绑定 tools。
        //    Return each tool schema for LLM tool binding.
        self.tool_definitions()
            .iter()
            .map(|tool| tool.schema.clone())
            .collect()
    }

    /// 判断工具集下是否存在指定工具。
    /// Return whether this tool collection has the named tool.
    fn has_tool(&self, tool_name: &str) -> bool {
        // 1. 遍历当前工具集下所有 ToolDefinition；Python 这里遍历 inspect.getmembers。
        //    Iterate through ToolDefinition values; Python iterates inspect.getmembers.
        // 2. 判断工具名是否和传入 tool_name 一致。
        //    Check whether any registered tool name matches tool_name.
        self.tool_definitions()
            .iter()
            .any(|tool| tool.name == tool_name)
    }

    /// 根据工具名和参数调用指定工具。
    /// Invoke a tool by name with model-provided arguments.
    async fn invoke(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        // 1. 遍历工具集的所有工具定义；Python 这里遍历 inspect.getmembers(self, inspect.ismethod)。
        //    Iterate through registered tool definitions; Python scans bound methods.
        // 2. 判断是否存在匹配的工具名；Python 这里检查 method._tool_name。
        //    Find the definition whose name matches tool_name; Python checks method._tool_name.
        let definition = self
            .tool_definitions()
            .iter()
            .find(|tool| tool.name == tool_name)
            // 5. 如果循环结束还没有找到工具，则返回错误。
            //    Return an error when no registered tool matches tool_name.
            .ok_or_else(|| anyhow!("工具[{tool_name}]未找到"))?;

        // 3. 筛选传递的 kwargs，只保留工具 schema 中声明的参数，多余的剔除。
        //    Filter kwargs and remove hallucinated parameters not declared by the tool schema.
        let filtered_kwargs = definition.filter_parameters(kwargs);

        // 4. 调用具体工具实现并获取工具结果；Python 这里执行 await method(**filtered_kwargs)。
        //    Dispatch the concrete tool and return its result; Python awaits method(**filtered_kwargs).
        self.call_tool(tool_name, filtered_kwargs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool {
        tools: Vec<ToolDefinition>,
    }

    impl EchoTool {
        fn new() -> Self {
            let mut parameters = Map::new();
            parameters.insert(
                "text".to_string(),
                json!({
                    "type": "string",
                    "description": "需要回显的文本"
                }),
            );

            Self {
                tools: vec![tool(
                    "echo",
                    "回显传入文本",
                    parameters,
                    vec!["text".to_string()],
                )],
            }
        }
    }

    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn tool_definitions(&self) -> &[ToolDefinition] {
            &self.tools
        }

        async fn call_tool(
            &self,
            tool_name: &str,
            kwargs: ToolArguments,
        ) -> Result<ToolResult<Value>> {
            match tool_name {
                "echo" => Ok(ToolResult {
                    data: Some(Value::Object(kwargs)),
                    ..ToolResult::default()
                }),
                _ => Err(anyhow!("工具[{tool_name}]未找到")),
            }
        }
    }

    #[test]
    fn builds_openai_tool_schema() {
        let tools = EchoTool::new().get_tools();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "echo");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["text"]["type"],
            "string"
        );
        assert_eq!(tools[0]["function"]["parameters"]["required"][0], "text");
    }

    #[test]
    fn checks_tool_existence() {
        let tool = EchoTool::new();

        assert!(tool.has_tool("echo"));
        assert!(!tool.has_tool("missing"));
    }

    #[tokio::test]
    async fn invoke_filters_hallucinated_arguments() {
        let tool = EchoTool::new();
        let mut kwargs = Map::new();
        kwargs.insert("text".to_string(), Value::String("hello".to_string()));
        kwargs.insert("extra".to_string(), Value::String("ignored".to_string()));

        let result = tool.invoke("echo", kwargs).await.unwrap();
        let data = result.data.unwrap();

        assert_eq!(data["text"], "hello");
        assert!(data.get("extra").is_none());
    }

    #[tokio::test]
    async fn invoke_returns_error_for_missing_tool() {
        let error = EchoTool::new()
            .invoke("missing", Map::new())
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "工具[missing]未找到");
    }
}
