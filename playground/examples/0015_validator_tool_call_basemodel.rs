use anyhow::{Context, ensure};
use async_openai::{
    Client,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestUserMessage, ChatCompletionTool,
        ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse, FunctionObjectArgs,
    },
};
use derive_name::Name;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use validator::Validate;

const MODEL: &str = "gpt-5.4-mini";

/// 传递用户信息进行数据提取和处理,涵盖Name、Age、Email。
#[derive(Deserialize, Validate, JsonSchema, Name, Debug)]
#[serde(deny_unknown_fields)]
struct UserInfo {
    /// 用户名称
    #[validate(length(min = 1))]
    name: String,
    /// 用户年龄，必须是正整数
    #[validate(range(min = 1))]
    age: u16,
    /// 用户的电子邮件
    #[validate(email)]
    email: String,
}

// Rust 里可以用 trait 做出类似 Pydantic BaseModel 的公共能力。
// 只要一个 struct 同时支持 Name、JsonSchema、Deserialize、Validate，
// 就可以获得 model_json_schema、model_validate_json、tool 等默认方法。
trait BaseModel: Sized + Name + JsonSchema + DeserializeOwned + Validate {
    fn description() -> &'static str;

    /// schemars 根据 struct 自动生成 schema，类似 Pydantic 的 model_json_schema()。
    fn model_json_schema() -> anyhow::Result<Value> {
        let mut schema = serde_json::to_value(schema_for!(Self))?;
        schema
            .as_object_mut()
            .context("schema 必须是 JSON object")?
            .remove("$schema");
        Ok(schema)
    }

    /// 把 struct 转成 OpenAI tool 定义。
    fn tool() -> anyhow::Result<ChatCompletionTool> {
        Ok(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                // 这里的 name 是发给模型看的工具名。
                // derive_name 让 Self::name() 返回字符串 "UserInfo"，类似 Python 的 UserInfo.__name__。
                .name(Self::name())
                // description 告诉模型这个工具用来提取什么信息。
                .description(Self::description())
                // parameters 是 JSON Schema，告诉模型 arguments 必须长什么样。
                .parameters(Self::model_json_schema()?)
                .strict(true)
                .build()?,
        })
    }

    /// 把模型返回的 JSON 字符串解析成当前 struct，并执行 validator 校验。
    fn model_validate_json(arguments: &str) -> anyhow::Result<Self> {
        let value: Self = serde_json::from_str(arguments)
            .with_context(|| format!("工具参数不是有效 JSON: {arguments}"))?;
        value.validate()?;
        Ok(value)
    }

    /// 从模型返回的 tool_call 中取出 JSON 参数，并转成当前 struct。
    fn from_response(response: &CreateChatCompletionResponse) -> anyhow::Result<Self> {
        // Chat Completions 可能返回多个 choice；本示例只请求默认的 1 个，所以取第一个。
        // tool_calls 是 Option，因为模型也可能直接返回文本内容。
        // first() 返回 Option，可以把“没有返回数据”的情况转成清晰错误。
        let tool_call = response
            .choices
            .first()
            .and_then(|choice| choice.message.tool_calls.as_ref()?.first())
            .context("模型没有返回工具调用")?;

        // async-openai 把不同工具调用建模成 enum。
        // 这里把 ChatCompletionMessageToolCalls::Function 里的数据解出来。
        // 这一步只是在本地检查和解包，没有再次调用模型。
        let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
            anyhow::bail!("模型返回了非函数工具调用");
        };

        // tool_choice 已经要求模型返回名为 UserInfo 的 tool_call，但这里再校验一次返回结果。
        // 这样上游返回异常、模型兼容层行为变化、或未来添加多个工具时，错误会更明确。
        ensure!(
            tool_call.function.name == Self::name(),
            "模型调用了未知工具: {}",
            tool_call.function.name
        );

        // function.arguments 是模型返回的 JSON 字符串。
        // 这里才真正得到一个当前 struct 的实例。
        Self::model_validate_json(&tool_call.function.arguments)
    }
}

impl BaseModel for UserInfo {
    fn description() -> &'static str {
        "传递用户的信息进行数据提取和处理，涵盖 name、age、email。"
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .model(MODEL)
        .messages(vec![
            ChatCompletionRequestUserMessage::from(
                "我叫 Custer，今年 34 岁，我的联系方式是 custertian@gmail.com",
            )
            .into(),
        ])
        // 注册可用工具：把 UserInfo 的名字、描述、参数 schema 放进请求。
        // 这一步只是告诉模型“可以按 UserInfo 这个格式返回数据”。
        .tools(UserInfo::tool()?)
        // 强制模型选择名为 "UserInfo" 的工具。
        // 模型会返回 tool_call，其中 function.arguments 是按 schema 生成的 JSON 字符串。
        .tool_choice(ChatCompletionToolChoiceOption::Function(
            UserInfo::name().into(),
        ))
        .build()?;

    let response = client.chat().create(request).await?;
    // 从模型返回的 tool_call.arguments 中解析并校验出 UserInfo。
    let user_info = UserInfo::from_response(&response)?;

    println!("{user_info:#?}");

    Ok(())
}
