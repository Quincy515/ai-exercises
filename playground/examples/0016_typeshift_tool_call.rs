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
use serde_json::Value;
use typeshift::typeshift;

const MODEL: &str = "gpt-5.4-mini";

/// 传递用户信息进行数据提取和处理，涵盖 name、age、email。
#[typeshift]
#[derive(Name, Debug)]
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
        // UserInfo::tool() 把 struct 转成 async-openai 需要的 tool 定义。
        // typeshift 负责生成 JSON Schema，省掉手写 schemars/validator 模板代码。
        .tools(UserInfo::tool()?)
        // 强制模型按 UserInfo 这个 schema 返回 tool_call.arguments。
        .tool_choice(ChatCompletionToolChoiceOption::Function(
            UserInfo::name().into(),
        ))
        .build()?;

    let response = client.chat().create(request).await?;
    let user_info = UserInfo::from_response(&response)?;

    println!("{user_info:#?}");

    Ok(())
}

impl UserInfo {
    fn description() -> &'static str {
        "传递用户的信息进行数据提取和处理，涵盖 name、age、email。"
    }

    /// typeshift::schema_json::<T>() 生成 JSON Schema，类似 Pydantic 的 model_json_schema()。
    fn model_json_schema() -> Value {
        let mut schema = typeshift::schema_json::<Self>();
        if let Some(object) = schema.as_object_mut() {
            object.remove("$schema");
        }
        schema
    }

    /// 只保留 async-openai 的 tool 包装逻辑；schema 本身交给 typeshift。
    fn tool() -> anyhow::Result<ChatCompletionTool> {
        Ok(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name(Self::name())
                .description(Self::description())
                .parameters(Self::model_json_schema())
                .strict(true)
                .build()?,
        })
    }

    /// 从 async-openai 响应中取出 tool_call.arguments，再交给 typeshift 解析和校验。
    fn from_response(response: &CreateChatCompletionResponse) -> anyhow::Result<Self> {
        let tool_call = response
            .choices
            .first()
            .and_then(|choice| choice.message.tool_calls.as_ref()?.first())
            .context("模型没有返回工具调用")?;

        let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
            anyhow::bail!("模型返回了非函数工具调用");
        };

        ensure!(
            tool_call.function.name == Self::name(),
            "模型调用了未知工具: {}",
            tool_call.function.name
        );

        Ok(typeshift::parse_str(&tool_call.function.arguments)?)
    }
}
