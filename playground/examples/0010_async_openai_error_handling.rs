use async_openai::{
    Client,
    error::{ApiError, OpenAIError},
    types::chat::{ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let client = Client::new();
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-5.4-mini")
        .messages([ChatCompletionRequestUserMessage::from("你好，请用一句话介绍你自己").into()])
        .build()?;

    match client.chat().create(request).await {
        Ok(response) => println!("{response:#?}"),
        Err(error) => print_openai_error(error),
    }

    Ok(())
}

fn print_openai_error(error: OpenAIError) {
    match error {
        OpenAIError::ApiError(api_error) => print_api_error(api_error),
        OpenAIError::Reqwest(error) => eprintln!("网络请求失败: {error}"),
        OpenAIError::JSONDeserialize(error, content) => {
            eprintln!("响应 JSON 解析失败: {error}");
            eprintln!("原始响应: {content}");
        }
        OpenAIError::StreamError(error) => eprintln!("流式响应失败: {error}"),
        OpenAIError::InvalidArgument(message) => eprintln!("请求参数无效: {message}"),
        OpenAIError::FileSaveError(message) => eprintln!("文件保存失败: {message}"),
        OpenAIError::FileReadError(message) => eprintln!("文件读取失败: {message}"),
    }
}

fn print_api_error(error: ApiError) {
    let failure = ApiFailure::from(error);
    eprintln!("{failure}");

    if let Some(error) = failure.api_error() {
        eprintln!("message: {}", error.message);
        if let Some(error_type) = &error.r#type {
            eprintln!("type: {error_type}");
        }
        if let Some(param) = &error.param {
            eprintln!("param: {param}");
        }
        if let Some(code) = &error.code {
            eprintln!("code: {code}");
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum ApiFailure {
    #[error("认证失败，请检查 OPENAI_API_KEY 是否正确。")]
    Authentication(ApiError),
    #[error("请求过于频繁，稍后重试或降低并发。")]
    RateLimited(ApiError),
    #[error("额度不足，请检查账户余额或配额。")]
    InsufficientQuota(ApiError),
    #[error("OpenAI API 返回错误: {0}")]
    Other(ApiError),
}

impl ApiFailure {
    fn api_error(&self) -> Option<&ApiError> {
        match self {
            Self::Authentication(error)
            | Self::RateLimited(error)
            | Self::InsufficientQuota(error)
            | Self::Other(error) => Some(error),
        }
    }
}

impl From<ApiError> for ApiFailure {
    fn from(error: ApiError) -> Self {
        let code = error.code.as_deref();
        let error_type = error.r#type.as_deref();

        match (code, error_type) {
            (Some("invalid_api_key"), _) => Self::Authentication(error),
            (Some("rate_limit_exceeded"), _) | (_, Some("rate_limit_exceeded")) => {
                Self::RateLimited(error)
            }
            (Some("insufficient_quota"), _) | (_, Some("insufficient_quota")) => {
                Self::InsufficientQuota(error)
            }
            _ => Self::Other(error),
        }
    }
}
