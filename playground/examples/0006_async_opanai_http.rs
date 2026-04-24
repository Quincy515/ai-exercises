use async_openai::{
    Client,
    types::chat::{ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-5.4-mini")
        .messages(vec![
            ChatCompletionRequestUserMessage::from("你好，你是?").into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;

    println!("{response:#?}");
    Ok(())
}
