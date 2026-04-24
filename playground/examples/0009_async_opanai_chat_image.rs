use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageDetail,
        ImageUrl,
    },
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let image_path = "playground/examples/assets/广州塔.jpeg";
    let image_bytes = std::fs::read(image_path)?;
    let image_url = format!("data:image/jpeg;base64,{}", STANDARD.encode(&image_bytes));

    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-5.4-mini")
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(vec![
                ChatCompletionRequestMessageContentPartText::from(
                    "请描述下这张图片，这张图片所在位置是哪里呢?",
                )
                .into(),
                ChatCompletionRequestMessageContentPartImage::from(ImageUrl {
                    url: image_url,
                    detail: Some(ImageDetail::High),
                })
                .into(),
            ])
            .build()?
            .into()])
        .build()?;

    let response = client.chat().create(request).await?;

    println!("{response:#?}");
    Ok(())
}
