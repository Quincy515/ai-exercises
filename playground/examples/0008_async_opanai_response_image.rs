use async_openai::{
    Client,
    types::responses::{CreateResponseArgs, InputContent, InputImageArgs, InputMessageArgs},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Read local image and encode as base64 data URL
    let image_path = "playground/examples/assets/广州塔.jpeg";
    let image_bytes = std::fs::read(image_path)?;
    let image_url = format!("data:image/jpeg;base64,{}", STANDARD.encode(&image_bytes));

    let client = Client::new();

    let request = CreateResponseArgs::default()
        .model("gpt-5.4-mini")
        .input(
            InputMessageArgs::default()
                .content(vec![
                    InputContent::from("请描述下这张图片，这张图片所在位置是哪里呢?"),
                    InputImageArgs::default()
                        .image_url(image_url)
                        .build()?
                        .into(),
                ])
                .build()?,
        )
        .build()?;

    let response = client.responses().create(request).await?;

    println!("{response:#?}");
    Ok(())
}
