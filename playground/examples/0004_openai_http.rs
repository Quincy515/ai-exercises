use base64::{Engine as _, engine::general_purpose::STANDARD};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let api_url = std::env::var("OPENAI_API_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());

    // Read local image and encode as base64 data URL
    let image_path = "playground/examples/assets/广州塔.jpeg";
    let image_bytes = std::fs::read(image_path)?;
    let image_url = format!("data:image/jpeg;base64,{}", STANDARD.encode(&image_bytes));

    let body = serde_json::json!({
        "model": "gpt-5.4-mini",
        "messages": [
            {
                "role": "user",
                "content": [ {"type": "text", "text": "请描述下这张图片，这张图片所在位置是哪里 呢?"}, {"type": "image_url", "image_url": {"url": image_url}} ]
            }
        ],
        "stream": false
    });
    let response = reqwest::Client::new()
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await?;
    let text = response.text().await?;
    let body: serde_json::Value = serde_json::from_str(&text).unwrap();
    println!("{:#?}", body);
    Ok(())
}
