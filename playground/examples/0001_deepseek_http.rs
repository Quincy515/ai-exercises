#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let api_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1/chat/completions".to_string());
    let body = serde_json::json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {
                "role": "user",
                "content": "你好，你是？"
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
