use async_openai::{Client, types::responses::CreateResponseArgs};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let client = Client::new();

    let request = CreateResponseArgs::default()
        .model("gpt-5.4-mini")
        .input("你好，你是?")
        .build()?;

    let response = client.responses().create(request).await?;

    println!("{response:#?}");
    Ok(())
}
