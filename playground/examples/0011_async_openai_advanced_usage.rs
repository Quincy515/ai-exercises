use async_openai::{
    Client,
    types::chat::{ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs},
};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct AppState {
    client: Client<async_openai::config::OpenAIConfig>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    reply: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let state = AppState {
        client: Client::new(),
    };
    let app = Router::new().route("/chat", post(chat)).with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server listening on http://0.0.0.0:3000");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    let message = payload.message.trim();
    if message.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "message must not be empty",
        ));
    }

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-5.4-mini")
        .messages([ChatCompletionRequestUserMessage::from(message).into()])
        .build()
        .map_err(|error| error_response(StatusCode::BAD_REQUEST, error.to_string()))?;

    let response = state
        .client
        .chat()
        .create(request)
        .await
        .map_err(|error| error_response(StatusCode::BAD_GATEWAY, error.to_string()))?;

    let reply = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .unwrap_or_default();

    Ok(Json(ChatResponse { reply }))
}

fn error_response(
    status: StatusCode,
    error: impl Into<String>,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: error.into(),
        }),
    )
}
