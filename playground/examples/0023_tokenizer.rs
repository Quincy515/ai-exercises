use llm_tokenizer::{
    TokenizerTrait, chat_template::ChatTemplateParams, create_tokenizer_from_file,
};
use serde_json::json;

fn main() -> Result<(), anyhow::Error> {
    let tokenizer_dir = format!("{}/examples/assets/tokenizer", env!("CARGO_MANIFEST_DIR"));
    let tokenizer = create_tokenizer_from_file(&tokenizer_dir)?;

    let prompt = "你好，你是?";
    let messages = vec![json!({"role": "user", "content": "帮我计算下45243*123"})];
    let chat_text = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default())?;

    println!("prompt: {}", token_len(tokenizer.as_ref(), prompt, true)?);
    println!(
        "messages: {}",
        token_len(tokenizer.as_ref(), &chat_text, false)?
    );

    Ok(())
}

fn token_len(
    tokenizer: &dyn TokenizerTrait,
    text: &str,
    add_special: bool,
) -> anyhow::Result<usize> {
    Ok(tokenizer.encode(text, add_special)?.token_ids().len())
}
