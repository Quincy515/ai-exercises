use std::{path::Path, time::Duration};

use anyhow::{Context, Result};
use chromiumoxide::{browser::Browser, page::ScreenshotParams};
use futures_util::StreamExt;
use tokio::{fs, time};

const CDP_URL: &str = "http://localhost:9222";
const TARGET_URL: &str = "https://imooc.com";
const SCREENSHOT_PATH: &str = "resources/screenshot.png";
const FULL_SCREENSHOT_PATH: &str = "resources/screenshot-full.png";

#[tokio::main]
async fn main() -> Result<()> {
    example().await
}

async fn example() -> Result<()> {
    // 1.创建一个playwright异步实例
    // Python: async with async_playwright() as playwright:
    // Rust: chromiumoxide 直接连接 CDP，并启动 handler 处理浏览器事件。
    let (mut browser, mut handler) = Browser::connect(CDP_URL).await.with_context(|| {
        format!("连接 CDP 失败，请确认 Chrome 已用 --remote-debugging-port=9222 启动: {CDP_URL}")
    })?;

    // Python: playwright = await async_playwright().start(); await playwright.stop()
    // Rust: handler_task 对应 chromiumoxide 的异步事件循环，示例结束时 abort 即可断开本进程监听。
    let handler_task = tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if event.is_err() {
                break;
            }
        }
    });

    let result = run_example_steps(&mut browser).await;
    handler_task.abort();
    result
}

async fn run_example_steps(browser: &mut Browser) -> Result<()> {
    // 2.连接到cdp获取浏览器实例
    // Python: browser = await playwright.chromium.connect_over_cdp("http://localhost:9222")
    // Rust: Browser::connect 已经返回 browser，这里刷新 CDP targets 获取当前页面。
    browser
        .fetch_targets()
        .await
        .context("刷新 CDP targets 失败")?;
    time::sleep(Duration::from_millis(100)).await;

    // Python: default_context = browser.contexts[0]
    // Rust: chromiumoxide 直接从 browser 读取 pages。
    let pages = browser.pages().await.context("获取当前页面列表失败")?;

    // 2.获取当前上下文的第一个页面
    // Python: page = default_context.pages[0]
    // Rust: 如果当前浏览器没有页面，创建一个 about:blank 页面方便后续演示。
    let page = match pages.first() {
        Some(page) => page.clone(),
        None => browser
            .new_page("about:blank")
            .await
            .context("创建默认空白页失败")?,
    };

    println!("页面标题: {}", page.get_title().await?.unwrap_or_default());
    println!("页面URL: {}", page.url().await?.unwrap_or_default());

    // 3.新增页面并且跳转到imooc.com
    // Python: page = await default_context.new_page(); await page.goto("https://imooc.com")
    // Rust: 新建 about:blank 页后调用 goto，保留 Python 的两步对应关系。
    let page = browser
        .new_page("about:blank")
        .await
        .context("创建新页面失败")?;
    page.goto(TARGET_URL)
        .await
        .with_context(|| format!("跳转到 {TARGET_URL} 失败"))?;

    // 4.在页面上执行js并获取结果
    // Python: href = await page.evaluate('() => document.location.href')
    let href: String = page
        .evaluate("() => document.location.href")
        .await?
        .into_value()
        .context("解析 JS 执行结果失败")?;
    println!("js执行结果: {href}");

    // 5.截图
    // Python: await page.screenshot(path="resources/screenshot.png")
    fs::create_dir_all(Path::new("resources"))
        .await
        .context("创建 resources 目录失败")?;
    page.save_screenshot(ScreenshotParams::builder().build(), SCREENSHOT_PATH)
        .await
        .with_context(|| format!("保存截图失败: {SCREENSHOT_PATH}"))?;

    // Python: await page.screenshot(path="resources/screenshot-full.png", full_page=True)
    page.save_screenshot(
        ScreenshotParams::builder().full_page(true).build(),
        FULL_SCREENSHOT_PATH,
    )
    .await
    .with_context(|| format!("保存完整页面截图失败: {FULL_SCREENSHOT_PATH}"))?;

    Ok(())
}
