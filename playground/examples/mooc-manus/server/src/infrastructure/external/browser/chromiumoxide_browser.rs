//! 基于 chromiumoxide 的浏览器 CDP 适配器。
//! Chromiumoxide based browser CDP adapter.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chromiumoxide::{
    browser::{Browser as ChromeBrowser, BrowserConfig},
    layout::Point,
    page::ScreenshotParams,
    Page,
};
use futures::StreamExt;
use scraper::Html;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::domain::{external::Browser, models::ToolResult};

use super::browser_fun::{
    GET_INTERACTIVE_ELEMENTS_FUNC, GET_VISIBLE_CONTENT_FUNC, INJECT_CONSOLE_LOGS_FUNC,
};

const MAX_CONTENT_CHARS: usize = 50_000;
const MAX_INIT_RETRIES: usize = 5;
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const PAGE_LOAD_TIMEOUT: Duration = Duration::from_secs(15);
const PAGE_LOAD_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const BLOCKED_ENV_VARS: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "LLM_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATABASE_URL",
    "REDIS_URL",
];

/// 基础 chromiumoxide 管理的浏览器扩展。
/// Browser adapter managed by chromiumoxide.
pub struct ChromiumoxideBrowser {
    /// 当前浏览器会话。
    /// Current browser session.
    session: Arc<Mutex<Option<BrowserSession>>>,
}

struct BrowserSession {
    browser: ChromeBrowser,
    page: Page,
    handler: Option<tokio::task::JoinHandle<()>>,
    last_used: Instant,
    _temp_dir: tempfile::TempDir,
    interactive_elements_cache: Vec<InteractiveElement>,
}

#[derive(Debug, Clone, Deserialize)]
struct InteractiveElement {
    index: usize,
    tag: String,
    text: String,
    #[serde(default)]
    selector: String,
}

impl ChromiumoxideBrowser {
    /// 构造函数，完成 chromiumoxide 浏览器适配器初始化。
    /// Create a chromiumoxide browser adapter.
    pub fn new() -> Self {
        // 浏览器相关
        Self {
            session: Arc::new(Mutex::new(None)),
        }
    }

    /// 确保浏览器存在，如果不存在则初始化。
    /// Ensure a browser session exists.
    async fn ensure_browser(&self) -> Result<tokio::sync::MutexGuard<'_, Option<BrowserSession>>> {
        let mut guard = self.session.lock().await;

        if guard.as_ref().is_some_and(BrowserSession::is_idle) {
            if let Some(session) = guard.take() {
                session.shutdown().await;
            }
        }

        if guard.is_none() {
            *guard = Some(self.initialize().await?);
        }

        if let Some(session) = guard.as_mut() {
            session.touch();
        }

        Ok(guard)
    }

    /// 确保浏览器页面存在。
    /// Ensure a page exists.
    async fn ensure_page(&self) -> Result<tokio::sync::MutexGuard<'_, Option<BrowserSession>>> {
        // 1.先保证浏览器存在
        self.ensure_browser().await
    }

    /// 初始化并确保资源是可用的。
    /// Initialize and verify browser resources.
    async fn initialize(&self) -> Result<BrowserSession> {
        // 1.定义重试次数+重试延迟确保资源存在
        let mut retry_interval = Duration::from_secs(1);
        let mut last_error = None;

        // 2.循环开始资源构建
        for attempt in 0..MAX_INIT_RETRIES {
            match BrowserSession::launch().await {
                Ok(session) => return Ok(session),
                Err(error) => {
                    // 10.清除所有资源
                    last_error = Some(error.to_string());

                    // 11.判断重试次数是否等于最大重试次数
                    if attempt == MAX_INIT_RETRIES - 1 {
                        break;
                    }

                    // 12.使用指数级增长进行休眠，最大休眠时间为10s
                    tokio::time::sleep(retry_interval).await;
                    retry_interval = (retry_interval * 2).min(Duration::from_secs(10));
                }
            }
        }

        Err(anyhow!(
            "初始化Chromiumoxide浏览器失败(已重试{}次): {}",
            MAX_INIT_RETRIES,
            last_error.unwrap_or_else(|| "未知错误".to_string())
        ))
    }

    async fn cleanup(&self) {
        let session = {
            let mut guard = self.session.lock().await;
            guard.take()
        };

        if let Some(session) = session {
            session.shutdown().await;
        }
    }

    /// 传递超时时间，等待当前页面是否加载完毕。
    /// Wait until document.readyState is complete.
    async fn wait_for_page_load(page: &Page) -> Result<bool> {
        // 2.使用异步任务事件循环中的时间来作为开始时间(只和异步任务相关)
        let started = tokio::time::Instant::now();

        // 3.循环检测网页是否加载成功
        while started.elapsed() < PAGE_LOAD_TIMEOUT {
            // 4.使用js代码判断网页是否加载成功
            let is_completed = page
                .evaluate("() => document.readyState === 'complete'")
                .await?
                .into_value::<bool>()
                .unwrap_or(false);

            if is_completed {
                return Ok(true);
            }

            // 5.未加载成功则休眠对应时间
            tokio::time::sleep(PAGE_LOAD_CHECK_INTERVAL).await;
        }

        Ok(false)
    }

    /// 提取当前页面内容。
    /// Extract visible page content.
    async fn extract_content(page: &Page) -> Result<String> {
        // 1.使用js代码获取当前页面可见元素内容
        let visible_content = page
            .evaluate(GET_VISIBLE_CONTENT_FUNC)
            .await?
            .into_value::<String>()
            .unwrap_or_default();

        // 2.使用 html-to-markdown-rs 将 HTML 文档转换为 Markdown；失败时回退到纯文本提取
        let markdown_content = html_to_markdown_or_text(&visible_content);

        // 3.模型上下文长度有限，提取最大不超过50k个字符
        Ok(truncate_chars(markdown_content, MAX_CONTENT_CHARS))
    }

    /// 提取当前页面上的可交互元素。
    /// Extract and cache interactive elements.
    async fn extract_interactive_elements(session: &mut BrowserSession) -> Result<Vec<String>> {
        // 2.清除当前页面上的缓存可交互元素列表
        session.interactive_elements_cache.clear();

        // 3.执行js脚本获取可交互的元素列表
        let interactive_elements = session
            .page
            .evaluate(GET_INTERACTIVE_ELEMENTS_FUNC)
            .await?
            .into_value::<Vec<InteractiveElement>>()
            .unwrap_or_default();

        // 4.更新缓存的可交互元素列表
        session.interactive_elements_cache = interactive_elements;

        // 5.格式化可交互元素为字符串
        Ok(session
            .interactive_elements_cache
            .iter()
            .map(|element| {
                format!(
                    "{}:<{}>{}</{}>",
                    element.index, element.tag, element.text, element.tag
                )
            })
            .collect())
    }

    /// 根据传递的索引/id获取对应的元素。
    /// Get an element by its cached index.
    async fn get_element_by_id(
        session: &BrowserSession,
        index: usize,
    ) -> Result<Option<chromiumoxide::Element>> {
        // 1.判断当前页面是否存在可交互元素缓存
        if session.interactive_elements_cache.is_empty()
            || index >= session.interactive_elements_cache.len()
        {
            return Ok(None);
        }

        // 2.构建选择器
        let selector = session
            .interactive_elements_cache
            .get(index)
            .and_then(|element| {
                if element.selector.is_empty() {
                    None
                } else {
                    Some(element.selector.as_str())
                }
            })
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| indexed_selector(index));

        Ok(session.page.find_element(selector.as_str()).await.ok())
    }

    async fn click_coordinates(page: &Page, x: f32, y: f32) -> Result<()> {
        page.click(Point::new(x as f64, y as f64)).await?;
        Ok(())
    }

    async fn move_mouse_coordinates(page: &Page, x: f32, y: f32) -> Result<()> {
        page.move_mouse(Point::new(x as f64, y as f64)).await?;
        Ok(())
    }

    async fn clear_element_value(page: &Page, selector: &str) -> Result<bool> {
        let selector = js_string(selector);
        let script = format!(
            r#"
() => {{
    const element = document.querySelector({selector});
    if (!element) return false;

    element.focus();
    if ("value" in element) {{
        element.value = "";
    }} else if (element.isContentEditable) {{
        element.textContent = "";
    }}

    element.dispatchEvent(new Event("input", {{ bubbles: true }}));
    element.dispatchEvent(new Event("change", {{ bubbles: true }}));
    return true;
}}
"#
        );

        Ok(page
            .evaluate(script.as_str())
            .await?
            .into_value::<bool>()
            .unwrap_or(false))
    }

    async fn input_active_element(page: &Page, text: &str) -> Result<bool> {
        let text = js_string(text);
        let script = format!(
            r#"
() => {{
    const element = document.activeElement;
    if (!element) return false;

    if ("value" in element) {{
        element.value = {text};
    }} else if (element.isContentEditable) {{
        element.textContent = {text};
    }} else {{
        return false;
    }}

    element.dispatchEvent(new Event("input", {{ bubbles: true }}));
    element.dispatchEvent(new Event("change", {{ bubbles: true }}));
    return true;
}}
"#
        );

        Ok(page
            .evaluate(script.as_str())
            .await?
            .into_value::<bool>()
            .unwrap_or(false))
    }

    async fn dispatch_key(page: &Page, key: &str) -> Result<()> {
        let key = js_string(key);
        let script = format!(
            r#"
() => {{
    const rawKey = {key};
    const parts = rawKey.split("+");
    const key = parts[parts.length - 1];
    const eventInit = {{
        key,
        bubbles: true,
        cancelable: true,
        ctrlKey: parts.includes("Control") || parts.includes("Ctrl"),
        metaKey: parts.includes("Meta") || parts.includes("Command"),
        altKey: parts.includes("Alt"),
        shiftKey: parts.includes("Shift")
    }};

    const target = document.activeElement || document.body || document;
    target.dispatchEvent(new KeyboardEvent("keydown", eventInit));
    target.dispatchEvent(new KeyboardEvent("keyup", eventInit));

    if (key === "Enter" && target.form && typeof target.form.requestSubmit === "function") {{
        target.form.requestSubmit();
    }}

    return true;
}}
"#
        );

        page.evaluate(script.as_str()).await?;
        Ok(())
    }

    async fn select_indexed_option(page: &Page, index: usize, option: usize) -> Result<bool> {
        let selector = js_string(indexed_selector(index).as_str());
        let script = format!(
            r#"
() => {{
    const element = document.querySelector({selector});
    if (!element || element.tagName.toLowerCase() !== "select") return false;
    if (element.options.length <= {option}) return false;

    element.selectedIndex = {option};
    element.dispatchEvent(new Event("input", {{ bubbles: true }}));
    element.dispatchEvent(new Event("change", {{ bubbles: true }}));
    return true;
}}
"#
        );

        Ok(page
            .evaluate(script.as_str())
            .await?
            .into_value::<bool>()
            .unwrap_or(false))
    }
}

impl Default for ChromiumoxideBrowser {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserSession {
    async fn launch() -> Result<Self> {
        // 3.创建隔离的浏览器用户目录，避免服务端工具污染用户本机 Chrome。
        let temp_dir = tempfile::Builder::new()
            .prefix("nexus-browser-")
            .tempdir()
            .context("创建浏览器临时用户目录失败")?;

        // 4.使用 chromiumoxide 启动服务端自管理的 Chrome 实例。
        let mut builder = BrowserConfig::builder()
            .user_data_dir(temp_dir.path())
            .new_headless_mode()
            .arg("--disable-dev-shm-usage")
            .arg("--disable-extensions")
            .arg("--disable-background-networking");

        // 5.清理传递给浏览器子进程的敏感环境变量。
        for var in BLOCKED_ENV_VARS {
            builder = builder.env(*var, "");
        }

        let config = builder
            .build()
            .map_err(|error| anyhow!("构建 chromiumoxide 浏览器配置失败: {error}"))?;
        let (browser, mut handler) = ChromeBrowser::launch(config).await?;
        let handler = tokio::spawn(async move { while handler.next().await.is_some() {} });

        // 6.创建工具专用空白页面。
        let page = browser.new_page("about:blank").await?;

        Ok(Self {
            browser,
            page,
            handler: Some(handler),
            last_used: Instant::now(),
            _temp_dir: temp_dir,
            interactive_elements_cache: Vec::new(),
        })
    }

    fn is_idle(&self) -> bool {
        self.last_used.elapsed() > IDLE_TIMEOUT
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    async fn shutdown(mut self) {
        // 6.判断当前页面是否关闭：chromiumoxide 没有同步 is_closed，这里执行最佳努力关闭
        let _ = self.page.clone().close().await;

        // 7.关闭当前适配器拥有的 Chrome 子进程，并等待进程退出。
        let _ = self.browser.close().await;
        let _ = self.browser.wait().await;

        // 8.停止 chromiumoxide handler；临时用户目录随 TempDir drop 自动回收。
        if let Some(handler) = self.handler.take() {
            handler.abort();
            let _ = handler.await;
        }
    }
}

impl Drop for BrowserSession {
    fn drop(&mut self) {
        if let Some(handler) = &self.handler {
            handler.abort();
        }
    }
}

#[async_trait]
impl Browser for ChromiumoxideBrowser {
    /// 获取当前网页的内容(内容+可交互元素列表)
    async fn view_page(&self) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.等待页面加载完成
        let _ = Self::wait_for_page_load(&session.page).await?;

        // 3.更新页面的可交互元素
        let interactive_elements = Self::extract_interactive_elements(session).await?;

        // 4.返回工具结果
        Ok(success_data(
            json!({
                "content": Self::extract_content(&session.page).await?,
                "interactive_elements": interactive_elements,
            })
            .to_string(),
        ))
    }

    /// 根据传递的url跳转到指定页面
    async fn navigate(&self, url: &str) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.在跳转之前先将可交互元素的缓存清空
        session.interactive_elements_cache.clear();

        // 3.使用goto进行跳转
        if let Err(error) = session.page.goto(url).await {
            return Ok(failure(format!("浏览器导航到[{url}]失败: {error}")));
        }

        let _ = tokio::time::timeout(PAGE_LOAD_TIMEOUT, session.page.wait_for_navigation()).await;
        let interactive_elements = Self::extract_interactive_elements(session).await?;

        Ok(success_data(
            json!({
                "interactive_elements": interactive_elements,
            })
            .to_string(),
        ))
    }

    /// 重启并跳转到指定URL
    async fn restart(&self, url: &str) -> Result<ToolResult<String>> {
        self.cleanup().await;
        self.navigate(url).await
    }

    /// 根据传递的索引位置+xy坐标实现点击
    async fn click(
        &self,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.判断传递的是xy坐标还是index
        if let (Some(x), Some(y)) = (coordinate_x, coordinate_y) {
            Self::click_coordinates(&session.page, x, y).await?;
        } else if let Some(index) = index {
            // 3.根据index获取元素
            let element = match Self::get_element_by_id(session, index).await? {
                Some(element) => element,
                None => {
                    return Ok(failure(format!("使用索引{index}查找该元素无效, 未找到")));
                }
            };

            // 4-6.chromiumoxide 的 element.click 会检查点击点并滚动到元素位置
            // 7.点击元素
            if let Err(error) = element.click().await {
                return Ok(failure(format!("点击元素出错: {error}")));
            }
        }

        Ok(success_message("点击成功"))
    }

    /// 根据传递的文本+换行标识+索引+xy位置实现输入框文本输入
    async fn input(
        &self,
        text: &str,
        press_enter: bool,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.判断下是传递xy还是index
        if let (Some(x), Some(y)) = (coordinate_x, coordinate_y) {
            // 3.点击指定位置后输入文本
            Self::click_coordinates(&session.page, x, y).await?;
            if !Self::input_active_element(&session.page, text).await? {
                return Ok(failure("输入文本失败, 当前焦点元素不可编辑"));
            }
        } else if let Some(index) = index {
            // 4.根据索引查找元素
            let element = match Self::get_element_by_id(session, index).await? {
                Some(element) => element,
                None => return Ok(failure("输入文本失败, 该元素不存在")),
            };

            let selector = indexed_selector(index);

            // 5.先清空原始输入框的内容然后再填充
            let _ = Self::clear_element_value(&session.page, selector.as_str()).await;
            match element.click().await {
                Ok(element) => {
                    if let Err(error) = element.type_str(text).await {
                        // 6.如果填充失败则尝试点击后输入文本
                        return Ok(failure(format!("输入文本失败: {error}")));
                    }
                }
                Err(error) => return Ok(failure(format!("输入文本失败: {error}"))),
            }
        }

        // 7.判断是否按Enter键
        if press_enter {
            Self::dispatch_key(&session.page, "Enter").await?;
        }

        Ok(success_message("输入文本成功"))
    }

    /// 传递xy坐标移动鼠标
    async fn move_mouse(&self, coordinate_x: f32, coordinate_y: f32) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        Self::move_mouse_coordinates(&session.page, coordinate_x, coordinate_y).await?;
        Ok(success_message("移动鼠标成功"))
    }

    /// 传递按键进行模拟
    async fn press_key(&self, key: &str) -> Result<ToolResult<String>> {
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        Self::dispatch_key(&session.page, key).await?;
        Ok(success_message("按键成功"))
    }

    /// 传递索引+下拉菜单选项选择指定的菜单信息
    async fn select_option(&self, index: usize, option: usize) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.获取元素信息
        if Self::get_element_by_id(session, index).await?.is_none() {
            return Ok(failure(format!(
                "使用索引[{index}]查找该下拉菜单元素不存在"
            )));
        }

        // 3.调用函数直接选择对应选项
        if Self::select_indexed_option(&session.page, index, option).await? {
            Ok(success_message("选择下拉菜单选项成功"))
        } else {
            Ok(failure(format!("选择下拉菜单选项失败: 选项[{option}]无效")))
        }
    }

    /// 向上滚动浏览器一个屏幕或者整个页面
    async fn scroll_up(&self, to_top: Option<bool>) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.判断是否滚动到最顶部
        if to_top.unwrap_or(false) {
            session.page.evaluate("() => window.scrollTo(0, 0)").await?;
        } else {
            session
                .page
                .evaluate("() => window.scrollBy(0, -window.innerHeight)")
                .await?;
        }

        Ok(success_message("向上滚动成功"))
    }

    /// 向下滚动浏览器一个屏幕或者到最底部
    async fn scroll_down(&self, to_down: Option<bool>) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.判断是否滚动到最底部
        if to_down.unwrap_or(false) {
            session
                .page
                .evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                .await?;
        } else {
            session
                .page
                .evaluate("() => window.scrollBy(0, window.innerHeight)")
                .await?;
        }

        Ok(success_message("向下滚动成功"))
    }

    /// 传递full_page完成网页截图
    async fn screenshot(&self, full_page: Option<bool>) -> Result<Vec<u8>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.创建一个截图配置
        let screenshot_options = ScreenshotParams::builder()
            .full_page(full_page.unwrap_or(false))
            .build();

        Ok(session.page.screenshot(screenshot_options).await?)
    }

    /// 传递js代码在当前页面控制台执行
    async fn console_exec(&self, javascript: &str) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.在正式开始执行代码之前先注入logs
        let _ = session.page.evaluate(INJECT_CONSOLE_LOGS_FUNC).await;

        // 3.正式执行js脚本
        let result = session.page.evaluate(javascript).await?;
        let value = result.value().cloned().unwrap_or(Value::Null);

        Ok(success_data(json!({ "result": value }).to_string()))
    }

    /// 根据传递的行数查看控制台的日志
    async fn console_view(&self, max_lines: Option<usize>) -> Result<ToolResult<String>> {
        // 1.确保页面存在
        let mut guard = self.ensure_page().await?;
        let session = guard.as_mut().ok_or_else(|| anyhow!("浏览器会话不存在"))?;

        // 2.可以指定另外一段js代码查看控制台的内容
        let logs = session
            .page
            .evaluate(
                r#"() => {
                    return window.console.logs || [];
                }"#,
            )
            .await?
            .value()
            .cloned()
            .unwrap_or_else(|| json!([]));

        let logs = match (logs, max_lines) {
            (Value::Array(logs), Some(max_lines)) => {
                let start = logs.len().saturating_sub(max_lines);
                Value::Array(logs.into_iter().skip(start).collect())
            }
            (logs, _) => logs,
        };

        Ok(success_data(json!({ "logs": logs }).to_string()))
    }
}

fn indexed_selector(index: usize) -> String {
    format!(r#"[data-manus-id="manus-element-{index}"]"#)
}

fn js_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn html_to_text(html: &str) -> String {
    let fragment = Html::parse_fragment(html);
    let text = fragment
        .root_element()
        .text()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    if text.is_empty() {
        html.to_string()
    } else {
        text
    }
}

fn html_to_markdown_or_text(html: &str) -> String {
    html_to_markdown_rs::convert(html, None)
        .ok()
        .and_then(|result| result.content)
        .filter(|content| !content.trim().is_empty())
        .unwrap_or_else(|| html_to_text(html))
}

fn truncate_chars(value: String, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value;
    }

    value.chars().take(max_chars).collect()
}

fn success_message(message: impl Into<String>) -> ToolResult<String> {
    ToolResult {
        success: true,
        message: Some(message.into()),
        data: None,
    }
}

fn success_data(data: String) -> ToolResult<String> {
    ToolResult {
        success: true,
        message: Some(String::new()),
        data: Some(data),
    }
}

fn failure(message: impl Into<String>) -> ToolResult<String> {
    ToolResult {
        success: false,
        message: Some(message.into()),
        data: None,
    }
}
