use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::domain::{
    external::Browser,
    models::ToolResult,
    services::tools::{tool, BaseTool, ToolArguments, ToolDefinition},
};

/// 浏览器工具
pub struct BrowserTool {
    name: String,
    browser: Box<dyn Browser>,
    definitions: Vec<ToolDefinition>,
}

impl BrowserTool {
    /// 构造函数，完成浏览器工具的初始化
    pub fn new(browser: Box<dyn Browser>) -> Self {
        Self {
            name: "browser".to_string(),
            browser,
            definitions: vec![
                tool(
                    "browser_view",
                    "查看当前浏览器页面内容，用于确认已打开页面的最新状态。",
                    ToolArguments::new(),
                    vec![],
                ),
                tool(
                    "browser_navigate",
                    "将浏览器导航至指定网址，当需要访问新页面时使用。",
                    ToolArguments::from_iter([(
                        "url".to_string(),
                        json!({
                            "type": "string",
                            "description": "要访问的完整URL，必须包含协议前缀(如https://)"
                        }),
                    )]),
                    vec!["url".to_string()],
                ),
                tool(
                    "browser_restart",
                    "重新启动浏览器并导航至指定URL，当需要重置浏览器时使用。",
                    ToolArguments::from_iter([(
                        "url".to_string(),
                        json!({
                            "type": "string",
                            "description": "要访问的完整URL，必须包含协议前缀(如https://)"
                        }),
                    )]),
                    vec!["url".to_string()],
                ),
                tool(
                    "browser_click",
                    "点击当前页面中的元素，在需要点击页面元素时使用。",
                    ToolArguments::from_iter([
                        (
                            "index".to_string(),
                            json!({
                                "type": "integer",
                                "description": "(可选)需要点击的元素索引"
                            }),
                        ),
                        (
                            "coordinate_x".to_string(),
                            json!({
                                "type": "number",
                                "description": "(可选)点击位置的x坐标"
                            }),
                        ),
                        (
                            "coordinate_y".to_string(),
                            json!({
                                "type": "number",
                                "description": "(可选)点击位置的y坐标"
                            }),
                        ),
                    ]),
                    vec![],
                ),
                tool(
                    "browser_input",
                    "覆盖浏览器当前页面可编辑区域的文本(input/textarea输入框)，在需要填充输入框时使用。",
                    ToolArguments::from_iter([
                        (
                            "text".to_string(),
                            json!({
                                "type": "string",
                                "description": "要填充到输入框的完整文本内容"
                            }),
                        ),
                        (
                            "press_enter".to_string(),
                            json!({
                                "type": "boolean",
                                "description": "输入后是否按下回车键"
                            }),
                        ),
                        (
                            "index".to_string(),
                            json!({
                                "type": "integer",
                                "description": "(可选)需要填充文本的元素索引"
                            }),
                        ),
                        (
                            "coordinate_x".to_string(),
                            json!({
                                "type": "number",
                                "description": "(可选)需要填充文本元素的x坐标"
                            }),
                        ),
                        (
                            "coordinate_y".to_string(),
                            json!({
                                "type": "number",
                                "description": "(可选)需要填充文本元素的y坐标"
                            }),
                        ),
                    ]),
                    vec!["text".to_string(), "press_enter".to_string()],
                ),
                tool(
                    "browser_move_mouse",
                    "将鼠标光标移动至当前浏览器页面的指定位置，用于模拟用户的鼠标移动",
                    ToolArguments::from_iter([
                        (
                            "coordinate_x".to_string(),
                            json!({
                                "type": "number",
                                "description": "目标光标位置的x坐标"
                            }),
                        ),
                        (
                            "coordinate_y".to_string(),
                            json!({
                                "type": "number",
                                "description": "目标光标位置的y坐标"
                            }),
                        ),
                    ]),
                    vec!["coordinate_x".to_string(), "coordinate_y".to_string()],
                ),
                tool(
                    "browser_press_key",
                    "在当前浏览器页面模拟按键，当需要执行特定的键盘操作时使用。",
                    ToolArguments::from_iter([(
                        "key".to_string(),
                        json!({
                            "type": "string",
                            "description": "要模拟的按键名称(例如: Enter、Tab、ArrowUp)，支持组合键(例如: Control+Enter)"
                        }),
                    )]),
                    vec!["key".to_string()],
                ),
                tool(
                    "browser_select_option",
                    "从当前浏览器页面的下拉列表元素中选择指定选项，用于选择下拉菜单中的选项",
                    ToolArguments::from_iter([
                        (
                            "index".to_string(),
                            json!({
                                "type": "integer",
                                "description": "需要操作的下拉列表元素的索引号(序号)"
                            }),
                        ),
                        (
                            "option".to_string(),
                            json!({
                                "type": "integer",
                                "description": "要选择的选项序号，从0开始(注: 指下拉框里的第几项)。"
                            }),
                        ),
                    ]),
                    vec!["index".to_string(), "option".to_string()],
                ),
                tool(
                    "browser_scroll_up",
                    "向上滚动浏览器页面，用于查看上方内容或返回页面顶部。",
                    ToolArguments::from_iter([(
                        "to_top".to_string(),
                        json!({
                            "type": "boolean",
                            "description": "(可选)是否直接滚动到页面顶部，而非向上滚动一屏。"
                        }),
                    )]),
                    vec![],
                ),
                tool(
                    "browser_scroll_down",
                    "向下滚动当前浏览器页面，用于查看下方内容或跳转到页面底部。",
                    ToolArguments::from_iter([(
                        "to_bottom".to_string(),
                        json!({
                            "type": "boolean",
                            "description": "(可选)是否直接滚动到页面底部，而非向下滚动一屏"
                        }),
                    )]),
                    vec![],
                ),
                tool(
                    "browser_console_exec",
                    "在浏览器控制台中执行JavaScript代码，当需要执行自定义脚本时使用。",
                    ToolArguments::from_iter([(
                        "javascript".to_string(),
                        json!({
                            "type": "string",
                            "description": "要执行的JavaScript代码，请注意运行时环境为浏览器控制台。"
                        }),
                    )]),
                    vec!["javascript".to_string()],
                ),
                tool(
                    "browser_console_view",
                    "查看浏览器控制台输出，用于检查JavaScript日志或调试页面错误。",
                    ToolArguments::from_iter([(
                        "max_lines".to_string(),
                        json!({
                            "type": "integer",
                            "description": "(可选)返回的最大日志行数。"
                        }),
                    )]),
                    vec![],
                ),
            ],
        }
    }

    /// 获取浏览器当前网页内容并返回
    async fn browser_view(&self) -> Result<ToolResult<String>> {
        self.browser.view_page().await
    }

    /// 传递url地址，使用浏览器导航到对应页面
    async fn browser_navigate(&self, url: &str) -> Result<ToolResult<String>> {
        self.browser.navigate(url).await
    }

    /// 重启浏览器并导航到指定页面后返回页面内容
    async fn browser_restart(&self, url: &str) -> Result<ToolResult<String>> {
        self.browser.restart(url).await
    }

    /// 传递页面元素索引或者页面xy坐标点击对应元素
    async fn browser_click(
        &self,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        self.browser.click(index, coordinate_x, coordinate_y).await
    }

    /// 根据传递的元素索引或xy坐标，实现浏览器内容输入
    async fn browser_input(
        &self,
        text: &str,
        press_enter: bool,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>> {
        self.browser
            .input(text, press_enter, index, coordinate_x, coordinate_y)
            .await
    }

    /// 传递xy坐标移动浏览器鼠标
    async fn browser_move_mouse(
        &self,
        coordinate_x: f32,
        coordinate_y: f32,
    ) -> Result<ToolResult<String>> {
        self.browser.move_mouse(coordinate_x, coordinate_y).await
    }

    /// 在浏览器页面模拟按键
    async fn browser_press_key(&self, key: &str) -> Result<ToolResult<String>> {
        self.browser.press_key(key).await
    }

    /// 传递索引+下拉元素选项序号执行选择
    async fn browser_select_option(
        &self,
        index: usize,
        option: usize,
    ) -> Result<ToolResult<String>> {
        self.browser.select_option(index, option).await
    }

    /// 向上滚动当前浏览器页面，支持滚动一页或者滚动到顶部
    async fn browser_scroll_up(&self, to_top: Option<bool>) -> Result<ToolResult<String>> {
        self.browser.scroll_up(to_top).await
    }

    /// 向下滚动当前浏览器页面，支持滚动一页或者滚动到底部
    async fn browser_scroll_down(&self, to_bottom: Option<bool>) -> Result<ToolResult<String>> {
        self.browser.scroll_down(to_bottom).await
    }

    /// 传递js脚本在当前浏览器控制台执行
    async fn browser_console_exec(&self, javascript: &str) -> Result<ToolResult<String>> {
        self.browser.console_exec(javascript).await
    }

    /// 传递浏览的最大行数查看控制台的输出
    async fn browser_console_view(&self, max_lines: Option<usize>) -> Result<ToolResult<String>> {
        self.browser.console_view(max_lines).await
    }
}

#[async_trait]
impl BaseTool for BrowserTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        let result = match tool_name {
            "browser_view" => self.browser_view().await?,
            "browser_navigate" => {
                let url = required_str(&kwargs, "url")?;
                self.browser_navigate(url).await?
            }
            "browser_restart" => {
                let url = required_str(&kwargs, "url")?;
                self.browser_restart(url).await?
            }
            "browser_click" => {
                let index = optional_usize(&kwargs, "index")?;
                let coordinate_x = optional_f32(&kwargs, "coordinate_x")?;
                let coordinate_y = optional_f32(&kwargs, "coordinate_y")?;
                self.browser_click(index, coordinate_x, coordinate_y)
                    .await?
            }
            "browser_input" => {
                let text = required_str(&kwargs, "text")?;
                let press_enter = required_bool(&kwargs, "press_enter")?;
                let index = optional_usize(&kwargs, "index")?;
                let coordinate_x = optional_f32(&kwargs, "coordinate_x")?;
                let coordinate_y = optional_f32(&kwargs, "coordinate_y")?;
                self.browser_input(text, press_enter, index, coordinate_x, coordinate_y)
                    .await?
            }
            "browser_move_mouse" => {
                let coordinate_x = required_f32(&kwargs, "coordinate_x")?;
                let coordinate_y = required_f32(&kwargs, "coordinate_y")?;
                self.browser_move_mouse(coordinate_x, coordinate_y).await?
            }
            "browser_press_key" => {
                let key = required_str(&kwargs, "key")?;
                self.browser_press_key(key).await?
            }
            "browser_select_option" => {
                let index = required_usize(&kwargs, "index")?;
                let option = required_usize(&kwargs, "option")?;
                self.browser_select_option(index, option).await?
            }
            "browser_scroll_up" => {
                let to_top = optional_bool(&kwargs, "to_top")?;
                self.browser_scroll_up(to_top).await?
            }
            "browser_scroll_down" => {
                let to_bottom = optional_bool(&kwargs, "to_bottom")?;
                self.browser_scroll_down(to_bottom).await?
            }
            "browser_console_exec" => {
                let javascript = required_str(&kwargs, "javascript")?;
                self.browser_console_exec(javascript).await?
            }
            "browser_console_view" => {
                let max_lines = optional_usize(&kwargs, "max_lines")?;
                self.browser_console_view(max_lines).await?
            }
            _ => return Err(anyhow!("工具[{tool_name}]未找到")),
        };

        Ok(ToolResult {
            success: result.success,
            message: result.message,
            data: result.data.map(Value::String),
        })
    }
}

fn required_str<'a>(kwargs: &'a ToolArguments, name: &str) -> Result<&'a str> {
    kwargs
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn required_bool(kwargs: &ToolArguments, name: &str) -> Result<bool> {
    kwargs
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn required_f32(kwargs: &ToolArguments, name: &str) -> Result<f32> {
    optional_f32(kwargs, name)?.ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn required_usize(kwargs: &ToolArguments, name: &str) -> Result<usize> {
    optional_usize(kwargs, name)?.ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

fn optional_bool(kwargs: &ToolArguments, name: &str) -> Result<Option<bool>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是布尔值")),
    }
}

fn optional_f32(kwargs: &ToolArguments, name: &str) -> Result<Option<f32>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是数字")),
    }
}

fn optional_usize(kwargs: &ToolArguments, name: &str) -> Result<Option<usize>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是非负整数")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{json, Map};

    use super::*;

    type Calls = Arc<Mutex<Vec<BrowserCall>>>;

    #[derive(Debug, Clone, PartialEq)]
    enum BrowserCall {
        ViewPage,
        Navigate(String),
        Restart(String),
        Click(Option<usize>, Option<f32>, Option<f32>),
        Input(String, bool, Option<usize>, Option<f32>, Option<f32>),
        MoveMouse(f32, f32),
        PressKey(String),
        SelectOption(usize, usize),
        ScrollUp(Option<bool>),
        ScrollDown(Option<bool>),
        Screenshot(Option<bool>),
        ConsoleExec(String),
        ConsoleView(Option<usize>),
    }

    struct MockBrowser {
        calls: Calls,
    }

    impl MockBrowser {
        fn record(&self, call: BrowserCall, data: &str) -> ToolResult<String> {
            self.calls.lock().unwrap().push(call);
            ToolResult {
                data: Some(data.to_string()),
                ..ToolResult::default()
            }
        }
    }

    #[async_trait]
    impl Browser for MockBrowser {
        async fn view_page(&self) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::ViewPage, "view"))
        }

        async fn navigate(&self, url: &str) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::Navigate(url.to_string()), "navigate"))
        }

        async fn restart(&self, url: &str) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::Restart(url.to_string()), "restart"))
        }

        async fn click(
            &self,
            index: Option<usize>,
            coordinate_x: Option<f32>,
            coordinate_y: Option<f32>,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                BrowserCall::Click(index, coordinate_x, coordinate_y),
                "click",
            ))
        }

        async fn input(
            &self,
            text: &str,
            press_enter: bool,
            index: Option<usize>,
            coordinate_x: Option<f32>,
            coordinate_y: Option<f32>,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(
                BrowserCall::Input(
                    text.to_string(),
                    press_enter,
                    index,
                    coordinate_x,
                    coordinate_y,
                ),
                "input",
            ))
        }

        async fn move_mouse(
            &self,
            coordinate_x: f32,
            coordinate_y: f32,
        ) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::MoveMouse(coordinate_x, coordinate_y), "move"))
        }

        async fn press_key(&self, key: &str) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::PressKey(key.to_string()), "press"))
        }

        async fn select_option(&self, index: usize, option: usize) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::SelectOption(index, option), "select"))
        }

        async fn scroll_up(&self, to_top: Option<bool>) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::ScrollUp(to_top), "scroll_up"))
        }

        async fn scroll_down(&self, to_down: Option<bool>) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::ScrollDown(to_down), "scroll_down"))
        }

        async fn screenshot(&self, full_page: Option<bool>) -> Result<Vec<u8>> {
            self.calls
                .lock()
                .unwrap()
                .push(BrowserCall::Screenshot(full_page));
            Ok(vec![1, 2, 3])
        }

        async fn console_exec(&self, javascript: &str) -> Result<ToolResult<String>> {
            Ok(self.record(
                BrowserCall::ConsoleExec(javascript.to_string()),
                "console_exec",
            ))
        }

        async fn console_view(&self, max_lines: Option<usize>) -> Result<ToolResult<String>> {
            Ok(self.record(BrowserCall::ConsoleView(max_lines), "console_view"))
        }
    }

    fn browser_tool() -> (BrowserTool, Calls) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let browser = MockBrowser {
            calls: Arc::clone(&calls),
        };

        (BrowserTool::new(Box::new(browser)), calls)
    }

    #[test]
    fn exposes_browser_schema_like_python_tool_decorator() {
        let (tool, _) = browser_tool();

        assert_eq!(tool.name(), "browser");
        assert!(tool.has_tool("browser_view"));
        assert!(tool.has_tool("browser_console_view"));

        let tools = tool.get_tools();
        let tool_names = tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            tool_names,
            vec![
                "browser_view",
                "browser_navigate",
                "browser_restart",
                "browser_click",
                "browser_input",
                "browser_move_mouse",
                "browser_press_key",
                "browser_select_option",
                "browser_scroll_up",
                "browser_scroll_down",
                "browser_console_exec",
                "browser_console_view",
            ]
        );
        assert_eq!(
            tools[1]["function"]["parameters"]["properties"]["url"]["description"],
            "要访问的完整URL，必须包含协议前缀(如https://)"
        );
        assert_eq!(
            tools[4]["function"]["parameters"]["required"],
            json!(["text", "press_enter"])
        );
        assert_eq!(tools[3]["function"]["parameters"]["required"], json!([]));
    }

    #[tokio::test]
    async fn dispatches_python_browser_tools() {
        let (tool, calls) = browser_tool();

        assert_eq!(
            tool.invoke("browser_view", Map::new())
                .await
                .unwrap()
                .data
                .unwrap(),
            "view"
        );
        tool.invoke(
            "browser_navigate",
            Map::from_iter([("url".to_string(), json!("https://example.com"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_restart",
            Map::from_iter([("url".to_string(), json!("https://restart.test"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_input",
            Map::from_iter([
                ("text".to_string(), json!("hello")),
                ("press_enter".to_string(), json!(false)),
                ("index".to_string(), json!(3)),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_move_mouse",
            Map::from_iter([
                ("coordinate_x".to_string(), json!(12.5)),
                ("coordinate_y".to_string(), json!(24.0)),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_press_key",
            Map::from_iter([("key".to_string(), json!("Control+Enter"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_select_option",
            Map::from_iter([
                ("index".to_string(), json!(2)),
                ("option".to_string(), json!(1)),
            ]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_scroll_up",
            Map::from_iter([("to_top".to_string(), json!(true))]),
        )
        .await
        .unwrap();
        tool.invoke("browser_scroll_down", Map::new())
            .await
            .unwrap();
        tool.invoke(
            "browser_console_exec",
            Map::from_iter([("javascript".to_string(), json!("window.location.href"))]),
        )
        .await
        .unwrap();
        tool.invoke(
            "browser_console_view",
            Map::from_iter([("max_lines".to_string(), json!(50))]),
        )
        .await
        .unwrap();

        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                BrowserCall::ViewPage,
                BrowserCall::Navigate("https://example.com".to_string()),
                BrowserCall::Restart("https://restart.test".to_string()),
                BrowserCall::Input("hello".to_string(), false, Some(3), None, None),
                BrowserCall::MoveMouse(12.5, 24.0),
                BrowserCall::PressKey("Control+Enter".to_string()),
                BrowserCall::SelectOption(2, 1),
                BrowserCall::ScrollUp(Some(true)),
                BrowserCall::ScrollDown(None),
                BrowserCall::ConsoleExec("window.location.href".to_string()),
                BrowserCall::ConsoleView(Some(50)),
            ]
        );
    }

    #[tokio::test]
    async fn click_accepts_index_or_coordinates_and_filters_extra_arguments() {
        let (tool, calls) = browser_tool();

        let result = tool
            .invoke(
                "browser_click",
                Map::from_iter([
                    ("index".to_string(), json!(7)),
                    ("coordinate_x".to_string(), json!(30.5)),
                    ("coordinate_y".to_string(), json!(40.25)),
                    ("ignored".to_string(), json!("value")),
                ]),
            )
            .await
            .unwrap();

        assert_eq!(result.data.unwrap(), "click");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![BrowserCall::Click(Some(7), Some(30.5), Some(40.25))]
        );
    }

    #[tokio::test]
    async fn rejects_missing_required_browser_argument() {
        let (tool, _) = browser_tool();
        let error = tool
            .invoke("browser_navigate", Map::new())
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "工具参数[url]缺失");
    }
}
