//! 浏览器服务扩展，涵盖：访问页面、URL 跳转、输入框数据填充、移动鼠标、滚动页面、截图、执行 JS 代码、查看控制台输出等
use anyhow::Result;
use async_trait::async_trait;

use crate::domain::models::ToolResult;

#[async_trait]
pub trait Browser: Send + Sync {
    /// 清理浏览器会话持有的页面与连接资源。
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }

    /// 获取当前浏览器页面的内容源码
    async fn view_page(&self) -> Result<ToolResult<String>>;

    /// 传递对应的 url 使用浏览器导航到该页面
    async fn navigate(&self, url: &str) -> Result<ToolResult<String>>;

    /// 重启浏览器并访问对应的 URL
    async fn restart(&self, url: &str) -> Result<ToolResult<String>>;

    /// 传递对应元素的索引或者 x、y 坐标实现点击功能
    async fn click(
        &self,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>>;

    /// 传递文本+回车表示+索引/xy坐标实现在网页输入框中输入对应的内容
    async fn input(
        &self,
        text: &str,
        press_enter: bool,
        index: Option<usize>,
        coordinate_x: Option<f32>,
        coordinate_y: Option<f32>,
    ) -> Result<ToolResult<String>>;

    /// 传递对应的 x、y 坐标移动鼠标
    async fn move_mouse(&self, coordinate_x: f32, coordinate_y: f32) -> Result<ToolResult<String>>;

    /// 传递按键标识 Enter/Control+C 等实现浏览器模拟按键
    async fn press_key(&self, key: &str) -> Result<ToolResult<String>>;

    /// 传递选项索引 + 选项序号表示在下拉菜单中选择指定的选项
    async fn select_option(&self, index: usize, option: usize) -> Result<ToolResult<String>>;

    /// 向上滚动浏览器，如果没有传递 to_top=true 则向上滚动一页，否则直接滚动到最顶部
    async fn scroll_up(&self, to_top: Option<bool>) -> Result<ToolResult<String>>;

    /// 向下滚动浏览器，如果没有传递 to_down=true 则向下滚动一页，否则直接滚动到最底部
    async fn scroll_down(&self, to_down: Option<bool>) -> Result<ToolResult<String>>;

    /// 对当前浏览器的页面进行截图，传递 full_page=True 则意味着整页截图
    async fn screenshot(&self, full_page: Option<bool>) -> Result<Vec<u8>>;

    /// 传递对应的 js 脚本在浏览器的当前页面控制台执行
    async fn console_exec(&self, javascript: &str) -> Result<ToolResult<String>>;

    /// 传递最大输出行数，获取控制台的输出结果，如果不传递则表示获取所有输出结果
    async fn console_view(&self, max_lines: Option<usize>) -> Result<ToolResult<String>>;
}
