import asyncio
import os
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_SERVER_URL = "https://qianfan.baidubce.com/v2/ai_search/mcp"
DEFAULT_TOOL_NAME = "chatCompletions"
DEFAULT_QUERY = "2025年广州马拉松"


async def main() -> None:
    # 1. 读取百度 AI 搜索配置。/ Read Baidu AI Search configuration.
    baidu_ai_search_api = os.getenv("MCP_CONNECT_API_URL", DEFAULT_SERVER_URL)
    tool_name = os.getenv("MCP_CONNECT_API_TOOL", DEFAULT_TOOL_NAME)
    query = os.getenv("MCP_CONNECT_API_QUERY", DEFAULT_QUERY)
    token = os.getenv("MCP_CONNECT_API_TOKEN") or os.getenv("BAIDU_AI_SEARCH_API_KEY")
    if not token:
        raise RuntimeError("缺少环境变量 MCP_CONNECT_API_TOKEN 或 BAIDU_AI_SEARCH_API_KEY")

    token = token.removeprefix("Bearer ").removeprefix("bearer ").strip()
    headers = {"Authorization": f"Bearer {token}"}

    # 2. 创建异步上下文管理器。/ Create an async exit stack.
    exit_stack = AsyncExitStack()

    try:
        # 3. 创建连接客户端。/ Create the streamable HTTP client.
        transport = await exit_stack.enter_async_context(
            streamablehttp_client(url=baidu_ai_search_api, headers=headers)
        )

        # 4. 获取读取、写入流。/ Get read and write streams.
        read_stream, write_steam, _ = transport

        # 5. 创建客户端会话。/ Create the client session.
        session: ClientSession = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_steam)
        )

        # 6. 初始化会话。/ Initialize the session.
        await session.initialize()

        # 7. 获取工具列表并输出。/ List tools and print the result.
        list_tools_result = await session.list_tools()
        print(list_tools_result)

        # 8. 调用指定工具实现百度搜索。/ Call the configured tool.
        call_tool_result = await session.call_tool(tool_name, {"query": query})
        print("工具调用结果：", call_tool_result.content[0].text)
    finally:
        await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())
