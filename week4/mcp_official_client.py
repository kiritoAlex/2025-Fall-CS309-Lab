"""
使用官方 @modelcontextprotocol/server-filesystem 的 MCP 客户端示例
演示如何连接到 MCP 服务器并使用其提供的工具
"""

import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI

# 加载环境变量
load_dotenv()

# ============= MCP 客户端配置 =============
# 官方文件系统服务器参数
# 需要先安装: npm install -g @modelcontextprotocol/server-filesystem
server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/seven/Project/2025-Fall-CS309-Lab/week4"  # 允许访问的目录
    ]
)

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    async def connect_to_server(self, server_params: StdioServerParameters):
        """连接到 MCP 服务器"""
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # 列出可用的工具
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\n🔧 Connected to MCP server. Available tools: {len(tools)}")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")

        return tools

    async def process_query(self, query: str, max_iterations: int = 10):
        """处理用户查询，使用 MCP 工具"""
        print(f"\n{'='*60}")
        print(f"🤖 MCP Client with LLM")
        print(f"{'='*60}")
        print(f"📝 User Query: {query}\n")

        # 获取可用工具并转换为 OpenAI 格式
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 初始化消息
        messages = [{"role": "user", "content": query}]

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # 调用 OpenAI API
            response = self.openai.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=available_tools
            )

            message = response.choices[0].message
            print(f"Stop reason: {response.choices[0].finish_reason}")

            # 如果不需要工具调用,返回结果
            if response.choices[0].finish_reason == "stop":
                final_response = message.content
                print(f"\n✅ Final Response:\n{final_response}")
                return final_response

            # 处理工具调用
            if response.choices[0].finish_reason == "tool_calls" and message.tool_calls:
                # 添加 assistant 的响应到消息历史
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })

                # 执行所有工具调用
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"\n🔧 Tool Call:")
                    print(f"   Tool: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")

                    # 通过 MCP 执行工具
                    result = await self.session.call_tool(tool_name, tool_args)

                    # 提取文本内容
                    result_text = ""
                    if hasattr(result, 'content') and result.content:
                        if isinstance(result.content, list):
                            # 合并所有文本内容
                            result_text = "\n".join(
                                item.text if hasattr(item, 'text') else str(item)
                                for item in result.content
                            )
                        else:
                            result_text = str(result.content)

                    print(f"   Result: {result_text[:200]}...")

                    # 添加工具结果到消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

        print("\n⚠️  Reached maximum iterations")
        return None

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    """主函数：演示 MCP 客户端的使用"""
    client = MCPClient()

    try:
        # 连接到服务器
        print("🔌 Connecting to MCP filesystem server...")
        await client.connect_to_server(server_params)

        # 测试案例 1: 简单文件操作
        print("\n" + "="*60)
        print("测试案例 1: 列出目录内容")
        print("="*60)
        await client.process_query("列出当前目录下的所有文件")

        # 测试案例 2: 读取文件
        print("\n" + "="*60)
        print("测试案例 2: 读取文件")
        print("="*60)
        await client.process_query("读取 README.md 文件并总结其主要内容")

        # 测试案例 3: 复杂任务
        print("\n" + "="*60)
        print("测试案例 3: 多步骤任务")
        print("="*60)
        await client.process_query(
            "找到所有 .py 文件，读取第一个文件，并告诉我它的主要功能是什么"
        )

    finally:
        await client.cleanup()

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
