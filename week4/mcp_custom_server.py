"""
自定义 MCP 服务器实现 - 文件系统工具
提供简单的文件系统操作工具，类似于 function_calling_agent.py 但使用 MCP 协议
"""

import asyncio
import logging
import os
from typing import Any

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import mcp.types as types

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filesystem-server")

# 允许访问的基础路径（安全限制）
ALLOWED_BASE_PATH = os.path.abspath(".")

# ============= 工具函数实现 =============
def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        # 安全检查：确保路径在允许的范围内
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(ALLOWED_BASE_PATH):
            return f"Error: Access denied - path outside allowed directory"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Read file: {file_path} ({len(content)} characters)")
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """写入文件"""
    try:
        # 安全检查
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(ALLOWED_BASE_PATH):
            return f"Error: Access denied - path outside allowed directory"

        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Wrote file: {file_path} ({len(content)} characters)")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {str(e)}")
        return f"Error writing file: {str(e)}"

def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    try:
        # 安全检查
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(ALLOWED_BASE_PATH):
            return f"Error: Access denied - path outside allowed directory"

        items = os.listdir(path)
        if not items:
            return f"Directory '{path}' is empty"

        # 分类文件和目录
        files = []
        dirs = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                dirs.append(f"{item}/")
            else:
                size = os.path.getsize(full_path)
                files.append(f"{item} ({size} bytes)")

        result = f"Contents of '{path}':\n\nDirectories:\n"
        result += "\n".join(dirs) if dirs else "(none)"
        result += "\n\nFiles:\n"
        result += "\n".join(files) if files else "(none)"

        logger.info(f"Listed directory: {path} ({len(items)} items)")
        return result
    except Exception as e:
        logger.error(f"Error listing directory {path}: {str(e)}")
        return f"Error listing directory: {str(e)}"

def search_in_file(file_path: str, keyword: str) -> str:
    """在文件中搜索关键词"""
    try:
        # 安全检查
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(ALLOWED_BASE_PATH):
            return f"Error: Access denied - path outside allowed directory"

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        matches = []
        for i, line in enumerate(lines, 1):
            if keyword in line:
                matches.append(f"Line {i}: {line.strip()}")

        if matches:
            result = f"Found {len(matches)} matches for '{keyword}' in {file_path}:\n" + "\n".join(matches)
        else:
            result = f"No matches found for '{keyword}' in {file_path}"

        logger.info(f"Searched in file: {file_path} for '{keyword}' ({len(matches)} matches)")
        return result
    except Exception as e:
        logger.error(f"Error searching file {file_path}: {str(e)}")
        return f"Error searching file: {str(e)}"

# ============= MCP 服务器 =============
# 创建服务器实例
server = Server("filesystem-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """返回服务器提供的所有工具"""
    return [
        types.Tool(
            name="read_file",
            description="读取指定文件的完整内容。适用于查看文本文件、配置文件、代码文件等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的路径（绝对路径或相对路径）"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="write_file",
            description="将内容写入指定文件。如果文件已存在会覆盖，如果目录不存在会自动创建。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "目标文件的路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        types.Tool(
            name="list_directory",
            description="列出指定目录下的所有文件和子目录，显示文件大小信息。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，默认为当前目录 (.)",
                        "default": "."
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="search_in_file",
            description="在指定文件中搜索包含关键词的所有行，返回行号和内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要搜索的文件路径"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "要搜索的关键词"
                    }
                },
                "required": ["file_path", "keyword"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any]
) -> list[types.TextContent]:
    """处理工具调用请求"""
    try:
        if name == "read_file":
            result = read_file(arguments["file_path"])
        elif name == "write_file":
            result = write_file(arguments["file_path"], arguments["content"])
        elif name == "list_directory":
            path = arguments.get("path", ".")
            result = list_directory(path)
        elif name == "search_in_file":
            result = search_in_file(arguments["file_path"], arguments["keyword"])
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(
            type="text",
            text=result
        )]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

async def main():
    """运行 MCP 服务器"""
    logger.info(f"Starting Filesystem MCP Server...")
    logger.info(f"Allowed base path: {ALLOWED_BASE_PATH}")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="filesystem-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
