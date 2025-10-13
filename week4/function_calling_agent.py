"""
Function Calling Agent 完整实现
演示如何使用 OpenAI 的 Function Calling API 构建一个文件系统操作 Agent
"""

import json
import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# ============= 工具函数实现 =============
def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """写入文件"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    try:
        items = os.listdir(path)
        if not items:
            return f"Directory '{path}' is empty"

        # 分类文件和目录
        files = []
        dirs = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                dirs.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full_path)
                files.append(f"📄 {item} ({size} bytes)")

        result = f"Contents of '{path}':\n"
        if dirs:
            result += "\nDirectories:\n" + "\n".join(dirs)
        if files:
            result += "\n\nFiles:\n" + "\n".join(files)

        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def search_in_file(file_path: str, keyword: str) -> str:
    """在文件中搜索关键词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        matches = []
        for i, line in enumerate(lines, 1):
            if keyword in line:
                matches.append(f"Line {i}: {line.strip()}")

        if matches:
            return f"Found {len(matches)} matches in {file_path}:\n" + "\n".join(matches)
        else:
            return f"No matches found for '{keyword}' in {file_path}"
    except Exception as e:
        return f"Error searching file: {str(e)}"

# ============= 工具定义 (JSON Schema) =============
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的完整内容。适用于查看文本文件、配置文件、代码文件等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的路径（绝对路径或相对路径）"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件。如果文件已存在会覆盖，如果目录不存在会自动创建。",
            "parameters": {
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
                "required": ["file_path", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的所有文件和子目录，显示文件大小信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，默认为当前目录 (.)"
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "在指定文件中搜索包含关键词的所有行，返回行号和内容。",
            "parameters": {
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
                "required": ["file_path", "keyword"],
                "additionalProperties": False
            }
        }
    }
]

# ============= 工具映射 =============
available_functions = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "search_in_file": search_in_file
}

# ============= Function Calling Agent =============
class FunctionCallingAgent:
    def __init__(self, model: str = "gpt-5-mini", verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.tools = tools
        self.available_functions = available_functions

    def run(self, user_query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        运行 Agent 处理用户查询

        Args:
            user_query: 用户的问题或任务
            max_iterations: 最大迭代次数

        Returns:
            包含结果和统计信息的字典
        """
        messages = [{"role": "user", "content": user_query}]
        total_tokens = 0
        tool_calls_count = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 Function Calling Agent")
            print(f"{'='*60}")
            print(f"📝 User Query: {user_query}\n")

        for iteration in range(max_iterations):
            if self.verbose:
                print(f"--- Iteration {iteration + 1} ---")

            # 调用 API
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            total_tokens += response.usage.total_tokens
            assistant_message = response.choices[0].message

            # 检查是否需要调用工具
            if not assistant_message.tool_calls:
                # 任务完成
                final_response = assistant_message.content
                if self.verbose:
                    print(f"\n✅ Final Response:\n{final_response}")
                    print(f"\n📊 Statistics:")
                    print(f"   - Total tokens used: {total_tokens}")
                    print(f"   - Tool calls made: {tool_calls_count}")
                    print(f"   - Iterations: {iteration + 1}")

                return {
                    "success": True,
                    "response": final_response,
                    "tokens": total_tokens,
                    "tool_calls": tool_calls_count,
                    "iterations": iteration + 1
                }

            # 添加 assistant 消息到历史
            messages.append(assistant_message)

            # 执行所有工具调用
            for tool_call in assistant_message.tool_calls:
                tool_calls_count += 1
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if self.verbose:
                    print(f"\n🔧 Tool Call #{tool_calls_count}:")
                    print(f"   Function: {function_name}")
                    print(f"   Arguments: {json.dumps(function_args, ensure_ascii=False)}")

                # 执行函数
                function_response = self.available_functions[function_name](**function_args)

                if self.verbose:
                    # 截断长输出
                    display_response = function_response[:200] + "..." if len(function_response) > 200 else function_response
                    print(f"   Result: {display_response}")

                # 添加工具结果到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_response
                })

        # 达到最大迭代次数
        if self.verbose:
            print(f"\n⚠️  Reached maximum iterations ({max_iterations})")

        return {
            "success": False,
            "response": "Maximum iterations reached without completion",
            "tokens": total_tokens,
            "tool_calls": tool_calls_count,
            "iterations": max_iterations
        }

# ============= 示例用法 =============
def main():
    agent = FunctionCallingAgent(verbose=True)

    # 测试案例 1: 简单文件读取
    print("\n" + "="*60)
    print("测试案例 1: 文件读取")
    print("="*60)
    agent.run("读取 README.md 文件的内容")

    # 测试案例 2: 文件修改
    print("\n" + "="*60)
    print("测试案例 2: 文件修改")
    print("="*60)
    agent.run("读取 test.txt 文件，在开头添加 '# Test File'，然后保存")

    # 测试案例 3: 复杂任务
    print("\n" + "="*60)
    print("测试案例 3: 多步骤任务")
    print("="*60)
    agent.run("列出当前目录的所有文件，找到所有 .py 文件，并在第一个 Python 文件中搜索 'import' 关键词")

if __name__ == "__main__":
    main()
