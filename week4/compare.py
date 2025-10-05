"""
对比 ReAct Agent 和 Function Calling Agent 的性能
比较指标：代码行数、token消耗、响应时间、成功率
"""

import sys
import os
import time
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from io import StringIO
from unittest.mock import patch

# 加载环境变量
load_dotenv()

# 添加week2路径以导入ReAct agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'week2'))
from agent import ReActAgent, read_file as react_read_file, write_to_file as react_write_to_file

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 初始化 tiktoken encoder
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    """使用 tiktoken 计算 token 数量"""
    return len(encoding.encode(text))

# ============= ReAct Agent 工具（与week2保持一致）=============
def list_files(directory="."):
    """列出目录下的所有文件"""
    items = os.listdir(directory)
    return "\n".join(items)

# ReAct Agent 工具集
react_tools = [react_read_file, react_write_to_file, list_files]

# ============= ReAct Agent Token 计数包装器 =============
class ReActAgentWithTokenCounting(ReActAgent):
    """扩展 ReActAgent 以支持 token 统计"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens = 0
        self.messages_log = []

    def call_model(self, messages):
        """重写 call_model 方法以统计 tokens"""
        # 统计输入 tokens
        for msg in messages:
            if msg not in self.messages_log:
                self.total_tokens += count_tokens(msg["content"])
                self.messages_log.append(msg)

        # 调用原始方法
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})

        # 统计输出 tokens
        self.total_tokens += count_tokens(content)
        self.messages_log.append({"role": "assistant", "content": content})

        return content

    def get_total_tokens(self):
        """获取总 token 数"""
        return self.total_tokens


# ============= Function Calling Agent =============
def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """写入文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    try:
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Function Calling工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的绝对路径或相对路径"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件的绝对路径或相对路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的所有文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，默认为当前目录"
                    }
                },
                "required": []
            }
        }
    }
]

# 工具函数映射
available_functions = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory
}

def function_calling_agent(query: str, max_iterations: int = 5) -> dict:
    """使用Function Calling实现的Agent"""
    messages = [{"role": "user", "content": query}]
    total_tokens = 0
    tool_calls_count = 0

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        total_tokens += response.usage.total_tokens
        assistant_message = response.choices[0].message

        # 如果没有工具调用，说明任务完成
        if not assistant_message.tool_calls:
            return {
                "success": True,
                "response": assistant_message.content,
                "tokens": total_tokens,
                "tool_calls": tool_calls_count,
                "iterations": iteration + 1
            }

        # 执行工具调用
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_calls_count += 1
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # 执行函数
            function_response = available_functions[function_name](**function_args)

            # 添加工具结果到消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_response
            })

    return {
        "success": False,
        "response": "Max iterations reached",
        "tokens": total_tokens,
        "tool_calls": tool_calls_count,
        "iterations": max_iterations
    }

# ============= 测试用例 =============
test_cases = [
    {
        "name": "简单文件读取",
        "query": "读取 test_data.txt 文件的内容",
        "setup": lambda: write_file("test_data.txt", "Hello, World!\nThis is a test file.")
    },
    {
        "name": "文件修改",
        "query": "读取 test_data.txt 并在开头添加一行 '# Modified'",
        "setup": lambda: write_file("test_data.txt", "Original content\nLine 2")
    },
    {
        "name": "目录浏览",
        "query": "列出当前目录下的所有文件",
        "setup": lambda: None
    }
]

def run_comparison():
    """运行对比测试"""
    print("=" * 80)
    print("ReAct Agent vs Function Calling Agent 对比测试")
    print("=" * 80)

    results = []

    for test_case in test_cases:
        print(f"\n📝 测试用例: {test_case['name']}")
        print(f"   查询: {test_case['query']}")
        print("-" * 80)

        # 准备测试环境
        if test_case['setup']:
            test_case['setup']()

        # 测试 ReAct Agent
        print("\n🔧 ReAct Agent:")
        react_start = time.time()
        react_tokens = 0
        react_tool_calls = 0

        try:
            # 创建 ReAct Agent 实例（使用带 token 统计的版本）
            react_agent = ReActAgentWithTokenCounting(
                tools=react_tools,
                model="gpt-4o-mini",
                project_directory=os.getcwd()
            )

            # 捕获输出并自动回答 "y"（跳过用户交互）
            with patch('builtins.input', return_value='y'):
                # 使用 StringIO 捕获打印输出以避免干扰
                output = StringIO()
                with patch('sys.stdout', output):
                    react_result = react_agent.run(test_case['query'])

            react_time = time.time() - react_start
            react_success = True

            # 获取 token 统计
            react_tokens = react_agent.get_total_tokens()
            output_text = output.getvalue()
            react_tool_calls = output_text.count("🔧 Action:")

            print(f"   ✓ 完成 (耗时: {react_time:.2f}s)")
            print(f"   Tokens: {react_tokens}")
            print(f"   Tool calls: {react_tool_calls}")

        except Exception as e:
            react_time = time.time() - react_start
            react_success = False
            react_result = {"response": str(e)}
            print(f"   ✗ 失败: {str(e)}")

        # 测试 Function Calling Agent
        print("\n🚀 Function Calling Agent:")
        fc_start = time.time()
        try:
            fc_result = function_calling_agent(test_case['query'])
            fc_time = time.time() - fc_start
            print(f"   ✓ 完成 (耗时: {fc_time:.2f}s)")
            print(f"   Tokens: {fc_result['tokens']}")
            print(f"   Tool calls: {fc_result['tool_calls']}")
        except Exception as e:
            fc_time = time.time() - fc_start
            fc_result = {"success": False, "tokens": 0, "tool_calls": 0}
            print(f"   ✗ 失败: {str(e)}")

        # 记录结果
        results.append({
            "test_case": test_case['name'],
            "react": {
                "success": react_success,
                "time": react_time,
                "tokens": react_tokens,
                "tool_calls": react_tool_calls
            },
            "function_calling": {
                "success": fc_result.get('success', False),
                "time": fc_time,
                "tokens": fc_result.get('tokens', 0),
                "tool_calls": fc_result.get('tool_calls', 0)
            }
        })

    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)

    for result in results:
        print(f"\n{result['test_case']}:")
        print(f"  ReAct: {'✓' if result['react']['success'] else '✗'} "
              f"({result['react']['time']:.2f}s, {result['react']['tokens']} tokens, {result['react']['tool_calls']} tool calls)")
        print(f"  Function Calling: {'✓' if result['function_calling']['success'] else '✗'} "
              f"({result['function_calling']['time']:.2f}s, "
              f"{result['function_calling']['tokens']} tokens, "
              f"{result['function_calling']['tool_calls']} tool calls)")

    # 计算平均值
    react_avg_tokens = sum(r['react']['tokens'] for r in results) / len(results)
    fc_avg_tokens = sum(r['function_calling']['tokens'] for r in results) / len(results)
    fc_avg_tools = sum(r['function_calling']['tool_calls'] for r in results) / len(results)
    react_avg_tools = sum(r['react']['tool_calls'] for r in results) / len(results)

    fc_success_rate = sum(1 for r in results if r['function_calling']['success']) / len(results)
    react_success_rate = sum(1 for r in results if r['react']['success']) / len(results)

    print(f"\n平均指标:")
    print(f"  ReAct Agent:")
    print(f"    - 成功率: {react_success_rate*100:.0f}%")
    print(f"    - 平均 tokens: {react_avg_tokens:.0f} (tiktoken)")
    print(f"    - 平均工具调用: {react_avg_tools:.1f} 次")
    print(f"\n  Function Calling Agent:")
    print(f"    - 成功率: {fc_success_rate*100:.0f}%")
    print(f"    - 平均 tokens: {fc_avg_tokens:.0f} (OpenAI API)")
    print(f"    - 平均工具调用: {fc_avg_tools:.1f} 次")

    print(f"\n性能对比:")
    if fc_success_rate > react_success_rate:
        print(f"  ✓ Function Calling 成功率更高 (+{(fc_success_rate - react_success_rate)*100:.0f}%)")

    # Token 对比
    token_diff = react_avg_tokens - fc_avg_tokens
    token_diff_pct = (token_diff / react_avg_tokens) * 100 if react_avg_tokens > 0 else 0
    if token_diff > 0:
        print(f"  ✓ Function Calling tokens 更少 (-{token_diff:.0f} tokens, {token_diff_pct:.1f}% 节省)")
    else:
        print(f"  ⚠ ReAct tokens 更少 ({abs(token_diff):.0f} tokens, {abs(token_diff_pct):.1f}% 节省)")

    print(f"  ✓ Function Calling 代码更简洁（~120行 vs ~220行）")

    # 清理测试文件
    if os.path.exists("test_data.txt"):
        os.remove("test_data.txt")

if __name__ == "__main__":
    run_comparison()
