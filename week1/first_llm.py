from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url = os.getenv("BASE_URL"))

def chat_with_llm(user_input):
    """Single-turn conversation with LLM"""
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {str(e)}"

# Test
if __name__ == "__main__":
    print("LLM助手已启动！输入'退出'结束对话。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == '退出':
            break
        response = chat_with_llm(user_input)
        print(f"助手: {response}")
