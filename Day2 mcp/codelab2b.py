# Standard library imports
import os
import asyncio
import base64
from io import BytesIO

# Third-party imports
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

# 添加图像处理库
from PIL import Image


# Load environment variables
load_dotenv()

# Setup and authentication
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Setup and authentication complete.")
    print("✅ ADK components imported successfully.")
except Exception as e:
    print(f"🔑 Authentication Error: {e}")
    raise

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

print("✅ MCP Tool created")

image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)


async def main():
    runner = InMemoryRunner(agent=image_agent)
    try:
        response = await runner.run_debug("Provide a sample tiny image", verbose=True)
        print("Response:", response)
        
        # 提取并显示图像
        for event in response:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    # 检查是否有 function_response
                    if hasattr(part, 'function_response') and part.function_response:
                        fn_response = part.function_response.response
                        if 'content' in fn_response:
                            for item in fn_response['content']:
                                # 找到图像数据
                                if item.get('type') == 'image' and 'data' in item:
                                    try:
                                        # Base64 解码
                                        image_data = base64.b64decode(item['data'])
                                        # 转换为 PIL Image
                                        image = Image.open(BytesIO(image_data))
                                        # 显示图像
                                        image.show()
                                        print(f"✅ 图像已显示: {image.size} pixels, {image.format}")
                                        
                                        # 可选：保存图像
                                        output_path = "tiny_image.png"
                                        image.save(output_path)
                                        print(f"✅ 图像已保存到: {output_path}")
                                    except Exception as e:
                                        print(f"❌ 图像处理错误: {e}")
    finally:
        # 显式关闭 MCP 工具和 Runner
        try:
            # 关闭 MCP 连接
            if hasattr(mcp_image_server, '_session_manager'):
                await mcp_image_server._session_manager.close()
        except Exception:
            pass  # 忽略关闭错误
        
        try:
            # 关闭 Runner
            if hasattr(runner, 'close'):
                await runner.close()
        except Exception:
            pass
        
        # 给清理任务一点时间
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())



