from pathlib import Path

from llm.custom_provider import CustomProvider
from nanobot.loop import AgentLoop

def main():
    # 初始化参数
    api_key = "ms-21a86c2d-7f00-47d7-8660-21b577037a02"  # 替换为实际的API密钥
    base_url = "https://api-inference.modelscope.cn/v1"  # 替换为实际的基础URL
    model = "Qwen/Qwen3.5-397B-A17B"
    workspace_path = Path("./workspace")
    
    # 确保工作目录存在
    workspace_path.mkdir(exist_ok=True)
    
    # 创建LLM提供者实例
    provider = CustomProvider(api_key, base_url, model)

    message = [
        {
            "role": "system",
            "content": "你是一个助手，你需要帮助用户完成任务。"
        },
        {
            "role": "user", 
            "content": "帮我查下今天成都的天气情况"
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询的城市名称",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    # 创建AgentLoop实例
    agent_loop = AgentLoop(
        provider=provider,
        wrokspace=workspace_path,
        model=model,
        max_iterations=5,  # 限制迭代次数用于测试
        max_tokens=1000,
        temperature=0.7
    )
    
    # 运行agent循环
    print("开始运行Agent循环...")
    try:
        agent_loop.run(initial_messages=message, tools=tools)
    except KeyboardInterrupt:
        print("\n用户中断，停止Agent循环")
    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == '__main__':
    main()
