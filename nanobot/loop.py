import time
from typing import Any

import json

from llm.LLMBase import LLMBase
from pathlib import Path
from loguru import logger


class AgentLoop:
    def __init__(
            self,
            provider: LLMBase,
            wrokspace: Path,
            model: str = "gpt-3.5-turbo",
            max_iterations: int = 10,
            max_tokens: int = 8192,
            temperature: float = 0.7
    ):
        self.provider = provider
        self.wrokspace = wrokspace
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._running = False

    def run(self, initial_messages: list[dict]
            , tools: list[dict[str, Any]] | None = None) -> None:
        self._running = True
        logger.info("Agent循环开始！")
        i = 0
        messages = initial_messages
        tools_used: list[str] = []
        final_content = None

        while self._running and i < self.max_iterations:
            try:
                response = self.provider.chat(
                    message=messages,
                    tools=tools,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                logger.info(f"{response}")
                logger.info(f"Agent循环，第{i}次，输出：{response.content}")

                if response.has_tool_calls():
                    # 先添加带有tool_calls的助手消息
                    tool_calls_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                            }
                        }
                        for tc in response.tool_calls
                    ]
                    messages = self.add_assistant_message(
                        messages, response.content, tool_calls_dicts,
                        reasoning_content=response.reasoning_content,
                    )
                    
                    # 然后为每个工具调用添加结果
                    for tool_call in response.tool_calls:
                        tools_used.append(tool_call.name)
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                        result = "37度，天气晴朗"
                        messages = self.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )
                    
                    # 继续下一次循环，让模型处理工具结果
                    continue
                else:
                    clean = response.content
                    messages = self.add_assistant_message(
                        messages, clean, reasoning_content=response.reasoning_content,
                    )
                    final_content = clean
                    
                    # 检查是否应该停止
                    if final_content and final_content.strip().lower() == "/stop":
                        logger.info("Agent循环结束！")
                        self._running = False
                        break
                    
            except Exception as e:
                logger.error(f"Agent循环错误：{e}")
            finally:
                i += 1
                if i < self.max_iterations:
                    time.sleep(2)

    def add_assistant_message(
            self, messages: list[dict[str, Any]],
            content: str | None,
            tool_calls: list[dict[str, Any]] | None = None,
            reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages