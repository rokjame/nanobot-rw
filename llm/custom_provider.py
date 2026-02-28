from typing import Any

from openai import AsyncOpenAI, OpenAI
import json_repair
from llm.LLMBase import LLMBase, LLMResponse, ToolCallRequest


class CustomProvider(LLMBase):
    def __init__(self, api_key : str, base_url : str, default_model : str):
        super().__init__(api_key, base_url)
        self.default_model = default_model
        self._client = OpenAI(api_key = api_key, base_url = base_url)

    def chat(
            self,
            message: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str = "gpt-3.5-turbo",
            max_tokens: int = 8192,
            temperature: float = 0.7,
    ) -> LLMResponse:
        # 验证和清理消息格式
        cleaned_messages = self._validate_messages(message)
        
        kwargs : dict[str, Any] = {
            "model": model or self.default_model,
            "messages": cleaned_messages,
            "max_tokens": max(1, max_tokens),
            "temperature": max(0, min(1, temperature)),
        }
        
        # 判断是否应该使用工具
        if tools:
            # 验证工具格式
            validated_tools = self._validate_tools(tools)
            kwargs["tools"] = validated_tools
            
        try:
            response = self._client.chat.completions.create(**kwargs)
            return self._parse(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.default_model

    def _validate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """验证和清理消息格式"""
        validated = []
        for msg in messages:
            # 确保必需字段存在
            if not isinstance(msg, dict):
                continue
            
            validated_msg = {
                "role": str(msg.get("role", "user")).lower(),
                "content": str(msg.get("content", ""))
            }
            
            # 添加其他可能的字段
            if "name" in msg:
                validated_msg["name"] = str(msg["name"])
            
            validated.append(validated_msg)
        
        return validated

    def _validate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """验证工具格式"""
        validated = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            
            # 确保工具的基本结构正确
            if "type" in tool and "function" in tool:
                validated.append(tool)
        
        return validated

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )