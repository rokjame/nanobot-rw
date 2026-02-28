from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning_content: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

"""
大模型调用基类
"""
class LLMBase(ABC):

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def chat(
            self,
            # 消息
            message: list[dict[str, Any]],
            # 工具
            tools: list[dict[str, Any]] | None = None,
            # 模型
            model: str | None = None,
            max_tokens: int = 8192,
            temperature: float = 0.7,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
