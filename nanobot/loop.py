import time
from typing import Any

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

    def run(self, message: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> None:
        self._running = True
        logger.info("Agent循环开始！")
        i = 0
        while self._running and i < self.max_iterations:
            try:
                msg = self.provider.chat(
                    message=message,
                    tools=tools,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                logger.info(f"{msg}")
                logger.info(f"Agent循环，第{i}次，输出：{msg.content}")
                
                # 检查是否应该停止
                if msg.content and msg.content.strip().lower() == "/stop":
                    logger.info("Agent循环结束！")
                    self._running = False
                    break
                
            except Exception as e:
                logger.error(f"Agent循环错误：{e}")
            finally:
                i += 1
                if i < self.max_iterations:
                    time.sleep(2)