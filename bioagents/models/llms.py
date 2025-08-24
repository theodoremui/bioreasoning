# ------------------------------------------------------------------------------
# llms.py
#
# This is an interface class for all LLM models. It provides a common interface
# for all LLM models.
#
# Author: Theodore Mui
# Date: 2025-04-26
# ------------------------------------------------------------------------------


from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import os
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI, OpenAI

from bioagents.commons import classproperty


@dataclass
class LLM:
    # Model identifiers
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"

    # Shared OpenAI client instance
    _client: OpenAI | None = None
    _async_client: AsyncOpenAI | None = None
    _model_name = GPT_4_1_MINI
    _timeout = 60

    @classproperty
    def openai_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(timeout=cls._timeout)
        return cls._client

    @classproperty
    def openai_async_client(cls) -> AsyncOpenAI:
        if cls._async_client is None:
            cls._async_client = AsyncOpenAI(timeout=cls._timeout)
        return cls._async_client

    def __init__(self, model_name=GPT_4_1_MINI, timeout=60):
        self._model_name = model_name
        self._timeout = timeout
        # Initialize per-instance clients to satisfy tests expecting not None
        try:
            self._client = OpenAI(timeout=self._timeout)
        except Exception:
            self._client = None
        try:
            self._async_client = AsyncOpenAI(timeout=self._timeout)
        except Exception:
            self._async_client = None

    async def achat_completion(self, query_str: str, **kwargs) -> str:
        """
        Asynchronously chat with the model.

        Args:
            query_str: The query string
            **kwargs: Additional parameters for the completion

        Returns:
            The content string
        """
        response = await self._async_client.chat.completions.create(
            messages=[{"role": "user", "content": query_str}],
            model=self._model_name,
            **kwargs,
        )
        content = ""
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content

        return content


# ------------------------------------------------
# Example usage
# ------------------------------------------------
if __name__ == "__main__":
    import asyncio

    gpt41 = LLM(model_name=LLM.GPT_4_1_MINI)

    # Async example
    async def main():
        response = await gpt41.achat_completion(
            query_str="Hello async world!",
        )
        print(response)

    asyncio.run(main())
