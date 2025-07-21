from dotenv import load_dotenv
import json
import os

from pydantic import BaseModel, Field, model_validator
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAIResponses
from typing import List, Tuple, Optional
from typing_extensions import Self

load_dotenv()


class ClaimVerification(BaseModel):
    claim_is_true: bool = Field(
        description="Based on the provided sources information, the claim passes or not."
    )
    supporting_citations: Optional[List[str]] = Field(
        description="A minimum of one and a maximum of three citations from the sources supporting the claim. If the claim is not supported, please leave empty",
        default=None,
        min_length=1,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_claim_ver(self) -> Self:
        if not self.claim_is_true and self.supporting_citations is not None:
            self.supporting_citations = ["The claim was deemed false."]
        return self


if os.getenv("OPENAI_API_KEY", None):
    LLM = OpenAIResponses(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
    LLM_VERIFIER = LLM.as_structured_llm(ClaimVerification)


def verify_claim(
    claim: str,
    sources: str,
) -> Tuple[bool, Optional[List[str]]]:
    response = LLM_VERIFIER.chat(
        [
            ChatMessage(
                role="user",
                content=f"I have this claim: {claim} that is allegedgly supported by these sources:\n\n'''\n{sources}\n'''\n\nCan you please tell me whether or not this claim is thrutful and, if it is, identify one to three passages in the sources specifically supporting the claim?",
            )
        ]
    )
    response_json = json.loads(response.message.content)
    return response_json["claim_is_true"], response_json["supporting_citations"]
