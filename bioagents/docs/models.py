from pydantic import BaseModel, Field, model_validator
from typing import List
from typing_extensions import Self


class Notebook(BaseModel):
    summary: str = Field(
        description="Summary of the document.",
    )
    highlights: List[str] = Field(
        description="Highlights of the documents: 3 to 10 bullet points that represent the crucial knots of the documents.",
        min_length=3,
        max_length=10,
    )
    questions: List[str] = Field(
        description="5 to 15 questions based on the topic of the document.",
        examples=[
            [
                "What is the capital of Spain?",
                "What is the capital of France?",
                "What is the capital of Italy?",
                "What is the capital of Portugal?",
                "What is the capital of Germany?",
            ]
        ],
        min_length=5,
        max_length=15,
    )
    answers: List[str] = Field(
        description="Answers to the questions reported in the 'questions' field, in the same order.",
        examples=[
            [
                "Madrid",
                "Paris",
                "Rome",
                "Lisbon",
                "Berlin",
            ]
        ],
        min_length=5,
        max_length=15,
    )

    @model_validator(mode="after")
    def validate_q_and_a(self) -> Self:
        if len(self.questions) != len(self.answers):
            raise ValueError("Questions and Answers must be of the same length.")
        return self
