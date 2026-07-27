"""Pydantic schemas for structured LLM output — no hand-parsed JSON."""

from pydantic import BaseModel, Field


class PolicyAnswer(BaseModel):
    """Structured response from the HR policy LLM."""

    answer: str = Field(description="Direct answer to the employee's question based on policy context.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the retrieved context supports this answer. 0.8-1.0 if clearly in policy, 0.5-0.7 if partial, 0.0-0.4 if not found.",
    )
    reasoning: str = Field(description="Brief explanation of why this confidence level was chosen.")
