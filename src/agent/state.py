from typing import TypedDict, Optional, List, Literal


class SourceChunk(TypedDict):
    content: str
    source: str  # filename
    chunk_index: int


class AgentState(TypedDict):
    question: str
    employee_id: Optional[str]
    intent: Optional[Literal["policy_lookup", "personal_record", "out_of_scope"]]
    retrieved_docs: List[str]
    sources: List[SourceChunk]
    answer: Optional[str]
    confidence: Optional[float]
    escalated: bool
    escalation_reason: Optional[str]
    conversation_history: List[dict]
    redacted_question: Optional[str]
