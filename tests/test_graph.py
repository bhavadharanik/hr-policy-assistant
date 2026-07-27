import pytest
from unittest.mock import MagicMock, patch
from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.agent.nodes import redact_pii, check_confidence
from src.agent.schemas import PolicyAnswer


def make_state(**overrides) -> AgentState:
    base: AgentState = {
        "question": "How many days of leave do I get?",
        "employee_id": None,
        "intent": None,
        "retrieved_docs": [],
        "sources": [],
        "answer": None,
        "confidence": None,
        "escalated": False,
        "escalation_reason": None,
        "conversation_history": [],
        "redacted_question": None,
    }
    base.update(overrides)
    return base


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_check_confidence_routes_to_answer_when_high():
    state = make_state(confidence=0.9)
    assert check_confidence(state) == "answer"


def test_check_confidence_routes_to_escalate_when_low():
    state = make_state(confidence=0.3)
    assert check_confidence(state) == "escalate"


def test_check_confidence_escalates_at_threshold_boundary():
    state = make_state(confidence=0.69)
    assert check_confidence(state) == "escalate"

    state = make_state(confidence=0.70)
    assert check_confidence(state) == "answer"


def test_redact_pii_removes_employee_id():
    result = redact_pii("Employee EMP001 is asking about leave.")
    assert "EMP001" not in result
    assert "[EMPLOYEE_ID]" in result


def test_redact_pii_removes_email():
    result = redact_pii("Contact alice@company.com for details.")
    assert "alice@company.com" not in result
    assert "[EMAIL]" in result


def test_redact_pii_removes_uk_phone():
    result = redact_pii("Call 07911123456 for support.")
    assert "07911123456" not in result
    assert "[PHONE]" in result


def test_redact_pii_leaves_clean_text_unchanged():
    text = "How many days of annual leave do I get?"
    assert redact_pii(text) == text


def test_initial_state_defaults():
    state = make_state()
    assert state["escalated"] == False
    assert state["sources"] == []
    assert state["conversation_history"] == []


def test_full_policy_lookup_flow():
    """End-to-end: policy question routes through RAG and returns an answer."""
    fake_doc = MagicMock()
    fake_doc.page_content = "Employees are entitled to 25 days of annual leave per year."
    fake_doc.metadata = {"source": "leave_policy.pdf"}

    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "policy_lookup"

    fake_answer = PolicyAnswer(
        answer="You get 25 days of annual leave.",
        confidence=0.95,
        reasoning="Clearly stated in policy.",
    )

    with patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.agent.nodes._structured_llm") as mock_structured, \
         patch("src.agent.nodes.similarity_search", return_value=[fake_doc]):
        mock_llm.invoke.return_value = fake_llm_intent
        mock_structured.invoke.return_value = fake_answer

        graph = build_graph()
        result = graph.invoke(make_state(question="How many days of leave do I get?"))

    assert result["intent"] == "policy_lookup"
    assert result["escalated"] == False
    assert "25 days" in result["answer"]
    assert result["confidence"] >= 0.7
    assert len(result["conversation_history"]) == 1


def test_out_of_scope_does_not_call_retriever():
    """Out-of-scope questions are rejected without touching RAG."""
    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "out_of_scope"

    with patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.agent.nodes.similarity_search") as mock_retriever:
        mock_llm.invoke.return_value = fake_llm_intent

        graph = build_graph()
        result = graph.invoke(make_state(question="What is the weather today?"))

    mock_retriever.assert_not_called()
    assert result["escalated"] == False
    assert result["answer"] is not None


def test_low_confidence_triggers_escalation():
    """Answers below confidence threshold are escalated to HR."""
    fake_doc = MagicMock()
    fake_doc.page_content = "Some vaguely related policy text."
    fake_doc.metadata = {"source": "policy.pdf"}

    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "policy_lookup"

    fake_answer = PolicyAnswer(
        answer="I'm not sure about this.",
        confidence=0.3,
        reasoning="Context does not clearly address this question.",
    )

    with patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.agent.nodes._structured_llm") as mock_structured, \
         patch("src.agent.nodes.similarity_search", return_value=[fake_doc]):
        mock_llm.invoke.return_value = fake_llm_intent
        mock_structured.invoke.return_value = fake_answer

        graph = build_graph()
        result = graph.invoke(make_state(question="What is the policy on cryptocurrency trading?"))

    assert result["escalated"] == True


def test_empty_retrieval_still_returns_answer():
    """When RAG returns no docs, generate_answer still runs without crashing."""
    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "policy_lookup"

    fake_answer = PolicyAnswer(
        answer="I could not find relevant policy information.",
        confidence=0.2,
        reasoning="No matching documents retrieved.",
    )

    with patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.agent.nodes._structured_llm") as mock_structured, \
         patch("src.agent.nodes.similarity_search", return_value=[]):
        mock_llm.invoke.return_value = fake_llm_intent
        mock_structured.invoke.return_value = fake_answer

        graph = build_graph()
        result = graph.invoke(make_state(question="What is the parental leave policy?"))

    assert result["answer"] is not None
    assert result["confidence"] < 0.5


def test_personal_record_without_employee_id():
    """Personal record questions without an ID return a helpful prompt."""
    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "personal_record"

    with patch("src.agent.nodes._llm") as mock_llm:
        mock_llm.invoke.return_value = fake_llm_intent

        graph = build_graph()
        result = graph.invoke(make_state(question="What is my leave balance?", employee_id=None))

    assert result["escalated"] == False
    assert "employee ID" in result["answer"]


def test_llm_error_propagates():
    """LLM errors are not silently swallowed."""
    fake_llm_intent = MagicMock()
    fake_llm_intent.content = "policy_lookup"

    fake_doc = MagicMock()
    fake_doc.page_content = "Some policy text."
    fake_doc.metadata = {"source": "policy.pdf"}

    with patch("src.agent.nodes._llm") as mock_llm, \
         patch("src.agent.nodes._structured_llm") as mock_structured, \
         patch("src.agent.nodes.similarity_search", return_value=[fake_doc]):
        mock_llm.invoke.return_value = fake_llm_intent
        mock_structured.invoke.side_effect = RuntimeError("OpenAI API error")

        graph = build_graph()
        with pytest.raises(RuntimeError):
            graph.invoke(make_state(question="How many days of leave do I get?"))
