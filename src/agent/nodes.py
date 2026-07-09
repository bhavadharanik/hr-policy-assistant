import json
import os
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import AgentState
from src.agent.tools import lookup_employee_record, escalate_to_hr
from src.rag.retriever import similarity_search
from src.config import AI_MODEL, CONFIDENCE_THRESHOLD

# Instantiate once at module level — avoids recreating on every node call
_llm = ChatOpenAI(model=AI_MODEL, temperature=0)


def redact_pii(text: str) -> str:
    """Redact common PII patterns before sending to LangSmith traces."""
    text = re.sub(r'\bEMP\d{3,6}\b', '[EMPLOYEE_ID]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b(\+44|0)\d{10}\b', '[PHONE]', text)
    return text


def _build_prompt(state: AgentState) -> list:
    """Build the messages list for the answer generation call."""
    context = "\n\n".join(state["retrieved_docs"])

    history_text = ""
    if state.get("conversation_history"):
        history_text = "\n\nPrevious conversation:\n"
        for turn in state["conversation_history"][-4:]:
            history_text += f"Employee: {turn['question']}\nAssistant: {turn['answer']}\n"

    system_prompt = """You are an HR Policy Assistant. Answer employee questions based strictly on the provided policy documents.

If the answer is clearly in the context, answer confidently and set confidence high (0.8-1.0).
If the answer is partially covered, answer with caveats and set confidence medium (0.5-0.7).
If the answer is not in the context at all, say so and set confidence low (0.0-0.4).

Always respond in this JSON format:
{
  "answer": "your answer here",
  "confidence": 0.85,
  "reasoning": "brief explanation of confidence level"
}"""

    user_message = f"""Policy Context:
{context}
{history_text}
Employee Question: {state['question']}"""

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]


def _parse_llm_response(content: str) -> tuple[str, float]:
    """Parse JSON response from LLM. Returns (answer, confidence)."""
    try:
        parsed = json.loads(content)
        return parsed["answer"], parsed["confidence"]
    except Exception:
        return content, 0.5


def classify_intent(state: AgentState) -> AgentState:
    """Classify the intent of the question to route to the right handler."""
    system_prompt = """Classify the employee's HR question into one of three categories:

- "policy_lookup": questions about company policies, rules, entitlements, procedures (e.g. "how many days leave do I get?", "can I expense a client dinner?")
- "personal_record": questions about the employee's own specific data (e.g. "what is my leave balance?", "when did I start?") - requires looking up their personal record
- "out_of_scope": questions completely unrelated to HR (e.g. "what is the weather?", "write me code")

Respond with ONLY the category name, nothing else."""

    response = _llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ])

    intent = response.content.strip().lower()
    if intent not in ["policy_lookup", "personal_record", "out_of_scope"]:
        intent = "policy_lookup"

    state["intent"] = intent
    state["redacted_question"] = redact_pii(state["question"])
    return state


def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    return state.get("intent", "policy_lookup")


def retrieve_policy(state: AgentState) -> AgentState:
    """Retrieve relevant policy chunks for the question, capturing source metadata."""
    docs = similarity_search(state["question"], k=4)
    state["retrieved_docs"] = [doc.page_content for doc in docs]
    state["sources"] = [
        {
            "content": doc.page_content,
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "chunk_index": i,
        }
        for i, doc in enumerate(docs)
    ]
    return state


def lookup_personal_record(state: AgentState) -> AgentState:
    """Look up employee's personal record for personal queries."""
    employee_id = state.get("employee_id")

    if not employee_id:
        state["answer"] = "To look up your personal record, please provide your employee ID (e.g. EMP001)."
        state["confidence"] = 1.0
        state["escalated"] = False
        return state

    result = lookup_employee_record.invoke({"employee_id": employee_id})

    if "error" in result:
        state["answer"] = f"I couldn't find a record for employee ID {employee_id}. Please check your ID or contact HR directly."
    else:
        name = result.get("name", "Unknown")
        dept = result.get("department", "Unknown")
        leave_balance = result.get("leave_balance", "Unknown")
        start_date = result.get("start_date", "Unknown")
        state["answer"] = (
            f"Here are your details:\n"
            f"- Name: {name}\n"
            f"- Department: {dept}\n"
            f"- Start Date: {start_date}\n"
            f"- Remaining Leave Balance: {leave_balance} days"
        )

    state["confidence"] = 1.0
    state["escalated"] = False
    return state


def escalate_out_of_scope(state: AgentState) -> AgentState:
    """Handle out of scope questions directly without RAG."""
    state["answer"] = (
        "I'm an HR Policy Assistant and can only help with HR-related questions "
        "such as leave policies, expenses, and workplace conduct. "
        "For other queries, please contact the appropriate team."
    )
    state["confidence"] = 1.0
    state["escalated"] = False
    return state


def generate_answer(state: AgentState) -> AgentState:
    """Generate an answer using retrieved policy context and conversation history."""
    messages = _build_prompt(state)
    response = _llm.invoke(messages)
    state["answer"], state["confidence"] = _parse_llm_response(response.content)
    return state


def stream_answer(answer: str) -> None:
    """Stream answer text to stdout character by character.

    Separated from generate_answer so that nodes remain pure state transformers
    and presentation concerns stay in the CLI layer.
    """
    print("\nAssistant: ", end="", flush=True)
    for char in answer:
        print(char, end="", flush=True)
        time.sleep(0.01)
    print()


def check_confidence(state: AgentState) -> str:
    """Route: answer directly or escalate based on confidence."""
    if state.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
        return "answer"
    return "escalate"


def escalate(state: AgentState) -> AgentState:
    """Escalate to human HR when confidence is too low."""
    result = escalate_to_hr.invoke({
        "reason": f"Low confidence answer (confidence: {state.get('confidence', 0):.2f})",
        "question": state["redacted_question"] or state["question"],
        "employee_id": state.get("employee_id"),
    })
    state["escalated"] = True
    state["escalation_reason"] = result["message"]
    state["answer"] = result["message"]
    return state


def append_to_history(state: AgentState) -> AgentState:
    """Append the current Q&A turn to conversation history."""
    if not state.get("conversation_history"):
        state["conversation_history"] = []
    state["conversation_history"].append({
        "question": state["question"],
        "answer": state["answer"],
    })
    return state


def final_answer(state: AgentState) -> AgentState:
    """Mark answer as not escalated and persist to conversation history."""
    state["escalated"] = False
    return append_to_history(state)
