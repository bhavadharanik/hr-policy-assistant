import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from src.agent.graph import build_graph
from src.agent.nodes import (
    classify_intent,
    retrieve_policy,
    generate_answer,
    stream_answer,
    lookup_personal_record,
    escalate_out_of_scope,
    check_confidence,
    escalate,
    final_answer,
)
from src.agent.state import AgentState


def _make_initial_state(question: str, employee_id: str = None, conversation_history: list = None) -> AgentState:
    return {
        "question": question,
        "employee_id": employee_id,
        "intent": None,
        "retrieved_docs": [],
        "sources": [],
        "answer": None,
        "confidence": None,
        "escalated": False,
        "escalation_reason": None,
        "conversation_history": conversation_history or [],
        "redacted_question": None,
    }


def run(question: str, employee_id: str = None, conversation_history: list = None) -> AgentState:
    """Run a single question through the graph and print the result."""
    graph = build_graph()
    result = graph.invoke(_make_initial_state(question, employee_id, conversation_history))

    print(f"\nQuestion: {question}")
    print(f"Intent:   {result.get('intent', 'N/A')}")
    if result.get("confidence") is not None:
        print(f"Confidence: {result['confidence']:.2f}")
    print(f"Escalated: {result['escalated']}")
    print(f"Answer: {result['answer']}")
    if result.get("sources"):
        unique_sources = list({s["source"] for s in result["sources"]})
        print(f"Sources: {', '.join(unique_sources)}")
    return result


def interactive() -> None:
    """Multi-turn interactive CLI session with streamed output."""
    print("\n🤖 HR Policy Assistant")
    print("Type your question or 'quit' to exit.\n")

    conversation_history = []
    employee_id = input("Enter your employee ID (optional, press Enter to skip): ").strip() or None

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        state = _make_initial_state(question, employee_id, conversation_history)

        state = classify_intent(state)
        print(f"[intent: {state['intent']}]", end=" ", flush=True)

        if state["intent"] == "policy_lookup":
            state = retrieve_policy(state)
            state = generate_answer(state)
            routing = check_confidence(state)
            if routing == "escalate":
                state = escalate(state)
                print(f"\nAssistant: {state['answer']}")
            else:
                # Presentation layer handles streaming — node stays pure
                stream_answer(state["answer"])
                state = final_answer(state)
                if state.get("confidence"):
                    print(f"[confidence: {state['confidence']:.2f}]")
                if state.get("sources"):
                    unique_sources = list({s["source"] for s in state["sources"]})
                    print(f"Sources: {', '.join(unique_sources)}")
        elif state["intent"] == "personal_record":
            state = lookup_personal_record(state)
            state = final_answer(state)
            print(f"\nAssistant: {state['answer']}")
        else:
            state = escalate_out_of_scope(state)
            state = final_answer(state)
            print(f"\nAssistant: {state['answer']}")

        conversation_history = state.get("conversation_history", [])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        interactive()
