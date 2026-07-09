from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    route_by_intent,
    retrieve_policy,
    generate_answer,
    check_confidence,
    escalate,
    escalate_out_of_scope,
    lookup_personal_record,
    final_answer,
)


def build_graph():
    """Build and compile the HR Policy Assistant graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_policy", retrieve_policy)
    graph.add_node("lookup_personal_record", lookup_personal_record)
    graph.add_node("escalate_out_of_scope", escalate_out_of_scope)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("final_answer", final_answer)
    graph.add_node("escalate", escalate)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Route by intent after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "policy_lookup": "retrieve_policy",
            "personal_record": "lookup_personal_record",
            "out_of_scope": "escalate_out_of_scope",
        }
    )

    # Policy lookup path
    graph.add_edge("retrieve_policy", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        check_confidence,
        {
            "answer": "final_answer",
            "escalate": "escalate",
        }
    )

    # Personal record and out of scope go directly to final_answer
    graph.add_edge("lookup_personal_record", "final_answer")
    graph.add_edge("escalate_out_of_scope", "final_answer")

    graph.add_edge("final_answer", END)
    graph.add_edge("escalate", END)

    return graph.compile()
