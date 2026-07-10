import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import build_graph
from src.agent.state import AgentState


def log_to_langsmith(results: list, passed: int, total: int, intent_correct: int):
    """Log eval results to LangSmith as a dataset run."""
    try:
        from langsmith import Client
        client = Client()

        dataset_name = "hr-policy-assistant-evals"
        run_name = f"eval-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create or get dataset
        try:
            dataset = client.create_dataset(dataset_name, description="HR Policy Assistant eval dataset")
        except Exception:
            dataset = client.read_dataset(dataset_name=dataset_name)

        # Log each result as an example + run
        for result in results:
            try:
                client.create_example(
                    inputs={"question": result["question"]},
                    outputs={
                        "answer": result["answer_preview"],
                        "intent": result["intent"],
                        "escalated": result.get("escalation_correct"),
                    },
                    dataset_id=dataset.id,
                )
            except Exception:
                pass  # Example may already exist

        print(f"\n[LangSmith] Eval results logged to dataset: '{dataset_name}'")
        print(f"[LangSmith] View at: https://smith.langchain.com")

    except Exception as e:
        print(f"\n[LangSmith] Could not log results: {e}")


def run_evals():
    graph = build_graph()

    with open(os.path.join(os.path.dirname(__file__), "dataset.json")) as f:
        dataset = json.load(f)

    results = []
    passed = 0
    intent_correct_count = 0

    for i, case in enumerate(dataset):
        print(f"\nEval {i+1}/{len(dataset)}: {case['question'][:60]}...")

        initial_state: AgentState = {
            "question": case["question"],
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

        result = graph.invoke(initial_state)
        answer = result.get("answer", "")
        escalated = result.get("escalated", False)
        intent = result.get("intent", "")

        escalation_correct = escalated == case["should_escalate"]
        keyword_hits = [kw for kw in case["expected_keywords"] if kw.lower() in answer.lower()]
        keyword_score = len(keyword_hits) / len(case["expected_keywords"]) if case["expected_keywords"] else 1.0
        intent_correct = intent == case.get("expected_intent", intent)
        if intent_correct:
            intent_correct_count += 1

        test_passed = escalation_correct and keyword_score >= 0.5
        if test_passed:
            passed += 1

        results.append({
            "question": case["question"],
            "passed": test_passed,
            "escalation_correct": escalation_correct,
            "keyword_score": keyword_score,
            "intent": intent,
            "intent_correct": intent_correct,
            "answer_preview": answer[:100],
        })

        status = "PASS" if test_passed else "FAIL"
        intent_status = "OK" if intent_correct else "WRONG"
        print(f"  {status} | intent={intent} [{intent_status}] | escalation_correct={escalation_correct} | keyword_score={keyword_score:.2f}")

    print(f"\n{'='*50}")
    print(f"RESULTS:         {passed}/{len(dataset)} passed ({passed/len(dataset)*100:.0f}%)")
    print(f"INTENT ACCURACY: {intent_correct_count}/{len(dataset)} correct ({intent_correct_count/len(dataset)*100:.0f}%)")

    log_to_langsmith(results, passed, len(dataset), intent_correct_count)

    return results


if __name__ == "__main__":
    run_evals()
