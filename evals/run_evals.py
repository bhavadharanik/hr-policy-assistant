"""Eval suite for the HR Policy Assistant.

Checks three named behaviours per case:
- intent classification accuracy
- escalation correctness (escalated vs not)
- answer quality via semantic similarity (not keyword matching)
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import build_graph
from src.agent.state import AgentState

SIMILARITY_PASS_THRESHOLD = 0.6


def _semantic_similarity(answer: str, expected_phrases: list[str]) -> dict[str, float]:
    """Score how well the answer semantically covers each expected phrase."""
    try:
        from langchain_openai import OpenAIEmbeddings
        import numpy as np

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not expected_phrases:
            return {}

        answer_vec = embeddings.embed_query(answer)
        scores = {}
        for phrase in expected_phrases:
            phrase_vec = embeddings.embed_query(phrase)
            dot = sum(a * b for a, b in zip(answer_vec, phrase_vec))
            norm_a = sum(a ** 2 for a in answer_vec) ** 0.5
            norm_b = sum(b ** 2 for b in phrase_vec) ** 0.5
            scores[phrase] = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        return scores
    except Exception as e:
        print(f"  [WARN] Semantic scoring failed, falling back to keyword match: {e}")
        return {phrase: 1.0 if phrase.lower() in answer.lower() else 0.0 for phrase in expected_phrases}


def log_to_langsmith(results: list, passed: int, total: int):
    """Log eval results to LangSmith."""
    try:
        from langsmith import Client
        client = Client()

        dataset_name = "hr-policy-assistant-evals"

        try:
            dataset = client.create_dataset(dataset_name, description="HR Policy Assistant eval dataset")
        except Exception:
            dataset = client.read_dataset(dataset_name=dataset_name)

        for result in results:
            try:
                client.create_example(
                    inputs={"question": result["question"]},
                    outputs={
                        "answer": result["answer_preview"],
                        "intent": result["intent"],
                        "passed": result["passed"],
                    },
                    dataset_id=dataset.id,
                )
            except Exception:
                pass

        print(f"\n[LangSmith] Logged to dataset: '{dataset_name}'")

    except Exception as e:
        print(f"\n[LangSmith] Could not log results: {e}")


def run_evals(verbose: bool = False):
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

        # Check 1: intent classification
        intent_correct = intent == case.get("expected_intent", intent)
        if intent_correct:
            intent_correct_count += 1

        # Check 2: escalation correctness
        escalation_correct = escalated == case["should_escalate"]

        # Check 3: semantic answer quality (not keyword matching)
        expected_phrases = case.get("expected_phrases", [])
        similarity_scores = _semantic_similarity(answer, expected_phrases)
        if expected_phrases:
            coverage = sum(1 for s in similarity_scores.values() if s >= SIMILARITY_PASS_THRESHOLD)
            quality_passed = coverage / len(expected_phrases) >= 0.5
        else:
            quality_passed = True

        test_passed = intent_correct and escalation_correct and quality_passed
        if test_passed:
            passed += 1

        result_entry = {
            "question": case["question"],
            "passed": test_passed,
            "intent": intent,
            "intent_correct": intent_correct,
            "escalation_correct": escalation_correct,
            "quality_passed": quality_passed,
            "similarity_scores": {k: round(v, 3) for k, v in similarity_scores.items()},
            "answer_preview": answer[:120],
        }
        results.append(result_entry)

        status = "PASS" if test_passed else "FAIL"
        print(f"  {status} | intent={'OK' if intent_correct else 'WRONG'} | escalation={'OK' if escalation_correct else 'WRONG'} | quality={'OK' if quality_passed else 'WEAK'}")

        if verbose and similarity_scores:
            for phrase, score in similarity_scores.items():
                print(f"    '{phrase}': {score:.3f}")

    print(f"\n{'='*50}")
    print(f"RESULTS:         {passed}/{len(dataset)} passed ({passed/len(dataset)*100:.0f}%)")
    print(f"INTENT ACCURACY: {intent_correct_count}/{len(dataset)} ({intent_correct_count/len(dataset)*100:.0f}%)")

    log_to_langsmith(results, passed, len(dataset))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_evals(verbose=args.verbose)
