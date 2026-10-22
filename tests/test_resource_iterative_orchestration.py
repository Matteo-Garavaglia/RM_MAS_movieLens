"""Offline tests; no LLM API, Qdrant, MCP, GPU required.
Run from repo root: python -m unittest discover -s tests -p 'test_resource_iterative_orchestration.py' -v
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from movie_rag.resource_iterative_orchestration import (
    OrchestrationLimits, ResourceIterativeOrchestrator, normalize_mcp_search_result)


class FakePlanner:
    async def __call__(self, request):
        return ({"semantic_query": "theory of regression" if "pratic" not in request.lower() else "regression theory and practice",
                 "filters": {"exclude_genres": ["Horror"]},
                 "clarification_required": False}, None)


class FakeProvider:
    def __init__(self, *, clarify=False, duplicate=False):
        self.clarify = clarify
        self.duplicate = duplicate
        self.n_decision = 0

    async def generate_text(self, **kwargs):
        agent = kwargs["agent"]
        if agent == "resource_validator":
            text = json.dumps({"status": "PARTIAL" if self.n_decision == 0 else "SUFFICIENT",
                "covered": ["theory"], "missing": ["practice"] if self.n_decision == 0 else [],
                "invalid_resource_ids": [], "reason": "needs exercise" if self.n_decision == 0 else "ok"})
        elif agent == "resource_orchestrator":
            self.n_decision += 1
            if self.clarify and self.n_decision == 1:
                text = json.dumps({"action":"CLARIFY","question":"Per quale classe?","reason":"level unspecified"})
            elif self.n_decision == 1:
                text = json.dumps({"action":"RETRIEVE", "semantic_query":"theory of regression" if self.duplicate else "practical regression exercises", "reason":"missing practical resources"})
            else:
                text = json.dumps({"action":"FINISH", "reason":"sufficient"})
        elif agent == "resource_clarifier":
            text = json.dumps({"question":"A quale livello scolastico?"})
        else:
            text = json.dumps({"answer":"Raccomando risorse basate sui risultati", "resource_ids":["r1", "r2", "invented"]})
        class R: pass
        r = R(); r.text = text; r.trace = {"agent":agent,"input_tokens":30,"output_tokens":20}
        return r


class Search:
    def __init__(self): self.calls = []
    async def __call__(self, query, filters, limit):
        self.calls.append((query, dict(filters), limit))
        number = len(self.calls)
        return {"results": [{"resource_id": "r1" if number == 1 else "r2",
                             "title": "Theory" if number == 1 else "Exercises",
                             "text": "Resource text", "metadata": {"source":"test"},
                             "score": 0.8}]}


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def make(self, *, clarify=False, duplicate=False, **limits):
        search = Search()
        workflow = ResourceIterativeOrchestrator(
            planner=FakePlanner(), search=search,
            provider=FakeProvider(clarify=clarify, duplicate=duplicate),
            limits=OrchestrationLimits(max_retrievals=3, max_clarifications=1, **limits))
        return workflow, search

    async def test_iterative_merge_and_grounded_ids(self):
        flow, search = self.make()
        result = await flow.run("Regressione teoria e pratica")
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(set(result.candidates), {"r1", "r2"})
        self.assertEqual(result.final_answer["resource_ids"], ["r1", "r2"])
        self.assertTrue(any(e.action == "grounding_ids_removed" for e in result.events))
        self.assertEqual(search.calls[0][1], search.calls[1][1])

    async def test_clarification_pause_resume(self):
        flow, search = self.make(clarify=True)
        state = await flow.run("Regressione")
        self.assertEqual(state.status, "WAITING_USER")
        self.assertIn("classe", state.pending_question)
        state = await flow.resume(state, "Quarta superiore")
        self.assertEqual(state.status, "COMPLETE")
        self.assertEqual(len(state.conversation), 1)
        self.assertEqual(len(search.calls), 2)  # no repeated first search during resume

    async def test_duplicate_prevents_extra_retrieval(self):
        flow, search = self.make(duplicate=True)
        state = await flow.run("Regressione")
        self.assertEqual(state.status, "COMPLETE")
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(state.termination_reason, "duplicate_or_empty_query")

    async def test_trace_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            flow, _ = self.make()
            flow.trace_dir = Path(temp)
            state = await flow.run("Regressione")
            target = Path(temp, state.run_id, "state.json")
            self.assertTrue(target.exists())
            d = json.loads(target.read_text())
            self.assertEqual(d["status"], "COMPLETE")
            self.assertEqual(set(d["candidates"]), {"r1", "r2"})

    async def test_empty_search_has_safe_answer(self):
        flow, _ = self.make()
        async def empty(*args): return {"results": []}
        flow.search = empty
        state = await flow.run("Regressione")
        self.assertEqual(state.status, "COMPLETE")
        self.assertEqual(state.final_answer["resource_ids"], [])

    async def test_planner_initial_clarification(self):
        flow, search = self.make()
        async def incomplete(request):
            if "Quarta" in request:
                return {"semantic_query":"regression for school", "filters":{}, "clarification_required":False}
            return {"semantic_query":"", "filters":{}, "clarification_required":True}
        flow.planner = incomplete
        state = await flow.run("Cerco materiali")
        self.assertEqual(state.status, "WAITING_USER")
        self.assertIn("livello", state.pending_question)
        self.assertEqual(len(search.calls), 0)
        state = await flow.resume(state, "Quarta superiore")
        self.assertEqual(state.status, "COMPLETE")
        self.assertGreaterEqual(len(search.calls), 1)

    def test_mcp_schema(self):
        self.assertEqual(normalize_mcp_search_result({"results": []}), {"results": []})
        with self.assertRaises(ValueError): normalize_mcp_search_result({"other": []})


if __name__ == "__main__":
    unittest.main()
