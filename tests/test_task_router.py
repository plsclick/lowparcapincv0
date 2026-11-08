import unittest

from lowparllm.ai.routing.models import TaskIntent
from lowparllm.ai.routing.registry import CapabilityStatus, capability_inventory
from lowparllm.ai.routing.router import TaskRouter
from lowparllm.core.config import AppConfig


class TaskRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = TaskRouter()

    def test_routes_direct_conversation_without_retrieval(self) -> None:
        plan = self.router.plan("Hello, how are you?")

        self.assertEqual(plan.intent, TaskIntent.GENERAL)
        self.assertIsNone(plan.research_mode)
        self.assertEqual(plan.required_capabilities, ("ollama.inference",))

    def test_routes_local_web_and_combined_research(self) -> None:
        cases = (
            ("Explain the PDF I imported about typography.", "local"),
            ("What are the latest React changes?", "web"),
            ("Compare my React notes with the latest official documentation.", "both"),
        )

        for request, mode in cases:
            with self.subTest(mode=mode):
                plan = self.router.plan(request)
                self.assertEqual(plan.research_mode, mode)
                self.assertIn("research.orchestration", plan.required_capabilities)

    def test_routes_programming_management_and_unknown_intents(self) -> None:
        self.assertEqual(
            self.router.plan("Explain how Python exceptions work").intent,
            TaskIntent.PROGRAMMING,
        )
        self.assertEqual(
            self.router.plan("List installed Ollama models").intent,
            TaskIntent.MODEL_MANAGEMENT,
        )
        self.assertEqual(
            self.router.plan("").intent,
            TaskIntent.UNKNOWN,
        )

    def test_pdf_creation_requests_do_not_select_removed_generation_capabilities(self) -> None:
        image = self.router.plan("Generate a cinematic futuristic city at night.")
        report = self.router.plan(
            "Create a detailed technical report about my project and save it as a PDF."
        )
        latest_report = self.router.plan(
            "Generate a PDF report about the latest Python changes."
        )

        self.assertEqual(report.intent, TaskIntent.MULTI_STEP)
        self.assertEqual(latest_report.intent, TaskIntent.WEB_RESEARCH)
        self.assertEqual(latest_report.research_mode, "web")
        for plan in (image, report, latest_report):
            self.assertNotIn("image.generation", plan.required_capabilities)
            self.assertNotIn("document.generation", plan.required_capabilities)
        self.assertNotIn("image.generation", {
            capability.identifier
            for capability in capability_inventory(AppConfig())
        })
        self.assertNotIn("document.generation", {
            capability.identifier
            for capability in capability_inventory(AppConfig())
        })

    def test_forced_research_mode_is_explicit_and_bounded(self) -> None:
        plan = TaskRouter(max_steps=1).plan(
            "Summarize these findings", forced_mode="web"
        )

        self.assertEqual(plan.research_mode, "web")
        self.assertEqual(len(plan.steps), 1)
        with self.assertRaises(ValueError):
            self.router.plan("Question", forced_mode="unsupported")

    def test_multi_step_research_keeps_both_research_mode_and_complexity(self) -> None:
        plan = self.router.plan(
            "Research the latest React releases and prepare a technical report."
        )

        self.assertEqual(plan.research_mode, "web")
        self.assertEqual(plan.complexity, "multi-step")


class CapabilityInventoryTests(unittest.TestCase):
    def test_inventory_marks_unavailable_backends_and_preserves_known_limits(self) -> None:
        capabilities = capability_inventory(
            AppConfig(),
            ollama_status={
                "software": {"ollama": {"service_available": False}}
            },
            searxng_status=(False, "not reachable"),
            rag_health={"qdrant_available": False},
        )
        by_id = {capability.identifier: capability for capability in capabilities}

        self.assertEqual(
            by_id["ollama.inference"].status, CapabilityStatus.UNAVAILABLE
        )
        self.assertEqual(
            by_id["search.searxng"].status, CapabilityStatus.FAILED
        )
        self.assertEqual(
            by_id["rag.retrieval"].status, CapabilityStatus.FAILED
        )
        self.assertEqual(
            by_id["adapter.inference"].status, CapabilityStatus.INCOMPATIBLE
        )
        self.assertEqual(
            by_id["coding.workspace"].status, CapabilityStatus.AVAILABLE
        )

    def test_inventory_marks_missing_rag_embedding_as_installation_required(self) -> None:
        capabilities = capability_inventory(
            AppConfig(),
            rag_health={
                "qdrant_available": True,
                "sqlite_available": True,
                "embedding_model_available": False,
                "embedding_model_missing": True,
                "embedding_error": "Embedding model is not installed.",
            },
        )
        rag = next(
            capability
            for capability in capabilities
            if capability.identifier == "rag.retrieval"
        )

        self.assertEqual(rag.status, CapabilityStatus.INSTALLATION)
        self.assertEqual(rag.detail, "Embedding model is not installed.")
