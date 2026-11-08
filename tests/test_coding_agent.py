import json
import tempfile
import unittest
from pathlib import Path

from lowparllm.ai.coding.agent import CodingAgent
from lowparllm.ai.coding.workspace import CodingWorkspace, WorkspaceError
from lowparllm.ai.llm.models import StreamEvent


class FakeChatService:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def stream_response(
        self,
        user_message,
        retrieved_context=None,
        response_guidance=None,
        *,
        ephemeral=False,
    ):
        self.calls.append(
            {
                "user_message": user_message,
                "retrieved_context": retrieved_context,
                "response_guidance": response_guidance,
                "ephemeral": ephemeral,
            }
        )
        response = next(self.responses)
        yield StreamEvent(content=json.dumps(response))
        yield StreamEvent(done=True)


class CodingAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_reads_files_before_returning_a_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
            chat = FakeChatService(
                [
                    {
                        "message": "I will inspect the entry point.",
                        "actions": [{"tool": "read_file", "path": "main.py"}],
                    },
                    {"message": "The entry point prints hello.", "actions": []},
                ]
            )
            agent = CodingAgent(CodingWorkspace(root), chat)

            outcome = await agent.start("Explain this application.")

            self.assertTrue(outcome.task_complete)
            self.assertIn("prints hello", outcome.message)
            self.assertIn("print('hello')", chat.calls[1]["retrieved_context"])
            self.assertIn("untrusted", chat.calls[0]["response_guidance"].lower())

    async def test_agent_proposes_changes_then_applies_only_after_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.py"
            original = "print('old')\n"
            source.write_text(original, encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(original.encode()).hexdigest()
            chat = FakeChatService(
                [
                    {
                        "message": "I inspected the file.",
                        "actions": [{"tool": "read_file", "path": "main.py"}],
                    },
                    {
                        "message": "I propose this one-line update.",
                        "actions": [
                            {
                                "tool": "write_file",
                                "path": "main.py",
                                "content": "print('new')\n",
                                "expected_sha256": digest,
                            }
                        ],
                    },
                    {"message": "The update is complete.", "actions": []},
                ]
            )
            workspace = CodingWorkspace(root)
            agent = CodingAgent(workspace, chat)

            proposal = await agent.start("Update the output.")
            self.assertTrue(proposal.pending_approval)
            self.assertIn("-print('old')", proposal.actions[0].arguments["preview"])
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            with self.assertRaises(WorkspaceError):
                await agent.continue_task()

            completed = await agent.continue_task(confirmed=True)

            self.assertTrue(completed.task_complete)
            self.assertEqual(source.read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual(len(completed.changes), 1)

    async def test_agent_rejects_out_of_workspace_model_requested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            chat = FakeChatService(
                [
                    {
                        "message": "Read an external file.",
                        "actions": [{"tool": "read_file", "path": "../secret.txt"}],
                    }
                ]
            )
            agent = CodingAgent(CodingWorkspace(root), chat)

            with self.assertRaises(ValueError):
                await agent.start("Read a file.")
