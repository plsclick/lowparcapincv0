import unittest

from lowparllm.ai.llm.conversation import Conversation


class ConversationTests(unittest.TestCase):
    def test_roles_and_order_are_preserved(self) -> None:
        conversation = Conversation("System prompt")
        request, trimmed = conversation.prepare_messages("Hello", 4096)
        self.assertEqual(trimmed, 0)
        self.assertEqual([item["role"] for item in request], ["system", "user"])
        conversation.complete_turn(request, "Hi", 4096)
        followup, _ = conversation.prepare_messages("Question?", 4096)
        self.assertEqual(
            [(item["role"], item["content"]) for item in followup],
            [
                ("system", "System prompt"),
                ("user", "Hello"),
                ("assistant", "Hi"),
                ("user", "Question?"),
            ],
        )

    def test_history_limit_trims_complete_old_turns(self) -> None:
        conversation = Conversation("system", max_history_messages=4)
        for number in range(3):
            request, _ = conversation.prepare_messages(f"u{number}", 4096)
            conversation.complete_turn(request, f"a{number}", 4096)
        self.assertEqual(conversation.message_count, 4)
        self.assertEqual(
            [message["content"] for message in conversation.history],
            ["u1", "a1", "u2", "a2"],
        )
        next_request, pruned = conversation.prepare_messages("latest", 4096)
        self.assertEqual(pruned, 2)
        self.assertEqual(
            [message["content"] for message in next_request[1:]],
            ["u2", "a2", "latest"],
        )

    def test_context_character_budget_keeps_latest_turn(self) -> None:
        conversation = Conversation("system", max_history_messages=40)
        for number in range(3):
            request, _ = conversation.prepare_messages("x" * 800, 512)
            conversation.complete_turn(request, f"answer-{number}", 512)
        next_request, pruned = conversation.prepare_messages("n" * 800, 512)
        self.assertGreater(pruned, 0)
        self.assertEqual(next_request[-1]["content"], "n" * 800)
        self.assertLessEqual(len(next_request), 5)

    def test_failed_or_cancelled_turns_do_not_mutate_history(self) -> None:
        conversation = Conversation("system")
        request, _ = conversation.prepare_messages("pending", 4096)
        del request
        self.assertEqual(conversation.message_count, 0)

    def test_reset_clears_session_history(self) -> None:
        conversation = Conversation("system")
        request, _ = conversation.prepare_messages("hello", 4096)
        conversation.complete_turn(request, "world", 4096)
        conversation.reset()
        self.assertEqual(conversation.message_count, 0)
        self.assertEqual(conversation.history, ())


if __name__ == "__main__":
    unittest.main()
