import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app


def document(text, page=None):
    metadata = {}
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=text, metadata=metadata)


class AppTests(unittest.TestCase):
    def test_build_retriever_requests_top_five_similarity_results(self):
        vectorstore = Mock()
        app.build_retriever(vectorstore)
        vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    def test_context_labels_pages_using_one_based_numbers(self):
        context = app.format_context([document("Fault details", page=0), document("More")])
        self.assertIn("[Page 1]\nFault details", context)
        self.assertIn("[Page not available]\nMore", context)

    def test_answer_question_uses_retrieved_context_and_adds_citation(self):
        retriever = Mock()
        retriever.invoke.return_value = [document("The fault is described here.", page=8)]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Check the documented fault.")

        answer = app.answer_question("What does this fault mean?", retriever, llm)

        self.assertIn("Check the documented fault.", answer)
        self.assertIn("pages 9", answer)
        prompt_messages = llm.invoke.call_args.args[0]
        self.assertIn("The fault is described here.", prompt_messages[0].content)
        retriever.invoke.assert_called_once_with("What does this fault mean?")

    def test_empty_question_is_rejected_before_retrieval(self):
        retriever = Mock()
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            app.answer_question("  ", retriever, Mock())
        retriever.invoke.assert_not_called()

    def test_history_is_rendered_and_cleared(self):
        rendered = app.render_history(
            [{"question": "What is <fault>?", "answer": "See page 2."}]
        )
        self.assertIn("What is &lt;fault&gt;?", rendered)
        self.assertEqual(app.clear_history(), ([], "_Your questions and answers will appear here._", [], ""))

    def test_submit_question_updates_chat_and_session_history(self):
        with patch.object(app, "answer_question", return_value="Grounded answer"):
            result = app.submit_question(" What happened? ", [], [])

        question, chat, history_panel, history_state, status = result
        self.assertEqual(question, "")
        self.assertEqual(status, "")
        self.assertEqual(chat[0], {"role": "user", "content": "What happened?"})
        self.assertEqual(chat[1], {"role": "assistant", "content": "Grounded answer"})
        self.assertEqual(
            history_state,
            [{"question": "What happened?", "answer": "Grounded answer"}],
        )
        self.assertIn("Grounded answer", history_panel)


if __name__ == "__main__":
    unittest.main()
