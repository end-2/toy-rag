import unittest
from langchain_core.documents import Document
from chat_history_retriever import ChatHistoryVectorStore


class TestChatHistoryVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = ChatHistoryVectorStore("test_user")
        self.expected_answers = "The capital of France is Paris."
        self.page_contents = [
            self.expected_answers,
            "The capital of Korea is Seoul.",
            "My name is End-2",
            "This is my favorite book.",
            "I am a student at LANGCHAIN.",
            "My job is to test the ChatHistoryVectorStore class.",
            "Paris and Seoul are beautiful cities.",
        ]

    def test_retriever(self):
        for content in self.page_contents:
            self.store.add_documents([Document(page_content=content)])

        retriever = self.store.as_retriever()
        result = retriever.invoke("What is the capital of France?")[0].page_content

        self.assertEqual(result, self.expected_answers)


if __name__ == "__main__":
    unittest.main()
