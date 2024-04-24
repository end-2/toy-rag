from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""


def chat_with_context_and_history(llm):
    """Create a chat chain that uses the context and chat history to answer questions."""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return qa_prompt | llm
