from langchain_core.prompts import ChatPromptTemplate

history_reform_system_prompt = """Given Chat history. \
Formalize it to increase the accuracy of similarity search. \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""


def chat_history_reformer(llm):
    """Create a chat chain that reformulates the chat history to increase the accuracy of similarity search."""
    history_reform_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_reform_system_prompt),
            ("human", "{history}"),
        ]
    )
    return history_reform_prompt | llm


query_reform_system_prompt = """Given a user's question. \
You want to find information to answer the question in the user's conversation history with a similarity search. \
Transform the question to increase the accuracy of the similarity search. \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""


def retriever_query_reformer(llm):
    """Create a chat chain that reformulates the user's question to increase the accuracy of similarity search."""
    query_reform_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_reform_system_prompt),
            ("human", "{query}"),
        ]
    )
    return query_reform_prompt | llm
