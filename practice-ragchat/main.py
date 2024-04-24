import logging
import dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from chat_history_retriever import ChatHistoryVectorStore
from reformer import chat_history_reformer, retriever_query_reformer
from chat import chat_with_context_and_history


dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO,
                    filename="../tmp/chat.log",
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

reformer_llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
chat_llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

chat_history_vector_store = ChatHistoryVectorStore("test_user", debugging=True)
retriever = chat_history_vector_store.as_retriever()

chain = chat_with_context_and_history(chat_llm) | StrOutputParser()

chat_history = ChatMessageHistory()

while True:
    user_input = input("Please enter message: ")
    if user_input == "exit":
        break

    # retriever_query = (retriever_query_reformer(reformer_llm) | StrOutputParser()).invoke(
    #     input={"query": user_input},
    #     config={"callbacks": [LoggingCallbackHandler(logger)]}
    # )
    # print(f"Retriever query: {retriever_query}")

    # context = retriever.invoke(retriever_query)
    context = retriever.invoke(user_input)
    print(f"Similarity search results: {context}")

    resp = chain.invoke(
        input={
            "context": context,
            "chat_history": chat_history.messages,
            "question": user_input,
        },
        config={
            "callbacks": [LoggingCallbackHandler(logger)]
        }
    )
    print(f"Query response: {resp}")

    # chat_history.add_user_message(user_input)
    # chat_history.add_ai_message(resp)
    # print(chat_history.messages)

    # history = (chat_history_reformer(reformer_llm) | StrOutputParser()).invoke(
    #     input={"history": f"Q: {user_input}\nA: {resp}"},
    #     config={"callbacks": [LoggingCallbackHandler(logger)]}
    # )
    # print(f"Reformulated history: {history}")
    # chat_history_vector_store.add_documents([Document(page_content=history)])
    chat_history_vector_store.add_documents([Document(page_content=user_input), Document(page_content=resp)])

