import logging
import dotenv

from langchain.memory import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from chain import memoir_qa_chain, multiple_query_generate_chain, multiple_query_retriever_chain, llm_rerank_chain, \
    reformatting

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

rag_llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
chat_llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.7)

db = Chroma(embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retriever_chain = (
        multiple_query_generate_chain(rag_llm) |
        multiple_query_retriever_chain(retriever) |
        reformatting
)

chat_history = ChatMessageHistory()

while True:
    user_input = input("Please enter message: ")
    if user_input == "exit":
        break

    docs = retriever_chain.invoke(
        input={
            "query": user_input
        }
    )
    contexts = (llm_rerank_chain(rag_llm) | StrOutputParser()).invoke(
        input={
            "documents": docs,
            "query": user_input
        }
    )
    resp = (memoir_qa_chain(chat_llm) | StrOutputParser()).invoke(
        input={
            "contexts": contexts,
            "query": user_input,
            "messages": ChatMessageHistory().messages
        },
        config={
            "callbacks": [LoggingCallbackHandler(logger)]
        }
    )
    print(f"Query response: {resp}")

    db.add_documents([Document(page_content=user_input)])
    # chat_history.add_user_message(user_input)
    # chat_history.add_ai_message(resp)
