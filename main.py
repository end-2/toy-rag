import logging
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO,
                    filename="app.log",
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat | StrOutputParser()

chat_history = ChatMessageHistory()

while True:
    user_input = input("Please enter message: ")
    if user_input == "exit":
        break

    chat_history.add_user_message(user_input)

    resp = chain.invoke(
        input={"messages": chat_history.messages},
        config={"callbacks": [LoggingCallbackHandler(logger)]}
    )

    print(resp)
    chat_history.add_ai_message(resp)
