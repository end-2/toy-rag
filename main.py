import logging
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
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

chain = prompt | chat

messages = []

while True:
    user_input = input("Please enter message: ")
    if user_input == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    resp = chain.invoke(
        input={"messages": messages},
        config={"callbacks": [LoggingCallbackHandler(logger)]}
    )

    print(resp.content)
    messages.append(AIMessage(content=resp.content))
