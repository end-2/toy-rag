import logging
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


logging.basicConfig(level=logging.INFO,
                    filename='app.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info("Starting chat app")

dotenv.load_dotenv()

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
    logger.info("Start chat loop")

    user_input = input("Please enter message: ")
    logger.info("User input: %s", user_input)

    if user_input == "exit":
        break

    messages.append(HumanMessage(content=user_input))
    logger.info("Messages: %s", messages)

    resp = chain.invoke({"messages": messages})
    logger.info("Response: %s", resp)

    print(resp.content)
    messages.append(AIMessage(content=resp.content))
