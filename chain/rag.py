import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

context_qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved <contexts> to answer the question. \
If you don't know the answer, just say that you don't know. \

<contexts>
{contexts}
</contexts>
"""

memoir_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_qa_system_prompt),
        MessagesPlaceholder("messages"),
        ("human", "{query}"),
    ]
)


def memoir_qa_chain(llm):
    return (
            memoir_qa_prompt |
            llm
    )


multiple_query_generate_system_prompt = """You are a helpful assistant that generates multiple similarity search queries based on a <query>. \
Please make three responses in the following <format>. \
<format>
1.
2.
3.
</format>
"""

multiple_query_generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", multiple_query_generate_system_prompt),
        ("human", "<query>{query}</query>"),
    ]
)


def multiple_query_generate_chain(llm):
    return (
            multiple_query_generate_prompt |
            llm |
            StrOutputParser() |
            (lambda x: re.findall(r'[1-9]\. \"(.+?)\"', x))
    )


def multiple_query_retriever_chain(retriever):
    return lambda x: [retriever.invoke(q) for q in x]


def reformatting(inputs):
    input_map = {}
    for docs in inputs:
        for doc in docs:
            if isinstance(doc, Document):
                input_map[doc.page_content] = 0
    output = ""
    for doc in input_map.keys():
        output += f"<document>{doc}</document>\n"
    return output


rerank_system_prompt = """You are an assistant for sorts out similar contents. \
<documents> is given. Select three <document> with similar content to <query>. \
Please select maximum three documents in the following <format>. \
<format></format>

<documents>
{documents}
</documents>

<query>
{query}
</query>
"""


def llm_rerank_chain(llm):
    return (
            ChatPromptTemplate.from_messages([
                ("system", rerank_system_prompt)
            ]) |
            llm
    )
