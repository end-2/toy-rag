from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from memory import Memory

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedding_model = GPT4AllEmbeddings()
memory = Memory(text_splitter, embedding_model)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
question = "What are the approaches to Task Decomposition?"

memory.load(loader)

for doc in memory.retrieve(question):
    print(doc)
