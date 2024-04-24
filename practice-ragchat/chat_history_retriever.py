from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings


class ChatHistoryVectorStore:
    def __init__(self, username, debugging=False):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = Chroma(
            collection_name=username,
            embedding_function=GPT4AllEmbeddings()
        )
        self.debugging = debugging

    def as_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
            # search_type="similarity_score_threshold",
            # search_kwargs={"score_threshold": .5, "k": 3}
        )

    def add_documents(self, documents):
        all_splits = self.text_splitter.split_documents(documents)
        if self.debugging:
            with open("../splits.txt", "a") as f:
                for split in all_splits:
                    f.write(f"{split.page_content}\n")
        self.vectorstore.add_documents(all_splits)
