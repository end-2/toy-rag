from langchain_community.vectorstores import Chroma


class Memory:
    def __init__(self, text_splitter, embedding_model):
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.vectorstore = None

    def load(self, loader):
        data = loader.load()
        all_splits = self.text_splitter.split_documents(data)

        if self.vectorstore is not None:
            self.vectorstore.add_documents(all_splits)
            return

        self.vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=self.embedding_model
        )

    def retrieve(self, question):
        return self.vectorstore.similarity_search(question)

    def clear(self):
        self.vectorstore = None
