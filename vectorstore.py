from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:

    def __init__(self, embedding_model: str, collection_name: str, persist_directory: str) -> None:

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = Chroma(collection_name=collection_name,embedding_function=self.embeddings,persist_directory=persist_directory)

    def is_empty(self) -> bool:
        return self.store._collection.count() == 0

    def ingest(self, documents: list[Document]) -> None:
        if self.is_empty():
            self.store.add_documents(documents)

    def get_store(self) -> Chroma:
        return self.store

    def get_all_documents(self) -> list[Document]:
        result = self.store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(result["documents"], result["metadatas"])
        ]
