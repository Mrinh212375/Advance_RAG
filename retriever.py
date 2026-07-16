from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class HybridRetriever:

    def __init__(
        self,
        vector_store: Chroma,
        documents: list[Document],
        reranker_model: str,
        dense_k: int = 6,
        lambda_mult: float = 0.75,
        bm25_k: int = 6,
        top_n: int = 5,
    ) -> None:
        self.vector_store_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": dense_k, "lambda_mult": lambda_mult}
        )

        self.bm25_retriever = BM25Retriever.from_documents(documents=documents)
        self.bm25_retriever.k = bm25_k

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.vector_store_retriever, self.bm25_retriever], weights=[0.5, 0.5]
        )

        compressor = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=reranker_model), top_n=top_n
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.hybrid_retriever
        )

    def get_retriever(self) -> ContextualCompressionRetriever:
        # return self.compression_retriever
        return self.hybrid_retriever


if __name__ == "__main__":

    import config
    from vectorstore import VectorStore

    cfg = config.load_config()

    vs = VectorStore(cfg.embedding_model, cfg.collection_name, cfg.persist_directory)
    documents = vs.get_all_documents()
    print(f"Loaded {len(documents)} documents from the vector store.")

    hybrid_retriever = HybridRetriever(vs.get_store(), documents, cfg.reranker_model).get_retriever()

    test_query = "What are the payment terms?"
    results = hybrid_retriever.invoke(test_query)
    

    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Chunk {i} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}\n")

