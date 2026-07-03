from pyexpat import model

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from dotenv import load_dotenv
import asyncio
import os
import sys
from opentelemetry import context
import pandas as pd
# from sympy import content
# from openai import AsyncOpenAI
# from ragas.metrics.collections import NoiseSensitivity
# from ragas.llms import llm_factory
# from ragas.dataset_schema import SingleTurnSample
from ragas import evaluate
# from ragas import EvaluationDataset
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall, LLMContextPrecisionWithReference,
    Faithfulness, SemanticSimilarity,
)
import re
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
# from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_cohere import ChatCohere, CohereRerank



load_dotenv(r"D:/Learning/Learning_Langgraph/key.env")
api_key = os.getenv("GROQ_API_KEY")
print(api_key)

class RAG:

    def __init__(self):
        self.llm = ChatGroq(model_name="qwen/qwen3-32b", groq_api_key=api_key)  # type: ignore[call-arg]
        self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2",
                                   encode_kwargs={"normalize_embeddings": True})
        self.qa_chain = self._create_chain()


    ####### Ingesting Documents #######
    def ingestion(self) -> list[Document]:
        loader = PyPDFLoader(file_path=r"D:\Learning\DocsForRAG\SDI PO TC-India.pdf")
        content = loader.load()
        return content


    def splitter(self)-> list[Document]:
        doc_content = self.ingestion()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_documents(doc_content)

    # def _create_embeddings(self):
        
    #     self.embeddings = 

    def _is_vectorStore_empty(self) -> bool:

        # return (self.vector_store is None)
        return (self.vector_store._collection.count() == 0)
    

    def _create_vectorDB(self):

        self.vector_store = Chroma(collection_name='learning',embedding_function=self.embeddings,persist_directory="./Chroma_langchain_DB")
        if self._is_vectorStore_empty():
            self.vector_store.add_documents(self.splitter())

        return self.vector_store
    
    def _create_retriever(self):

        print("-----Creating vectorDB-------")
        self.vector_store = self._create_vectorDB()
        self.vector_store_retriever = self.vector_store.as_retriever(search_type = "mmr", search_kwargs = {"k":5, "lambda_mult": 0.75})
        # bm25 = BM25Retriever(k=5,)
        self.bm25_retriever = BM25Retriever.from_documents(documents=self.splitter())
        self.bm25_retriever.k = 5


        self.hybrid_retriever = EnsembleRetriever(retrievers = [self.vector_store_retriever,self.bm25_retriever],weights =[0.5,0.5])

        compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),top_n=3)
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.hybrid_retriever)


        return self.compression_retriever 
    
    def testing(self):

        q = "If a buyer wants to walk away from an order even though the supplier did nothing wrong, how much advance warning must it give?"

        # 1. ensemble BEFORE the reranker — the wide candidate pool
        pool = self.hybrid_retriever.invoke(q)
        print("POOL:", [d.page_content[:60] for d in pool])

        # 2. each leg separately — which retriever is (or isn't) finding Art 13
        print("DENSE:", [d.page_content[:60] for d in self.vector_store_retriever.invoke(q)])
        print("BM25 :", [d.page_content[:60] for d in self.bm25_retriever.invoke(q)])

        # 3. final, after rerank — what the LLM actually sees
        print("FINAL:", [d.page_content[:60] for d in self.compression_retriever.invoke(q)])


    def _create_chain(self):

        system_prompt = '''You are an intelligent context analyzer, your task is to analyze the context and user query to response for that user query. 
                             Use the context below to answer the question.\n\nContext:\n{context} \n

                            Now consider below points while answering:
                            - Don't need to include think block in the response, e.g - <think>.....</think>, remove the whole <think> block, i.e response should not even contain <think>...</think>.
                            - Just directly give the response precisely according to the context.
                            - If the answer is not in the context, say I don't know, don't use your knowledge to genearte response.  
                            - always remeber your task is to lookup both the context and user_query and answer(if possible).
                    '''
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        print("-----Creating retrieval--------")
        self.retriever = self._create_retriever()
        return create_retrieval_chain(self.retriever, combine_docs_chain)
    
    def invoke(self,query):

        # print("------- Creating Chain --------")
        # self.qa_chain = 

        # query = input("Enter your query: ")

        response =self.qa_chain.invoke({"input": query})

        print("\n---------Chain Response---------")
        retrieved_context = [f"context {i+1}:\n{d.page_content}" for i, d in enumerate(response.get("context", []))]
        # print(retrieved_chunk)
        print(f"Query:\n{query}")
        clean = re.sub(r'<think>.*?</think>\n\n\n?', '', response['answer'], flags=re.DOTALL)
        print(f"\nFinal Answer:\n{clean}")
        return clean,retrieved_context



if __name__ == "__main__":


    rg = RAG()
    rg.testing()
    pass
    golden_doc = pd.read_csv(r"D:\Learning\DocsForRAG\Ground_Truth_SDI\golden_set.csv")
    question = golden_doc['question']
    reference_answer = golden_doc['reference_answer']
    generated_answer = []
    retrived_chunk = []
    metric_scores = []

    for i, q in enumerate(question):
        ans, retrieved_context = rg.invoke(q)
        # clean_ans = re.sub(r'<think>.*?</think>\n\n\n?', '', ans, flags=re.DOTALL)
        generated_answer.append(ans)
        retrived_chunk.append(retrieved_context)
        # metric_score = await scorer.ascore(
        #     user_input=q,
        #     response=ans,
        #     reference=reference_answer[i],
        #     retrieved_contexts=retrieved_context
        # )
        # metric_scores.append(metric_score.value)

    # print(generated_answer)
    my_dict = {"user_input": list(question), "reference": list(reference_answer), "response": generated_answer, "retrieved_contexts":retrived_chunk}
    dataset = Dataset.from_dict(my_dict)

    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key) 
    evaluator_llm = LangchainLLMWrapper(llm)

    run_config = RunConfig(
    max_workers=1,     # serialize — one call at a time, no bursts. THE key knob.
    timeout=180,
    max_retries=10,    # on a 429, wait and retry instead of returning nan
    max_wait=60,       # cap the backoff
    )
    metrics = [LLMContextRecall(), LLMContextPrecisionWithReference(),
    Faithfulness(), SemanticSimilarity(),]

    eval_result = evaluate(dataset,llm = evaluator_llm,metrics=metrics, embeddings=HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2"),run_config=run_config)
    print(eval_result)
    # output = pd.DataFrame()
    # eval_result.results()
    # print(f"output dataframe:\n{output}")
    # with pd.ExcelWriter("Output.xlsx", mode='w') as writer:
    #     output.to_excel(writer, sheet_name="testing", index=False)
    # eval_result.to_pandas().to_excel("baseline_scores.xlsx", index=False, sheet_name='second_iteration')
    file = "baseline_scores.xlsx"
    if os.path.exists(file):
        with pd.ExcelWriter(file, mode="a", if_sheet_exists="replace") as writer:
            eval_result.to_pandas().to_excel(writer, sheet_name="hybrid_rerank", index=False)
    else:
        eval_result.to_pandas().to_excel(file, sheet_name="hybrid_rerank", index=False)


