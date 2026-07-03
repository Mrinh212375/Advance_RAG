import pandas as pd
from langchain_groq import ChatGroq

from vectorstore import VectorStore
from retriever import HybridRetriever
from chain import RAGChain
from eval import RAGEvaluator

import config


def build_rag_chain(cfg: config.Config) -> RAGChain:
    vs = VectorStore(cfg.embedding_model, cfg.collection_name, cfg.persist_directory)
    documents = vs.get_all_documents()

    retriever = HybridRetriever(vs.get_store(), documents, cfg.reranker_model).get_retriever()
    llm = ChatGroq(model_name=cfg.llm_model, groq_api_key=cfg.groq_api_key)  # type: ignore[call-arg]

    return RAGChain(llm, retriever)


if __name__ == "__main__":

    cfg = config.load_config()

    rag = build_rag_chain(cfg)

    golden_doc = pd.read_csv(cfg.golden_set_path)
    questions = golden_doc["question"]
    references = golden_doc["reference_answer"]

    answers = []
    contexts = []
    for q in questions:
        answer, retrieved_context = rag.invoke(q)
        print(f"\nQuery:\n{q}")
        print(f"\nFinal Answer:\n{answer}")
        answers.append(answer)
        contexts.append(retrieved_context)

    evaluator = RAGEvaluator(cfg.eval_llm_model, cfg.groq_api_key, cfg.embedding_model)
    eval_result = evaluator.evaluate(questions, references, answers, contexts)
    print(eval_result)

    evaluator.save_to_excel(eval_result, cfg.eval_output_file, cfg.eval_sheet_name)
