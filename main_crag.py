import pandas as pd
from langchain_groq import ChatGroq

from vectorstore import VectorStore
from retriever import HybridRetriever
from chain import RAGChain
from eval import RAGEvaluator
from query_transform import qry_transformer
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI

import config


def build_rag_chain(cfg: config.Config) -> RAGChain:
    vs = VectorStore(cfg.embedding_model, cfg.collection_name, cfg.persist_directory)
    documents = vs.get_all_documents()

    retriever = HybridRetriever(vs.get_store(), documents, cfg.reranker_model).get_retriever()
    llm = ChatGroq(model_name=cfg.llm_model, groq_api_key=cfg.groq_api_key)  # type: ignore[call-arg]

    return RAGChain(llm, retriever)


###### CRAG workflow ############
from crag.workflow import crag_workflow


if __name__ == "__main__":

####### invoke crag graph ###########

    golden_doc = pd.read_csv(config.load_config().golden_set_path)
    questions = golden_doc["question"]
    references = golden_doc["reference_answer"]

    answers = []
    contexts = []

    for q in questions:
        result = crag_workflow.invoke({"user_query":q, "max_retry":2})

        answers.append(result['generated_response'])
        contexts.append([d.page_content for d in result['retrieved_docs']])

    evaluator = RAGEvaluator(config.load_config().open_ai_model, config.load_config().openai_api_key, config.load_config().embedding_model)
    eval_result = evaluator.evaluate(questions, references, answers, contexts)
    print(eval_result)

    evaluator.save_to_excel(eval_result, config.load_config().eval_output_file, config.load_config().eval_sheet_name)

    # print(f"\n\nFinal response:{result['max_retry']}")

