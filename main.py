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


if __name__ == "__main__":

    cfg = config.load_config()

    # rag = build_rag_chain(cfg) 

    query_transformer_model = ChatOpenAI(model_name=cfg.open_ai_model, api_key=cfg.openai_api_key)

    generator_llm = ChatOpenAI(model_name=cfg.open_ai_model, api_key=cfg.openai_api_key)
    query_transformer = qry_transformer(query_transformer_model)

    vs = VectorStore(cfg.embedding_model, cfg.collection_name, cfg.persist_directory)
    documents = vs.get_all_documents()

    retriever = HybridRetriever(vs.get_store(), documents, cfg.reranker_model).get_retriever()

    golden_doc = pd.read_csv(cfg.golden_set_path)
    questions = golden_doc["question"]
    references = golden_doc["reference_answer"]

    answers = []
    contexts = []
    for q in questions:

        ### query(q) rewriting or HYDE or expansion or Multiquery retrieval.
        expanded_query = query_transformer.expander(q)
        hyde = query_transformer.hyde(q)
        print(f"\n expanded query: {expanded_query}")
        print(f"\n hyde query: {hyde}")
        # answer, retrieved_context = rag.invoke(new_query)
        # print(f"\nQuery:\n{q}")
        # print(f"\nFinal Answer:\n{answer}")
        # expanded_context = retriever.invoke(expanded_query)
        # hyde_context = retriever.invoke(hyde)

        # # rankers = [expanded_context, hyde_context]
        # doc_list = {}
        # for d in documents:
        #     doc_list[d]=0
        
        # for i,j in zip(expanded_context,hyde_context):

        #     doc_list[expanded_context[i]] += 1/(60 + i)
        #     doc_list[hyde_context[j]] += 1/(60 + j)
        
        # sorted_docs = sorted(doc_list.items(), key=lambda x: x[1], reverse=True)

        # retrieved_context = "\n".join([i[0].page_content for i in sorted_docs[:4]])
        system_prompt = '''You are an intelligent context analyzer, your task is to analyze the context and user query to response for that user query.
                        Use the context below to answer the question.\n\nContext:\n{context} \n

                    Now consider below points while answering:
                    - Don't need to include think block in the response, e.g - <think>.....</think>, remove the whole <think> block, i.e response should not even contain <think>...</think>.
                    - Just directly give the response precisely according to the context.
                    - If the answer is not in the context, say I don't know, don't use your knowledge to genearte response.
                    - always remeber your task is to lookup both the context and user_query and answer(if possible).
            '''
        
        def fuse(ranked_lists, k=60, top=10):
            scores, lookup = {}, {}
            for ranked in ranked_lists:
                for rank, doc in enumerate(ranked):
                    key = doc.page_content
                    scores[key] = scores.get(key, 0) + 1/(k + rank)
                    lookup[key] = doc
            return [lookup[key] for key in sorted(scores, key=scores.get, reverse=True)[:top]]

        # in the loop:
        fused = fuse([retriever.invoke(expanded_query), retriever.invoke(hyde)])
        

        #### Fuse then Rerank steps using Cross-Encoder-Reranker
        compressor = CrossEncoderReranker(model = HuggingFaceCrossEncoder(model_name = cfg.reranker_model),top_n = 4)
        fused_reranked_docs = compressor.compress_documents(documents=fused, query=q)

        # context_list = [d.page_content for d in fused]

        context_list = [d.page_content for d in fused_reranked_docs]
        retrieved_context = "\n".join(context_list)
        prompt = system_prompt.format(context=retrieved_context) + f"\n\nQuestion: {q}"
        response = generator_llm.invoke(prompt)

        # response = generator_llm.invoke(system_prompt.format(context = retrieved_context))
        print(f"Answer:\n {response.content}")
        answers.append(response.content)
        contexts.append(context_list)

    evaluator = RAGEvaluator(cfg.open_ai_model, cfg.openai_api_key, cfg.embedding_model)
    eval_result = evaluator.evaluate(questions, references, answers, contexts)
    print(eval_result)

    evaluator.save_to_excel(eval_result, cfg.eval_output_file, cfg.eval_sheet_name)
