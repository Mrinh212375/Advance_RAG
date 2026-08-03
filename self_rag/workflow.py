from self_rag.query_transform import qry_transformer
from self_rag.retriever import HybridRetriever
from self_rag.grader import grade_docs, grade_response
from self_rag.generator import generate_response
from self_rag.abstain import abstain_response
from typing import TypedDict, Literal, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START,END
from config import load_config
# from ..config import load_config
from langchain_openai import ChatOpenAI
from vectorstore import VectorStore
from langgraph.types import Send, Command
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
import operator

print(load_config().openai_api_key)
print(load_config().open_ai_model)

llm = ChatOpenAI(model=load_config().open_ai_model, api_key=load_config().openai_api_key)

RELEVANCE_THRESHOLD = 0.5

class state(TypedDict):
    user_query: str
    hyde_retrieved_docs: list[Document]
    rewritten_retrieved_docs: list[Document]
    expanded_retrieved_docs: list[Document]
    retrieved_docs: list[Document]
    relevance: float
    generated_response: str
    max_retry: int
    grounded: bool
    useful: bool
    max_generation_retry: int


vs = VectorStore(load_config().embedding_model, load_config().collection_name, load_config().persist_directory)
documents = vs.get_all_documents()
retriever = HybridRetriever(vs.get_store(), documents, load_config().reranker_model).get_retriever()


def query_transformer_dispatcher(state:state) -> Command:

    print("[node] query_transformer_dispatcher")

    qt = qry_transformer(llm=llm)

    is_retry = state.get('relevance') is not None
    if not is_retry:
        retry_context = "This is the first attempt at transforming this query for retrieval, no prior attempts have failed."
        remaining_retries = state['max_retry']
    else:
        remaining_retries = state['max_retry'] - 1
        retry_context = (
            f"The previous query transformation failed to retrieve relevant documents ({remaining_retries} retries "
            "remaining after this one). Try a noticeably different transformation strategy this time (different "
            "vocabulary, broader or narrower phrasing, synonyms) instead of repeating the same transformation as before."
        )

    return Command(goto=[
        Send("hyde_node", {**state,"transformer":qt,"retry_context":retry_context}),
        Send("rewritter_node", {**state,"transformer":qt,"retry_context":retry_context}),
        Send("expander_node", {**state,"transformer":qt,"retry_context":retry_context})
        ], update={"max_retry":remaining_retries})

def hyde_retriever(state:dict):
    print("[node] hyde_retriever")

    transformed_query = state['transformer'].hyde(state['user_query'], context=state['retry_context'])
    print(f"hyde_transformed_query:{transformed_query}")
    return {"hyde_retrieved_docs": retriever.invoke(transformed_query)}

def rewriter_retrever(state:dict):
    print("[node] rewriter_retrever")

    transformed_query = state['transformer'].rewriter(state['user_query'], context=state['retry_context'])
    print(f"rewritter_transformed_query:{transformed_query}")
    return {"rewritten_retrieved_docs": retriever.invoke(transformed_query),}

def expander_retriever(state:dict):
    print("[node] expander_retriever")

    transformed_query = state['transformer'].expander(state['user_query'], context=state['retry_context'])
    print(f"expander_transformed_query:{transformed_query}")
    return {"expanded_retrieved_docs": retriever.invoke(transformed_query),}


def fuse_rerank_node(state:state):
    print("[node] fuse_rerank_node")

    k=60
    scores, lookup = {}, {}
    for ranked in [state['hyde_retrieved_docs'],state['rewritten_retrieved_docs'],state['expanded_retrieved_docs']]:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1/(k + rank)
            lookup[key] = doc
    fused_docs =  [lookup[key] for key in sorted(scores, key=scores.get, reverse=True)][:10]

    #### Fuse then Rerank steps using Cross-Encoder-Reranker
    compressor = CrossEncoderReranker(model = HuggingFaceCrossEncoder(model_name = load_config().reranker_model, model_kwargs={"max_length": 512}),top_n = 5)
    fused_reranked_docs = compressor.compress_documents(documents=fused_docs, query=state['user_query'])

    return {"retrieved_docs": fused_reranked_docs}


def generator_node(state:state):
    print("[node] generator")

    generator = generate_response(llm=llm)
    llm_response = generator.generate(state['retrieved_docs'], state['user_query'])
    remaining_generation_retries = state['max_generation_retry'] - 1
    return {"generated_response":llm_response, "max_generation_retry": remaining_generation_retries}

def grader_node(state:state):
    print("[node] grader")

    grader = grade_docs(llm)
    relevance = grader.grading(docs=state['retrieved_docs'], query=state['user_query'])
    return {"relevance": relevance}

def response_grader_node(state:state):
    print("[node] response_grader")

    grader = grade_response(llm)
    grounded = grader.grade_hallucination(docs=state['retrieved_docs'], response=state['generated_response'])
    useful = grader.grade_answer(query=state['user_query'], response=state['generated_response'])
    return {"grounded": grounded, "useful": useful}

def abstain_node(state:state):
    print("[node] abstain")

    return {"generated_response": abstain_response()}

def router_fn(state:state) -> Literal["generator", "query_transformer_dispatcher", "abstain"]:
    print("[node] router")

    if state['relevance'] >= RELEVANCE_THRESHOLD:
        return "generator"
    elif state['max_retry'] > 0:
        return "query_transformer_dispatcher"
    else:
        return "abstain"

def response_router_fn(state:state) -> Literal["generator", "abstain", "__end__"]:
    print("[node] response_router")

    if state['grounded'] and state['useful']:
        return END
    elif state['max_generation_retry'] > 0:
        # Response either hallucinated or didn't resolve the query, regenerate from the same context.
        return "generator"
    else:
        return "abstain"

graph = StateGraph(state)

graph.add_node(
    "query_transformer_dispatcher",
    query_transformer_dispatcher,
    destinations=("hyde_node", "rewritter_node", "expander_node"),
)
graph.add_node("hyde_node",hyde_retriever)
graph.add_node("rewritter_node",rewriter_retrever)
graph.add_node("expander_node",expander_retriever)
graph.add_node("fuse_rerank",fuse_rerank_node)
graph.add_node("grader",grader_node)
graph.add_node("generator",generator_node)
graph.add_node("response_grader",response_grader_node)
graph.add_node("abstain",abstain_node)


graph.add_edge(START,"query_transformer_dispatcher")
graph.add_edge("hyde_node","fuse_rerank")
graph.add_edge("rewritter_node","fuse_rerank")
graph.add_edge("expander_node","fuse_rerank")
graph.add_edge("fuse_rerank","grader")
graph.add_conditional_edges("grader",router_fn)
graph.add_edge("generator", "response_grader")
graph.add_conditional_edges("response_grader", response_router_fn)
graph.add_edge("abstain", END)


self_rag_workflow = graph.compile()
