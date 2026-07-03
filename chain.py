import re

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class RAGChain:

    def __init__(self, llm, retriever) -> None:
        self.llm = llm
        self.retriever = retriever
        self.qa_chain = self._build_chain()

    def _build_chain(self):

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
        return create_retrieval_chain(self.retriever, combine_docs_chain)

    def invoke(self, query: str):

        response = self.qa_chain.invoke({"input": query})

        retrieved_context = [f"context {i+1}:\n{d.page_content}" for i, d in enumerate(response.get("context", []))]
        clean = re.sub(r'<think>.*?</think>\n\n\n?', '', response['answer'], flags=re.DOTALL)
        return clean, retrieved_context
