import os
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


class generate_response:

    def __init__(self,llm):
        self.llm = llm

    def generate(self, docs: list[Document], query: str) -> str:

        template = '''
            You are an expert assistant answering a user's query using only the information present in the given context.

            Please follow these points while answering:
             - Use only the information present in the context, do not use outside knowledge.
             - If the context does not contain enough information to answer the query, say that you don't know, don't answer from your own knowledge.
             - Keep the answer concise and directly address the query.

            Context:
            {context}

            Query: {query}
            '''

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate.from_template(template).format(context=context, query=query)
        response = self.llm.invoke(prompt).content.strip()

        return response
