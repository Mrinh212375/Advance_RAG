from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score between 0.0 and 1.0 indicating how relevant the documents are to the query, where 0.0 is completely irrelevant and 1.0 is fully relevant.")


class grade_docs:

    def __init__(self,llm):
        self.llm = llm.with_structured_output(RelevanceScore)

    def grading(self, docs: list[Document], query: str) -> float:

        template = '''
            You are an expert in grading list of documents w.r.t to a query in RAG systems.

            Please follow these points while grading:
             - Focus on whether the document contains keywords or semantic meaning related to the query.
             - A document does not need to fully answer the query to be considered relevant, partial relevance counts, but you need to religiously judge the relevance wheather partial or full, this is very critical in the pipeline.
             - Respond with a relevance score between 0.0 and 1.0, where 0.0 means completely irrelevant and 1.0 means fully relevant. Partial relevance should be reflected with an intermediate score.

            Query: {query}
            Document: {document}
            '''

        prompt = PromptTemplate.from_template(template).format(query=query, document=[doc.page_content for doc in docs])
        relevancy = self.llm.invoke(prompt).score
        print(f"Relevance Score:{relevancy}")

        return relevancy


