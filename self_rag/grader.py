from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score between 0.0 and 1.0 indicating how relevant the documents are to the query, where 0.0 is completely irrelevant and 1.0 is fully relevant.")


class HallucinationGrade(BaseModel):
    grounded: bool = Field(description="True if every claim in the generated response is supported by the given documents, False if the response contains any unsupported or hallucinated information.")


class AnswerGrade(BaseModel):
    useful: bool = Field(description="True if the generated response directly and sufficiently resolves the user's query, False otherwise.")


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


class grade_response:

    def __init__(self,llm):
        self.hallucination_llm = llm.with_structured_output(HallucinationGrade)
        self.answer_llm = llm.with_structured_output(AnswerGrade)

    def grade_hallucination(self, docs: list[Document], response: str) -> bool:

        template = '''
            You are an expert in checking whether a generated response is grounded in a given set of documents.

            Please follow these points while grading:
             - Check whether every factual claim made in the response is supported by the documents.
             - If the response contains any information that cannot be traced back to the documents, mark it as not grounded.
             - Respond with True if the response is fully grounded in the documents, False otherwise.

            Documents: {documents}
            Response: {response}
            '''

        prompt = PromptTemplate.from_template(template).format(documents=[doc.page_content for doc in docs], response=response)
        grounded = self.hallucination_llm.invoke(prompt).grounded
        print(f"Grounded:{grounded}")

        return grounded

    def grade_answer(self, query: str, response: str) -> bool:

        template = '''
            You are an expert in checking whether a generated response resolves a user's query.

            Please follow these points while grading:
             - Check whether the response directly and sufficiently addresses the query.
             - A response that is grounded but incomplete or off-topic should still be marked as not useful.
             - Respond with True if the response is useful in resolving the query, False otherwise.

            Query: {query}
            Response: {response}
            '''

        prompt = PromptTemplate.from_template(template).format(query=query, response=response)
        useful = self.answer_llm.invoke(prompt).useful
        print(f"Useful:{useful}")

        return useful
