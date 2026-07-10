from langchain_core.prompts import PromptTemplate

class qry_transformer:

    def __init__(self,llm):
        self.llm = llm
    
    def rewriter(self, query):

        template = '''
            You are an expert in query re-writing to make this query good enough for retrieval in a RAG system
                please try to follow the below points while rewriting the query:
                 - Rewrite the query in such a way that it should reduce the lexical gap between query vocabulary and Document.
                 - Make the vocabulary of the query simple so that it will match easily with the document.
                 - Just give the Rewritten query in your response, nothing else.
                 
            Original query: {query}
            '''
        prompt = PromptTemplate.from_template(template).format(query=query)
        return self.llm.invoke(prompt).content
    
    def hyde(self,query):

        hyde_template = '''
            You are an expert in creating Hypothetical responses for an user query to enhance retrieval in RAG system.
            Given the question below, write a short hypothetical passage that could plausibly
            appear in a purchase order terms-and-conditions document and would directly answer it.

            Please follow these points while writing the passage:
             - Write it in the same register as formal contract clauses
             - Do not hedge, qualify, or mention that this is hypothetical.
             - Keep it concise, no preamble or explanation.
             - Just give the hypothetical passage in your response, nothing else.

            Question: {query}
            '''
        
        prompt = PromptTemplate.from_template(hyde_template).format(query=query)
        return self.llm.invoke(prompt).content
    
    def expander(self,query):

        expander_template = '''
                        Expand a brief query into a more detailed, comprehensive version.

                        You are a query expansion assistant. Take brief user queries and expand them into more detailed, comprehensive versions that:
                            1. Add relevant context and clarifications
                            2. Include related terminology and concepts
                            3. Specify what aspects should be covered
                            4. Maintain the original intent
                            5. Keep it as a single, coherent question
                            6. Just give the expanded query in your response, nothing else

                            Expand the query to be 2-3x more detailed while staying focused.

                            Query to expand:\n {query}
                            '''
        prompt = PromptTemplate.from_template(expander_template).format(query=query)
        return self.llm.invoke(prompt).content
