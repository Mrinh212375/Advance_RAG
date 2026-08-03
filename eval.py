import os
import sys
import types

# ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai (only used for an
# isinstance check against VertexAI chat models, which this project never uses), but that module
# was removed in langchain-community>=0.4 in favor of the standalone langchain-google-vertexai
# package. Stub it out so the import in ragas.llms.base succeeds without pulling in VertexAI.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        pass

    _vertexai_shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall, LLMContextPrecisionWithReference,
    Faithfulness, SemanticSimilarity,
)
from langchain_openai import ChatOpenAI

class RAGEvaluator:

    def __init__(self, eval_llm_model: str, groq_api_key: str, embedding_model: str) -> None:
        llm = ChatOpenAI(model=eval_llm_model, api_key=groq_api_key)
        self.evaluator_llm = LangchainLLMWrapper(llm, bypass_temperature=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.metrics = [
            LLMContextRecall(), LLMContextPrecisionWithReference(),
            Faithfulness(), SemanticSimilarity(),
        ]
        self.run_config = RunConfig(
            max_workers=1,     # serialize — one call at a time, no bursts. THE key knob.
            timeout=180,
            max_retries=10,    # on a 429, wait and retry instead of returning nan
            max_wait=60,       # cap the backoff
        )

    def evaluate(self, questions, references, answers, contexts):
        dataset = Dataset.from_dict({
            "user_input": list(questions),
            "reference": list(references),
            "response": answers,
            "retrieved_contexts": contexts,
        })
        return evaluate(
            dataset,
            llm=self.evaluator_llm,
            metrics=self.metrics,
            embeddings=self.embeddings,
            run_config=self.run_config,
        )

    def save_to_excel(self, eval_result, file_path: str, sheet_name: str) -> None:
        df = eval_result.to_pandas()
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
