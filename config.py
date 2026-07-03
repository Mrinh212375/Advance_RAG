import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    embedding_model: str
    llm_model: str
    eval_llm_model: str
    reranker_model: str
    collection_name: str
    persist_directory: str
    pdf_path: str
    golden_set_path: str
    eval_output_file: str
    eval_sheet_name: str


def load_config() -> Config:
    with open(Path(__file__).parent / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    load_dotenv(cfg["paths"]["env_file"])

    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        embedding_model=cfg["models"]["embedding_model"],
        llm_model=cfg["models"]["llm_model"],
        eval_llm_model=cfg["models"]["eval_llm_model"],
        reranker_model=cfg["models"]["reranker_model"],
        collection_name=cfg["vectorstore"]["collection_name"],
        persist_directory=cfg["vectorstore"]["persist_directory"],
        pdf_path=cfg["paths"]["pdf_path"],
        golden_set_path=cfg["paths"]["golden_set_path"],
        eval_output_file=cfg["evaluation"]["output_file"],
        eval_sheet_name=cfg["evaluation"]["sheet_name"],
    )
