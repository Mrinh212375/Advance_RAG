from dataloader import DataLoader
from chunking import Chunker
from vectorstore import VectorStore

import config


def run_ingestion(vs: VectorStore, pdf_path: str, collection_name: str) -> None:

    if vs.is_empty():

        # docs = DataLoader(pdf_path).load_file_content()
        chunks = Chunker(pdf_path).custom_section_aware_splitter()
        vs.ingest(chunks)
        print(f"Ingested {len(chunks)} chunks into '{collection_name}'.")
    else:
        print(f"Vector store '{collection_name}' already populated — skipping ingestion.")


if __name__ == "__main__":
    cfg = config.load_config()
    vs = VectorStore(cfg.embedding_model, cfg.collection_name, cfg.persist_directory)
    run_ingestion(vs, cfg.pdf_path, cfg.collection_name)
