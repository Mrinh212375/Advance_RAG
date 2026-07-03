from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pymupdf4llm


class DataLoader:

    def __init__(self, file_path) -> None:
        self.file = file_path

    def load_file_content(self) -> list[Document]:

        loader = PyPDFLoader(file_path=self.file)
        content = loader.load()
        return content
    
    def load_content_as_Md(self) -> str:
        
        md_content = pymupdf4llm.to_markdown(self.file)
        return md_content
