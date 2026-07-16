from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dataloader import DataLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI

import config


class Chunker:

    def __init__(self, file2chunk) -> None:
        self.file_path = file2chunk
        self.dataloader = DataLoader(self.file_path)

    def static_splitter(self) -> list[Document]:
        # doc_content = DataLoader()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_documents(self.dataloader.load_file_content())
    

    def docling_section_aware_splitter(self):
        
        loader = DoclingLoader(
        file_path=self.file_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(),
        )
        chunks = loader.load()

        return chunks  

    def MarkdownHeader_section_aware_splitter(self):

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Heading"), ("##", "Sections"),("###", "Subsections")]
        )
        chunks = splitter.split_text(self.dataloader.load_content_as_Md())   # each chunk tagged with its header in metadata

        return chunks

    
    def custom_section_aware_splitter(self,threshold=1500):

        md_content = self.dataloader.load_content_as_Md()

        def is_article_heading(line):
            """True for '## **13. TERMINATION...**' or '**7. TITLE AND RISK**'."""
            s = line.strip()
            while s.startswith("#"):          # drop any leading '#'
                s = s[1:].strip()
            if not s.startswith("**"):        # article headings are bold
                return False
            s = s[2:].strip()                 # drop the opening '**'
            number = s.split(".")[0].strip()  # text before the first dot
            return number.isdigit() and len(number) <= 2   # '13' yes, '1.14' no


        def read_article_heading(line):
            """Pull (number, title) out of a heading line."""
            s = line.strip().lstrip("#").strip().strip("*").strip()
            number, _, title = s.partition(".")   # split on the first dot only
            return number.strip(), title.strip()


        def is_subsection(line):
            """True for '- 13.1. ...' or '5.10. ...' but NOT '1.14.1.'."""
            s = line.strip()
            if s.startswith("-"):
                s = s[1:].strip()
            if not s:
                return False
            token = s.split()[0].rstrip(".")   # first word, e.g. '13.1'
            parts = token.split(".")
            return len(parts) == 2 and all(p.isdigit() for p in parts)


        def is_noise(line):
            """Page-footer text or a lone page number."""
            s = line.strip()
            if "Schneider Electric-General Terms" in s:
                return True
            return s.isdigit() and len(s) <= 2


        # ---------- splitting + packing (plain loops) ----------

        def split_into_articles(text):
            """-> list of (number, title, body_lines)."""
            articles, current = [], None
            for line in text.splitlines():
                if is_article_heading(line):
                    if current:
                        articles.append(current)
                    number, title = read_article_heading(line)
                    current = (number, title, [])
                elif current:
                    current[2].append(line)


            if current:
                articles.append(current)
            return articles


        def split_into_subsections(lines):
            """Start a new block at every 'N.M.' line; keep everything else together."""
            blocks, buffer = [], []
            for line in lines:
                if is_subsection(line) and buffer:
                    blocks.append("\n".join(buffer).strip())
                    buffer = [line]
                else:
                    buffer.append(line)
            if buffer:
                blocks.append("\n".join(buffer).strip())
            return [b for b in blocks if b]


        def pack(pieces, threshold):
            """Fill a 'bag' with pieces; tie it off and start fresh when it would overflow."""
            groups, bag = [], ""
            for piece in pieces:
                if bag and len(bag) + len(piece) + 1 > threshold:
                    groups.append(bag)
                    bag = piece
                else:
                    bag = piece if not bag else bag + "\n" + piece
            if bag:
                groups.append(bag)
            return groups


        # ---------- the chunker ----------

        class SectionAwareChunker:
            def __init__(self, markdown_text, threshold):
                self.text = markdown_text
                self.threshold = threshold

            def _make(self, heading, content, number, title):
                return Document(
                    page_content=f"{heading}\n\n{content}".strip(),
                    metadata={"article": number, "article_title": title},
                )

            def split(self):
                chunks = []
                for number, title, body in split_into_articles(self.text):
                    heading = f"## {number}. {title}"
                    body_lines = [ln for ln in body if not is_noise(ln)]
                    body_text = "\n".join(body_lines).strip()

                    # small enough -> keep the whole article as one chunk
                    if len(heading) + len(body_text) <= self.threshold:
                        chunks.append(self._make(heading, body_text, number, title))
                        continue

                    # too big -> pack whole subsections, never cutting one in half
                    # (a lone subsection bigger than threshold simply stays whole)
                    for group in pack(split_into_subsections(body_lines), self.threshold):
                        chunks.append(self._make(heading, group, number, title))
                return chunks
            
        chunker = SectionAwareChunker(md_content,threshold)

        return chunker.split()

    #### This is Anthropic's contextual retrieval(basically a chunking strategy to prepend contexts before every chunks)
    def contextual_chunking(self):

        cfg = config.load_config()

        contextual_creator_model = ChatOpenAI(
                    model=cfg.open_ai_model,
                    temperature=0,
                    max_tokens=1024,
                    max_retries=2,
                    api_key=cfg.openai_api_key
                )


        entire_doc = self.dataloader.load_file_content()
        chunks = self.custom_section_aware_splitter()
        # print(entire_doc)

        prompt = '''
        Here is the whole document:
        <document>
        {WHOLE_DOCUMENT}
        </document>

        Here is the chunk we want to situate within the whole document
        <chunk>
        {CHUNK_CONTENT}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        Don't include any special character in your response,just normal text is enough.
        '''

        for i in chunks:
            print("calling\n")
            response = contextual_creator_model.invoke(prompt.format(WHOLE_DOCUMENT = entire_doc[0].page_content, CHUNK_CONTENT = i.page_content))
            i.page_content = response.content + "\n" + i.page_content
        
        return chunks


    ### Langchain Unstructured
    # def unstructured_section_aware_splitter():

if __name__=="__main__":

    ch = Chunker(r"D:\Learning\DocsForRAG\SDI PO TC-India.pdf")
    chunks = ch.contextual_chunking()

    for i in chunks:
        print(f"\n\n{i.page_content}")




