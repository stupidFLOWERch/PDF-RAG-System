from langchain_core.documents import Document as LCDocument
from  langchain_text_splitters import TextSplitter

from typing import List
import tiktoken

class TokenTextSplitter(TextSplitter):
    """
    自定义 Token 分块器。
    用 tiktoken 精确控制 Token 数量，适合学术论文等长文档。
    
    Usage:
        splitter = TokenTextSplitter(chunk_size=700, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
    """
    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        encoding_name: str = "cl100k_base",
        **kwargs
    ):

        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str) -> List[str]:
        token_ids = self.encoder.encode(text)

        if len(token_ids) < self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids)) 
            token_slice= token_ids[start: end]

            chunk_text = self.encoder.decode(token_slice).strip()
            if chunk_text:
                chunks.append(chunk_text)

            start += step

        return chunks
