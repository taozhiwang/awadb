from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, Field
from langchain.embeddings.base import Embeddings

class AwaEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that awadb library is installed."""

        try:
            from awadb import AwaEmbedding
        except ImportError as exc:
            raise ImportError(
                "Could not import awadb library. "
                "Please install it with `pip install awadb`"
            ) from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.EmbeddingBatch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.Embedding(text)