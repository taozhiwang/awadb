from awadb.llm_embedding.base import Embeddings

# Use all-mpnet-base-v2 as the default model
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self):
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def Embedding(self, sentence):
        tokens = []
        if self.tokenizer != None:
            tokens = self.tokenizer.tokenize(sentence)
        else:
            tokens.append(sentence)
        return self.model.encode(tokens[0])

    def EmbeddingBatch(
        self,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            results.append(self.model.encode(text))
        return results