from app.rag.simple_reranker import SimpleReranker
from app.rag.noop_reranker import NoOpReranker

class RerankerRegistry:
    def __init__(self):
        self._registry = {
            "simple": SimpleReranker,
            "none": NoOpReranker,
        }

    def get(self, name):
        if name is None or name == "":
            return None

        reranker_class = self._registry.get(name)

        if reranker_class is None:
            raise ValueError(f"Unknown reranker strategy: {name}")

        return reranker_class()