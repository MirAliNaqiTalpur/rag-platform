from app.rag.generator import MockGenerator

class GeneratorRegistry:
    def __init__(self):
        self._registry = {
            "mock": MockGenerator,
            "gemini": "gemini",  # lazy loading
        }

    def get(self, name):
        if name == "mock":
            return MockGenerator()

        elif name == "gemini":
            # import only when needed
            from app.rag.gemini_generator import GeminiGenerator
            return GeminiGenerator()

        else:
            raise ValueError(f"Unknown generator: {name}")