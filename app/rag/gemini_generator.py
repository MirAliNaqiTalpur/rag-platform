import os

from google import genai
from google.genai import errors as genai_errors

from app.rag.base_generator import BaseGenerator


class ModelBusyError(Exception):
    pass


class UpstreamModelError(Exception):
    pass


class GeminiGenerator(BaseGenerator):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)

        self.default_model = (
            os.getenv("DEFAULT_MODEL")
            or os.getenv("DEFAULT_GEMINI_MODEL")
            or "gemini-3.1-flash-lite-preview"
        ).strip()

        raw_models = (
            os.getenv("AVAILABLE_MODELS")
            or os.getenv("AVAILABLE_GEMINI_MODELS")
            or self.default_model
        )

        self.allowed_models = {
            model.strip() for model in raw_models.split(",") if model.strip()
        }

        if self.default_model not in self.allowed_models:
            self.allowed_models.add(self.default_model)

    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def get_allowed_models(self):
        return sorted(self.allowed_models)

    def resolve_model(self, model_name=None):
        allow_custom = os.getenv("ALLOW_CUSTOM_MODELS", "false").strip().lower() == "true"

        chosen = (model_name or self.default_model).strip()

        if not chosen:
            raise ValueError("No Gemini model was provided and no default model is configured.")

        if chosen in self.allowed_models:
            return chosen

        if allow_custom:
            return chosen

        raise ValueError(
            f"Unsupported Gemini model: {chosen}. "
            f"Allowed models: {sorted(self.allowed_models)}"
        )

    def generate(self, query, documents, model_name=None):
        context = "\n\n".join(self._doc_text(doc) for doc in documents)

        prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        chosen_model = self.resolve_model(model_name)

        try:
            response = self.client.models.generate_content(
                model=chosen_model,
                contents=prompt,
            )
            return response.text if response.text else "No response returned by Gemini."

        except genai_errors.ServerError as e:
            message = str(e)
            if "503" in message or "UNAVAILABLE" in message:
                raise ModelBusyError(
                    f"Gemini model '{chosen_model}' is temporarily experiencing high demand. Please try again shortly or switch to another model."
                ) from e
            raise UpstreamModelError(
                f"Gemini server error while using model '{chosen_model}'."
            ) from e