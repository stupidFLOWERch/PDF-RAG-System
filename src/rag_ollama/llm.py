import os
import ollama
from google import genai

class LLMClient:

    def __init__(self):

        self.provider = os.getenv(
            "LLM_PROVIDER",
            "ollama"
        )

        # Gemini client
        if self.provider == "gemini":

            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set"
                )

            self.client = genai.Client(
                api_key=api_key
            )

    def generate(
        self,
        messages,
        options=None
    ):

        if self.provider == "ollama":

            return self._ollama(
                messages,
                options
            )

        elif self.provider == "gemini":

            return self._gemini(
                messages,
                options
            )

        else:

            raise ValueError(
                f"Unknown LLM provider: {self.provider}"
            )

    # ==========================================
    # Ollama
    # ==========================================

    def _ollama(
        self,
        messages,
        options=None
    ):

        response = ollama.chat(

            model=os.getenv(
                "OLLAMA_MODEL",
                "llama3:latest"
            ),

            messages=messages,

            options=options or {
                "temperature": 0.1
            }
        )

        return response["message"]["content"].strip()

    # ==========================================
    # Gemini
    # ==========================================

    def _gemini(
        self,
        messages,
        options=None
    ):

        # Convert your RAG messages into one prompt
        prompt = ""

        for message in messages:

            role = message["role"]
            content = message["content"]

            prompt += (
                f"{role.upper()}:\n"
                f"{content}\n\n"
            )

        temperature = 0.1

        if options:
            temperature = options.get(
                "temperature",
                0.1
            )

        response = self.client.models.generate_content(

            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.1-flash-lite"
            ),

            contents=prompt,

            config={
                "temperature": temperature
            }
        )

        return response.text.strip()