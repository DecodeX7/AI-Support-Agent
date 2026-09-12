import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMProvider:

    def __init__(self):

        self.provider = os.getenv(
            "LLM_PROVIDER",
            "ollama"
        ).lower()

        self.ollama_model = os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b"
        )

        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        self.groq_model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )

        self.groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if self.provider == "groq":

            if not self.groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is required "
                    "when LLM_PROVIDER=groq"
                )

            self.client = OpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

        elif self.provider == "ollama":

            self.client = None

        else:

            raise ValueError(
                f"Unsupported LLM_PROVIDER: "
                f"{self.provider}"
            )


    def generate(
        self,
        system_prompt,
        user_prompt
    ):

        if self.provider == "ollama":

            return self._generate_ollama(
                system_prompt,
                user_prompt
            )

        if self.provider == "groq":

            return self._generate_groq(
                system_prompt,
                user_prompt
            )

        raise RuntimeError(
            "Invalid LLM provider"
        )


    def _generate_ollama(
        self,
        system_prompt,
        user_prompt
    ):

        url = (
            f"{self.ollama_base_url}"
            "/api/chat"
        )

        payload = {
            "model": self.ollama_model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            "stream": False,

            "options": {
                "temperature": 0.2
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "message"
        ][
            "content"
        ].strip()


    def _generate_groq(
        self,
        system_prompt,
        user_prompt
    ):

        response = (
            self.client.chat.completions.create(
                model=self.groq_model,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.2,

                max_tokens=300
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )