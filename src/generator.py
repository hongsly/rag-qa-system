import ollama
from openai import OpenAI, OpenAIError
from src.utils import Chunk

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a Q&A assistant for RAG (retrieval-augmented generation) research. "
    + "Answer ONLY using information from the provided support material in <documents>...</documents>. "
    + "When citing information, ALWAYS include reference to the source <document> using its <metadata>. "
    + "If the support <documents> do not contain enough information to answer the question, respond with: "
    + "'I don't have enough information in the provided materials to answer this question. '"
    + "DO NOT use your general knowledge - only cite the support material. "
)

SYSTEM_PROMPT_WITHOUT_CONTEXT = (
    "You are a Q&A assistant for RAG (retrieval-augmented generation) research. "
    + "Answer the user's question in RAG area. "
)


class Generator:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key) if api_key else None

    def generate(
        self,
        query: str,
        context: list[Chunk] | None = None,
        model: str = "gpt-4o-mini",
        retrieval_mode: str = "none",
    ) -> str:
        instructions = (
            SYSTEM_PROMPT_WITH_CONTEXT if context else SYSTEM_PROMPT_WITHOUT_CONTEXT
        )
        prompt = self._get_prompt(query, context)
        if model.startswith("gpt-") and self.client is not None:
            return self._get_openai_response(
                instructions, prompt, model, retrieval_mode
            )
        else:
            return self._get_ollama_response(instructions, prompt, model)

    def _get_openai_response(
        self, system_prompt: str, prompt: str, model: str, retrieval_mode: str
    ):
        try:
            response = self.client.responses.create(
                model=model,
                input=prompt,
                instructions=system_prompt,
                metadata={"retrieval_mode": retrieval_mode},
            )
            return response.output_text
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _get_ollama_response(self, system_prompt: str, prompt: str, model: str):
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                options={"num_ctx": 8192},
            )
            return response.response
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _get_prompt(self, query: str, context: list[Chunk] | None = None):
        prompt = f"<question>{query}</question>"
        if context:
            prompt += "\n<documents>\n"
            for c in context:
                prompt += f' <document id="{c["chunk_id"]}">\n'
                prompt += "  <metadata>\n"
                prompt += f'   <title>{c["metadata"]["title"]}</title>\n'
                prompt += (
                    f'   <authors>{",".join(c["metadata"]["authors"])}</authors>\n'
                )
                prompt += f'   <year>{c["metadata"]["year"]}</year>\n'
                prompt += "  </metadata>\n"
                prompt += f'  <content>{c["chunk_text"]}</content>\n'
                prompt += " </document>\n"
            prompt += "</documents>"
        return prompt
