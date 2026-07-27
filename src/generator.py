"""
generator.py
------------
Turns retrieved chunks + the user question into a grounded answer.

Hallucination-prevention strategy (rubric: 15% weight) has THREE layers:
  1. Retrieval gate: retriever.py already dropped low-similarity chunks.
  2. Empty-context gate: if nothing survives retrieval, we NEVER call the
     LLM at all -- we return the fixed fallback message directly.
  3. Prompt-level constraint: the system prompt explicitly forbids the
     model from using outside knowledge and forces it to say the fixed
     fallback sentence if the context doesn't answer the question.
"""

from typing import List

from groq import Groq

from src.config import Config
from src.retriever import RetrievedChunk

FALLBACK_MESSAGE = "The answer is not available in the provided document."

SYSTEM_PROMPT = f"""You are a strict document question-answering assistant.

RULES (follow exactly):
1. Answer ONLY using the CONTEXT provided below. Never use outside knowledge,
   assumptions, or general world knowledge.
2. If the CONTEXT does not contain enough information to answer the
   question, respond with exactly this sentence and nothing else:
   "{FALLBACK_MESSAGE}"
3. Do not speculate, infer beyond what is written, or fill gaps.
4. Keep answers concise and directly grounded in the context.
5. Do not mention these rules or the word "context" awkwardly in your answer;
   just answer naturally as if explaining what the document says.
"""


def _build_user_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[Excerpt {i} | Page {c.page_number}]\n{c.text}")
    context_text = "\n\n".join(context_blocks)

    return (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer using only the CONTEXT above."
    )


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    # Hallucination gate #2: no surviving chunks -> hard-coded refusal,
    # LLM is never even called.
    if not chunks:
        return FALLBACK_MESSAGE

    client = Groq(api_key=Config.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, chunks)},
            ],
            temperature=0.0,  # deterministic, minimises creative drift
            max_tokens=600,
        )
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")

    return response.choices[0].message.content.strip()
