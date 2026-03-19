import os
import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

router = APIRouter()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ==============================
# REQUEST
# ==============================


class QuizRequest(BaseModel):
    material_id: str
    user_id: str
    type: str = "multiple_choice"
    difficulty: str = "medium"
    count: int = 5


# ==============================
# HELPER
# ==============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def clean_json_output(raw_output: str):
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)

        if match:
            json_str = match.group(1)
        else:
            start = raw_output.find("[")
            end = raw_output.rfind("]")
            json_str = raw_output[start:end + 1]

        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json.loads(json_str.strip())

    except Exception as e:
        return {
            "raw_output": raw_output,
            "parse_error": str(e)
        }


# ==============================
# ENDPOINT
# ==============================
@router.post("/quiz")
async def generate_quiz(request: QuizRequest):
    try:
        count = max(1, min(request.count, 20))

        # ==============================
        # TYPE & DIFFICULTY
        # ==============================
        difficulty_map = {
            "easy": "mudah — konsep dasar",
            "medium": "sedang — aplikasi konsep",
            "hard": "sulit — analisis dan evaluasi"
        }

        type_map = {
            "multiple_choice": {
                "desc": "PILIHAN GANDA",
                "format": f"""
[
  {{
    "question": "...",
    "options": {{"A":"...","B":"...","C":"...","D":"..."}},
    "correct_answer": "A",
    "explanation": "...",
    "difficulty": "{request.difficulty}"
  }}
]
"""
            },
            "true_false": {
                "desc": "BENAR/SALAH",
                "format": f"""
[
  {{
    "question": "...",
    "options": {{"A":"Benar","B":"Salah"}},
    "correct_answer": "A",
    "explanation": "...",
    "difficulty": "{request.difficulty}"
  }}
]
ATURAN:
A = BENAR
B = SALAH
"""
            },
            "essay": {
                "desc": "ESAI",
                "format": f"""
[
  {{
    "question": "...",
    "key_points": ["..."],
    "sample_answer": "...",
    "difficulty": "{request.difficulty}"
  }}
]
"""
            }
        }

        selected = type_map.get(request.type, type_map["multiple_choice"])

        # ==============================
        # QDRANT + EMBEDDING
        # ==============================
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="tutorku_materials",
            embedding=embeddings
        )

        q_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.material_id",
                    match=MatchValue(value=request.material_id)
                )
            ]
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"filter": q_filter, "k": 15}
        )

        # ==============================
        # LLM (SUMOPOD)
        # ==============================
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4000,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        # ==============================
        # PROMPT
        # ==============================
        system_prompt = """
Kamu adalah pembuat soal profesional.

ATURAN KETAT:
- Output HARUS JSON VALID
- TANPA markdown
- TANPA penjelasan tambahan
- Jumlah soal = {count}
- Ikuti format EXACT

TIPE: {type_desc}
KESULITAN: {difficulty_desc}

FORMAT:
{format_template}

Jika konteks kosong, tetap buat soal umum sesuai topik.

KONTEKS:
{context}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
                "count": lambda _: count,
                "type_desc": lambda _: selected["desc"],
                "difficulty_desc": lambda _: difficulty_map.get(request.difficulty),
                "format_template": lambda _: selected["format"],
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        raw_output = rag_chain.invoke("Buat soal sekarang.")

        # ==============================
        # PARSE
        # ==============================
        questions = clean_json_output(raw_output)

        return {
            "material_id": request.material_id,
            "user_id": request.user_id,
            "type": request.type,
            "difficulty": request.difficulty,
            "count": len(questions) if isinstance(questions, list) else 0,
            "questions": questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
