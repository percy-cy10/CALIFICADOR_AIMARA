import io
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import speech_recognition as sr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from words import get_random_word, get_word_by_id

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path(__file__).with_name("data.json")
_write_lock = Lock()


# -------------------- Helpers de texto / scoring --------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    # quita diacríticos (combining marks) y deja letras + espacios
    return "".join(c for c in text if c.isalpha() or c == " ")


def calcular_levenshtein_score(p1: str, p2: str) -> int:
    if not p1 and not p2:
        return 100
    if not p1 or not p2:
        return 0
    return int((1 - Levenshtein.normalized_distance(p1, p2)) * 100)


def calcular_sequence_score(p1: str, p2: str) -> int:
    if not p1 and not p2:
        return 100
    return int(SequenceMatcher(None, p1, p2).ratio() * 100)


def calcular_fuzzy_score(p1: str, p2: str) -> int:
    if not p1 and not p2:
        return 100
    return int(fuzz.ratio(p1, p2))


def calcular_fonetico_score(p1: str, p2: str) -> int:
    grupos = {
        "b": "bv",
        "v": "bv",
        "c": "cksz",
        "s": "scz",
        "z": "zs",
        "g": "gj",
        "j": "jg",
        "ll": "lly",
        "y": "yll",
        "r": "rr",
    }

    def nf(p: str) -> str:
        p = p.lower()
        for s, g in grupos.items():
            repl = s[0]
            for l in g:
                p = p.replace(l, repl)
        return p

    if not p1 and not p2:
        return 100
    return int(SequenceMatcher(None, nf(p1), nf(p2)).ratio() * 100)


def _guess_audio_format(filename: Optional[str], content_type: Optional[str]) -> str:
    # Prioriza extensión del filename
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower().strip()
        if ext:
            return ext

    # Fallback por content-type
    if content_type:
        ct = content_type.lower()
        if "wav" in ct:
            return "wav"
        if "mpeg" in ct or "mp3" in ct:
            return "mp3"
        if "mp4" in ct or "m4a" in ct:
            return "m4a"
        if "ogg" in ct:
            return "ogg"
        if "webm" in ct:
            return "webm"

    # Último recurso
    return "wav"


# -------------------- Endpoints básicos --------------------
@app.get("/test")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "aymara_api"}


@app.get("/word")
def word():
    return get_random_word()


@app.post("/evaluate")
async def evaluate_audio(word_id: int = Form(...), audio: UploadFile = File(...)):
    #word_obj = get_word_by_id(word_id)
    #word_obj = find_word_in_data(word_id)
    data = load_data()
    word_obj = find_word_in_data(data, word_id)
    if not word_obj:
        raise HTTPException(status_code=404, detail="Palabra no encontrada")

    reference_word = normalize_text(word_obj.get("aymara", ""))
    if not reference_word:
        raise HTTPException(status_code=500, detail="La palabra de referencia está vacía o inválida")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Archivo de audio vacío")

        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format="webm"  # 🔥 CLAVE
        )

        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)


        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)

        spoken_text_raw = recognizer.recognize_google(audio_data, language="es-ES")
        spoken_text = normalize_text(spoken_text_raw)

        fuzzy_score = calcular_fuzzy_score(reference_word, spoken_text)
        lev_score = calcular_levenshtein_score(reference_word, spoken_text)
        seq_score = calcular_sequence_score(reference_word, spoken_text)
        fonetico_score = calcular_fonetico_score(reference_word, spoken_text)

        final_score = int(
            fuzzy_score * 0.3
            + lev_score * 0.3
            + seq_score * 0.2
            + fonetico_score * 0.2
        )

        return {
            "final_score": final_score,
            "reference": reference_word,
            "spoken": spoken_text,
        }

    except HTTPException:
        raise
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="No se pudo reconocer el audio (muy bajo, ruido, etc.)")
    except sr.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Fallo en el servicio de reconocimiento: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {e}")


# -------------------- JSON categories (data.json) --------------------
def load_data() -> Dict[str, Any]:
    if not DATA_FILE.exists():
        empty: Dict[str, Any] = {"categories": []}
        save_data(empty)
        return empty

    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="data.json tiene JSON inválido")


def save_data(data: Dict[str, Any]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        tmp = DATA_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(DATA_FILE)


def find_category(data: Dict[str, Any], category_id: int) -> Optional[Dict[str, Any]]:
    for c in data.get("categories", []):
        if c.get("id") == category_id:
            return c
    return None

def find_word_in_data(data: Dict[str, Any], word_id: int) -> Optional[Dict[str, Any]]:
    for c in data.get("categories", []):
        for w in c.get("words", []):
            if w.get("id") == word_id:
                return w
    return None


@app.get("/categories")
def list_categories():
    data = load_data()
    return [{"id": c["id"], "name": c["name"]} for c in data.get("categories", [])]


@app.get("/categories/{category_id}")
def get_category(category_id: int):
    data = load_data()
    c = find_category(data, category_id)
    if not c:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"id": c["id"], "name": c["name"]}


@app.get("/categories/{category_id}/words")
def get_words_by_category(category_id: int):
    data = load_data()
    c = find_category(data, category_id)
    if not c:
        raise HTTPException(status_code=404, detail="Category not found")

    return [
        {
            "id": w["id"],
            "spanish": w["spanish"],
            "aymara": w["aymara"],
            "category_id": category_id,
        }
        for w in c.get("words", [])
    ]


@app.get("/export")
def export_json():
    if not DATA_FILE.exists():
        save_data({"categories": []})
    return FileResponse(DATA_FILE, media_type="application/json", filename="data.json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
