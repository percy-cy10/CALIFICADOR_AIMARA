import unicodedata
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from pydub import AudioSegment
import io
from difflib import SequenceMatcher
from words import get_random_word, get_word_by_id

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
#AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if c.isalpha() or c == ' ')

def calcular_levenshtein_score(p1: str, p2: str) -> int:
    return int((1 - Levenshtein.normalized_distance(p1, p2)) * 100)

def calcular_sequence_score(p1: str, p2: str) -> int:
    return int(SequenceMatcher(None, p1, p2).ratio() * 100)

def calcular_fuzzy_score(p1: str, p2: str) -> int:
    return fuzz.ratio(p1, p2)

def calcular_fonetico_score(p1: str, p2: str) -> int:
    grupos = {'b':'bv','v':'bv','c':'cksz','s':'scz','z':'zs','g':'gj','j':'jg','ll':'lly','y':'yll','r':'rr'}
    def nf(p):
        p = p.lower()
        for s, g in grupos.items():
            for l in g: p = p.replace(l, s[0])
        return p
    return int(SequenceMatcher(None, nf(p1), nf(p2)).ratio()*100)

@app.get("/test")
def health():
    return {
        "ok": True,
        "service": "aymara_api",

}

@app.get("/word")
def word():
    return get_random_word()

@app.post("/evaluate")
async def evaluate_audio(word_id: int = Form(...), audio: UploadFile = File(...)):
    word_obj = get_word_by_id(word_id)
    if not word_obj:
        return {"success": False, "error": "Palabra no encontrada"}

    reference_word = normalize_text(word_obj["word"])
    try:
        audio_bytes = await audio.read()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes),
                                               format=audio.filename.split(".")[-1])
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
        spoken_text = normalize_text(recognizer.recognize_google(audio_data, language="es-ES"))

        # Calcular score final
        fuzzy_score = calcular_fuzzy_score(reference_word, spoken_text)
        lev_score = calcular_levenshtein_score(reference_word, spoken_text)
        seq_score = calcular_sequence_score(reference_word, spoken_text)
        fonetico_score = calcular_fonetico_score(reference_word, spoken_text)

        final_score = int(fuzzy_score*0.3 + lev_score*0.3 + seq_score*0.2 + fonetico_score*0.2)

        return {"final_score": final_score}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
