import random

# Lista de palabras con ID
WORDS = [
    {"id": 1, "word": "hola"},
    {"id": 2, "word": "perro"},
    {"id": 3, "word": "carro"},
    {"id": 4, "word": "casa"},
    {"id": 5, "word": "amigo"},
    {"id": 6, "word": "jugar"}
]

def get_random_word():
    return random.choice(WORDS)

def get_word_by_id(word_id: int):
    for w in WORDS:
        if w["id"] == word_id:
            return w
    return None
