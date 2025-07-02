from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import os

# Load Hugging Face token dari .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Inisialisasi FastAPI dan pipeline diarization
app = FastAPI(title="Diarization API")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

@app.get("/")
async def home():
    return {"message": "🟢 Diarization API is running. Use POST /diarize-audio to upload audio."}

@app.post("/diarize-audio")
async def diarize_audio(file: UploadFile = File(...)):
    # Simpan file audio sementara
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())

    try:
        # Tambahkan spacer agar pyannote tidak skip awal audio
        audio = AudioSegment.from_wav(temp_audio_path)
        spacer = AudioSegment.silent(duration=2000)
        audio = spacer.append(audio, crossfade=0)
        audio.export(temp_audio_path, format="wav")

        # Jalankan diarization
        dz = pipeline(temp_audio_path)
        results = []
        for turn, _, speaker in dz.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })

        return {"segments": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        os.remove(temp_audio_path)
