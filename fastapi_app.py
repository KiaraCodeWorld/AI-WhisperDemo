from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoFeatureExtractor
#import whisper
import torch
from typing import List

# Checking if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model:#
#model = whisper.load_model("base", device=DEVICE)
#model = whisper.load_model("tiny.en.pt", device=DEVICE)

model_id = "./huggigface-whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

config = model.config
print(config)
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, config=config)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

app = FastAPI()

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    # For each file, let's store the results in a list of dictionaries.
    results = []

    for file in files:
        # Create a temporary file.
        with NamedTemporaryFile(delete=True) as temp:
            # Write the user's uploaded file to the temporary file.
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())
            
            # Let's get the transcript of the temporary file.
            result = pipe(temp.name) #model.transcribe(temp.name)

            # Now we can store the result object for this file.
            results.append({
                'filename': file.filename,
                'transcript': result['text'],
            })

    return JSONResponse(content={'results': results})


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
