import os
import torch
from xcodec2.modeling_xcodec2 import XCodec2Model
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import asyncio
import functools
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Llasa-1B")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "your_hf_token_if_needed")
XCODEC_MODEL_NAME = os.environ.get("XCODEC_MODEL_NAME", "HKUSTAudio/xcodec2")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")  # vLLM often uses EMPTY or a dummy key

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 2000
DEFAULT_REPETITION_PENALTY = 1.1

# Special tokens for Llasa (based on standard examples; adjust if your model differs)
TEXT_START = "<|TEXT_UNDERSTANDING_START|>"
TEXT_END = "<|TEXT_UNDERSTANDING_END|>"
SPEECH_START = "<|SPEECH_GENERATION_START|>"
SPEECH_END = "<|SPEECH_GENERATION_END|>"

AUDIO_SAMPLERATE = 16000  # XCodec2 uses 16kHz
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

STREAM_CHUNK_SIZE_CODES = 50   # Decode every ~50 codes for low latency
INITIAL_CHUNK_SIZE_CODES = 10

app = FastAPI()

XCODEC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_NAME)
codec_model = codec_model.to(XCODEC_DEVICE).eval()
if XCODEC_DEVICE == "cuda":
    codec_model = codec_model.half()

print(f"XCodec2 model loaded on {XCODEC_DEVICE}")
print(f"Tokenizer loaded from {MODEL_NAME}")
print(f"Connected to vLLM at {VLLM_BASE_URL}")

class AudioRequest(BaseModel):
    text: str
    voice: str = "default"  # Not used in base Llasa (no voice prompt), kept for compatibility
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

def format_prompt(text: str):
    """Format prompt for Llasa TTS (text understanding -> speech generation)"""
    return f"{TEXT_START}{text}{TEXT_END}"

async def generate_audio_chunks(
    text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float
):
    loop = asyncio.get_running_loop()
    formatted_prompt = format_prompt(text)
    print(f"Formatted Prompt: {formatted_prompt[:200]}...")

    stream_kwargs = {
        "model": MODEL_NAME,
        "prompt": formatted_prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [SPEECH_END],
        "stream": True,
        "extra_body": {"repetition_penalty": repetition_penalty},
    }

    response_stream = await client.completions.create(**stream_kwargs)

    accumulated_tokens = []
    speech_start_found = False
    processed_codes = 0
    first_chunk_yielded = False

    async for chunk in response_stream:
        if chunk.choices and chunk.choices[0].text:
            new_text = chunk.choices[0].text
            new_ids = tokenizer.encode(new_text)
            accumulated_tokens.extend(new_ids)

            if not speech_start_found:
                try:
                    speech_start_idx = accumulated_tokens.index(tokenizer.convert_tokens_to_ids(SPEECH_START))
                    speech_start_found = True
                    code_tokens = accumulated_tokens[speech_start_idx + 1:]
                except ValueError:
                    continue
            else:
                code_tokens = accumulated_tokens[tokenizer.convert_tokens_to_ids(SPEECH_START) + 1:]

            # Llasa uses single codebook: codes are directly the token ids (0 to 65535)
            valid_codes = [c for c in code_tokens if 0 <= c < 65536]

            current_codes = len(valid_codes)
            decode_chunk_size = INITIAL_CHUNK_SIZE_CODES if not first_chunk_yielded else STREAM_CHUNK_SIZE_CODES

            if current_codes >= processed_codes + decode_chunk_size:
                codes_to_decode = valid_codes[processed_codes: processed_codes + decode_chunk_size]
                codes_tensor = torch.tensor(codes_to_decode, device=XCODEC_DEVICE).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    audio_wave = codec_model.decode_code(codes_tensor)

                audio_np = audio_wave[0, 0].cpu().float().numpy()
                audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()

                if pcm_bytes:
                    yield pcm_bytes
                    first_chunk_yielded = True

                processed_codes += decode_chunk_size

    # Final remaining codes
    if speech_start_found:
        remaining_codes = [c for c in accumulated_tokens[accumulated_tokens.index(tokenizer.convert_tokens_to_ids(SPEECH_START)) + 1:] if 0 <= c < 65536]
        if len(remaining_codes) > processed_codes:
            final_codes = remaining_codes[processed_codes:]
            if final_codes:
                codes_tensor = torch.tensor(final_codes, device=XCODEC_DEVICE).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    audio_wave = codec_model.decode_code(codes_tensor)
                audio_np = audio_wave[0, 0].cpu().float().numpy()
                audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()
                if pcm_bytes:
                    yield pcm_bytes

@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            text = message.get("input", "")
            segment_id = message.get("segment_id", "default")

            if not text:
                break

            await websocket.send_json({"type": "start", "segment_id": segment_id})

            async for audio_chunk in generate_audio_chunks(
                text=text,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                repetition_penalty=DEFAULT_REPETITION_PENALTY
            ):
                await websocket.send_bytes(audio_chunk)

            await websocket.send_json({"type": "end", "segment_id": segment_id})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/v1/audio/speech/stream")
async def http_audio_stream(request: AudioRequest):
    async def stream_pcm():
        async for chunk in generate_audio_chunks(
            text=request.text,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty
        ):
            yield chunk

    return StreamingResponse(stream_pcm(), media_type="audio/pcm")

@app.get("/")
async def read_root():
    return {
        "message": "Llasa TTS Streaming API (using XCodec2)",
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream"
        },
        "note": "Run vLLM server separately: vllm serve unsloth/Llasa-1B --dtype bfloat16 --max-model-len 8192"
    }

if __name__ == "__main__":
    print("Starting FastAPI server for Llasa TTS streaming...")
    uvicorn.run("streaming_api_server:app", host="0.0.0.0", port=8002, reload=False)