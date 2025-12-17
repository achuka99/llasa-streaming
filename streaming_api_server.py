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
import re

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

STREAM_CHUNK_SIZE_CODES = 200   # Decode every ~200 codes for better audio quality
INITIAL_CHUNK_SIZE_CODES = 100  # Initial chunk slightly smaller to start streaming faster

app = FastAPI()

XCODEC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_NAME)
codec_model = codec_model.to(XCODEC_DEVICE).eval()
# Don't use half precision - causes issues with FFT operations
# if XCODEC_DEVICE == "cuda":
#     codec_model = codec_model.half()

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
    """Format prompt for Llasa TTS using chat template (text understanding -> speech generation)"""
    # Use the official Llasa format with chat template
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
    ]
    # Note: The vLLM API will handle tokenization, so we return the raw prompt
    # that includes the special tokens the model expects
    return f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|><|SPEECH_GENERATION_START|>"

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

    accumulated_text = ""
    speech_start_found = False
    processed_codes = 0
    first_chunk_yielded = False
    total_codes_found = 0

    async for chunk in response_stream:
        if chunk.choices and chunk.choices[0].text:
            new_text = chunk.choices[0].text
            accumulated_text += new_text
            print(f"[STREAM] Received text chunk: {new_text[:100]}")

            # Parse codes from <|s_XXXX|> anywhere in the accumulated text
            codes = re.findall(r'<\|s_(\d+)\|>', accumulated_text)
            valid_codes = [int(c) for c in codes if 0 <= int(c) < 65536]
            
            # If we find codes, we've started speech generation (even if marker not found explicitly)
            if valid_codes and not speech_start_found:
                speech_start_found = True
                print(f"[SPEECH_START] Detected speech codes in stream")

            total_codes_found = len(valid_codes)
            print(f"[CODES_FOUND] Total codes parsed so far: {total_codes_found}")
            
            if total_codes_found > 0 and processed_codes == 0:
                print(f"[SAMPLE_CODES] First 5 codes: {valid_codes[:5]}")

            current_codes = len(valid_codes)
            decode_chunk_size = INITIAL_CHUNK_SIZE_CODES if not first_chunk_yielded else STREAM_CHUNK_SIZE_CODES

            if current_codes >= processed_codes + decode_chunk_size:
                codes_to_decode = valid_codes[processed_codes: processed_codes + decode_chunk_size]
                print(f"[DECODING] Decoding {len(codes_to_decode)} codes (processed: {processed_codes}, total: {current_codes})")
                
                # Convert codes to tensor and decode
                codes_tensor = torch.tensor(codes_to_decode, device=XCODEC_DEVICE, dtype=torch.long).unsqueeze(0).unsqueeze(0)
                
                try:
                    with torch.no_grad():
                        audio_wave = codec_model.decode_code(codes_tensor)

                    audio_np = audio_wave[0, 0].cpu().float().numpy()
                    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                    pcm_bytes = audio_int16.tobytes()
                    
                    print(f"[AUDIO] Generated {len(pcm_bytes)} bytes of audio ({len(codes_to_decode)} codes)")

                    if pcm_bytes:
                        yield pcm_bytes
                        first_chunk_yielded = True
                        print(f"[YIELD] Yielded audio chunk")
                except Exception as e:
                    print(f"[ERROR] Decoding error: {e}")
                    raise

                processed_codes += decode_chunk_size

    # Final remaining codes
    print(f"[STREAM_END] Speech start found: {speech_start_found}, Total codes: {total_codes_found}, Processed: {processed_codes}")
    if speech_start_found:
        codes = re.findall(r'<\|s_(\d+)\|>', accumulated_text)
        remaining_codes = [int(c) for c in codes if 0 <= int(c) < 65536]
        print(f"[FINAL] Remaining codes to process: {len(remaining_codes) - processed_codes}")
        if len(remaining_codes) > processed_codes:
            final_codes = remaining_codes[processed_codes:]
            if final_codes:
                print(f"[FINAL_DECODE] Decoding final {len(final_codes)} codes")
                codes_tensor = torch.tensor(final_codes, device=XCODEC_DEVICE, dtype=torch.long).unsqueeze(0).unsqueeze(0)
                try:
                    with torch.no_grad():
                        audio_wave = codec_model.decode_code(codes_tensor)
                    audio_np = audio_wave[0, 0].cpu().float().numpy()
                    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
                    pcm_bytes = audio_int16.tobytes()
                    print(f"[FINAL_AUDIO] Generated {len(pcm_bytes)} bytes of final audio")
                    if pcm_bytes:
                        yield pcm_bytes
                except Exception as e:
                    print(f"[ERROR] Final decoding error: {e}")
                    raise
        else:
            print(f"[FINAL] No remaining codes (processed: {processed_codes}, total: {len(remaining_codes)})")

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