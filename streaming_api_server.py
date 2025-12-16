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
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Llasa-1B")
XCODEC2_MODEL_NAME = os.environ.get("XCODEC2_MODEL_NAME", "HKUST-Audio/xcodec2")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token123")

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Llasa/XCodec2 specific settings
DEFAULT_TEMPERATURE = 1.2
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_REPETITION_PENALTY = 1.0
SPEECH_TOKEN_PREFIX = "<|s_"
SPEECH_TOKEN_SUFFIX = "|>"
STOP_SEQUENCE = "<|SPEECH_GENERATION_END|>"
AUDIO_SAMPLERATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

# Streaming configuration - XCodec2 generates tokens at ~50Hz
# So we decode in smaller chunks for lower latency
STREAM_CHUNK_SIZE_TOKENS = 50  # About 1 second of audio
INITIAL_CHUNK_SIZE_TOKENS = 25  # First chunk smaller for faster start

app = FastAPI()
CODEC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize clients and models
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
xcodec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL_NAME)
xcodec_model = xcodec_model.to(CODEC_DEVICE).eval()
if CODEC_DEVICE == "cuda":
    xcodec_model = xcodec_model.half()

print(f"XCodec2 model loaded on {CODEC_DEVICE}")
print(f"Tokenizer loaded from {MODEL_NAME}")
print(f"Connected to vLLM at {VLLM_BASE_URL}")


class AudioRequest(BaseModel):
    text: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def format_prompt_for_llasa(prompt_text):
    """
    Format text for Llasa TTS model.
    Uses the training format: Convert text to speech with special tags.
    """
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{prompt_text}<|TEXT_UNDERSTANDING_END|>"
    full_prompt = f"Convert the text to speech:{formatted_text}<|SPEECH_GENERATION_START|>"
    return full_prompt


def extract_speech_ids_from_text(text):
    """
    Extract speech token IDs from generated text.
    Looks for tokens like <|s_12345|> and extracts the numeric ID.
    """
    import re
    speech_tokens = re.findall(r"<\|s_(\d+)\|>", text)
    speech_ids = [int(token_id) for token_id in speech_tokens]
    return speech_ids


def decode_xcodec_tokens_sync(speech_ids):
    """
    Decode XCodec2 speech tokens to audio waveform.
    XCodec2 uses single codebook, so shape is [batch, 1, sequence]
    """
    if not speech_ids:
        return torch.tensor([[]], device=CODEC_DEVICE, dtype=torch.float32)
    
    # Convert to tensor: [1, 1, num_tokens]
    codes_tensor = torch.tensor(speech_ids, dtype=torch.long, device=CODEC_DEVICE)
    codes_tensor = codes_tensor.unsqueeze(0).unsqueeze(0)
    
    # Decode to audio
    with torch.no_grad():
        audio_hat = xcodec_model.decode_code(codes_tensor)
    
    return audio_hat


def apply_fade(audio_tensor, fade_samples):
    """Applies fade-in and fade-out to audio tensor."""
    if audio_tensor is None or audio_tensor.numel() < 2 * fade_samples:
        return audio_tensor
    
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    fade_in = torch.linspace(0., 1., fade_samples, device=audio_tensor.device)
    fade_out = torch.linspace(1., 0., fade_samples, device=audio_tensor.device)
    
    audio_tensor[..., :fade_samples] *= fade_in
    audio_tensor[..., -fade_samples:] *= fade_out
    
    return audio_tensor.squeeze()


def convert_to_pcm16_bytes(audio_tensor, fade_ms=5):
    """Converts audio tensor to raw PCM 16-bit bytes."""
    if audio_tensor is None or audio_tensor.numel() == 0:
        return b''
    
    if fade_ms > 0:
        fade_samples = int(AUDIO_SAMPLERATE * fade_ms / 1000)
        fade_samples = (fade_samples // 2) * 2
        if fade_samples > 0:
            audio_tensor = apply_fade(audio_tensor.detach(), fade_samples)
    
    # Convert to int16 PCM
    audio_numpy = (audio_tensor.squeeze().cpu().to(torch.float32).numpy() * 32767)
    audio_numpy = np.clip(audio_numpy, -32768, 32767).astype(np.int16)
    
    return audio_numpy.tobytes()


async def generate_audio_chunks(
    text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float
):
    """
    Async generator that yields raw PCM audio chunks for streaming.
    Adapted from Orpheus pattern but for Llasa + XCodec2.
    """
    loop = asyncio.get_running_loop()
    
    try:
        # Format prompt for Llasa
        formatted_prompt = await loop.run_in_executor(
            None, functools.partial(format_prompt_for_llasa, text)
        )
        print(f"Formatted Prompt: {formatted_prompt[:100]}...")
        
        # Create streaming completion request
        stream_kwargs = dict(
            model=MODEL_NAME,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=[STOP_SEQUENCE],
            stream=True,
            extra_body={
                'repetition_penalty': repetition_penalty,
                'skip_special_tokens': False  # CRITICAL: Keep speech tokens
            },
        )
        
        response_stream = await client.completions.create(**stream_kwargs)
        
        accumulated_text = ""
        processed_token_count = 0
        first_chunk_yielded = False
        speech_started = False
        
        async for chunk in response_stream:
            if chunk.choices:
                chunk_text = chunk.choices[0].text or ""
                accumulated_text += chunk_text
                
                # Check if we've started generating speech tokens
                if not speech_started:
                    if SPEECH_TOKEN_PREFIX in accumulated_text:
                        speech_started = True
                        print("Speech token generation started")
                    else:
                        continue
                
                # Extract all speech IDs from accumulated text
                all_speech_ids = extract_speech_ids_from_text(accumulated_text)
                current_total_tokens = len(all_speech_ids)
                
                # Determine chunk size
                if not first_chunk_yielded:
                    current_chunk_size = INITIAL_CHUNK_SIZE_TOKENS
                else:
                    current_chunk_size = STREAM_CHUNK_SIZE_TOKENS
                
                # Process new tokens in chunks
                if current_total_tokens >= processed_token_count + current_chunk_size:
                    tokens_to_process = (
                        (current_total_tokens - processed_token_count) // current_chunk_size
                    ) * current_chunk_size
                    
                    end_idx = processed_token_count + tokens_to_process
                    
                    if end_idx > processed_token_count:
                        # Get new tokens to decode
                        new_speech_ids = all_speech_ids[processed_token_count:end_idx]
                        
                        # Decode to audio
                        audio_hat = await loop.run_in_executor(
                            None, decode_xcodec_tokens_sync, new_speech_ids
                        )
                        
                        # Convert to PCM bytes
                        pcm_bytes = convert_to_pcm16_bytes(audio_hat, fade_ms=10)
                        
                        if pcm_bytes:
                            yield pcm_bytes
                            first_chunk_yielded = True
                            print(f"Yielded chunk: {len(new_speech_ids)} tokens -> {len(pcm_bytes)} bytes")
                        
                        processed_token_count = end_idx
        
        # Process remaining tokens
        all_speech_ids = extract_speech_ids_from_text(accumulated_text)
        
        if len(all_speech_ids) > processed_token_count:
            remaining_ids = all_speech_ids[processed_token_count:]
            
            if remaining_ids:
                print(f"Processing final {len(remaining_ids)} tokens")
                audio_hat = await loop.run_in_executor(
                    None, decode_xcodec_tokens_sync, remaining_ids
                )
                
                pcm_bytes = convert_to_pcm16_bytes(audio_hat, fade_ms=10)
                
                if pcm_bytes:
                    yield pcm_bytes
        
        print(f"Generation complete. Total tokens: {len(all_speech_ids)}")
        
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio generation.
    
    Protocol:
    - Client sends JSON: {"input": "text", "continue": true/false, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                text = message.get("input", "")
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")
                
                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break
                
                if text:
                    # Send start message
                    await websocket.send_json({"type": "start", "segment_id": segment_id})
                    
                    # Stream audio chunks
                    async for audio_chunk in generate_audio_chunks(
                        text=text,
                        temperature=DEFAULT_TEMPERATURE,
                        top_p=DEFAULT_TOP_P,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY
                    ):
                        await websocket.send_bytes(audio_chunk)
                    
                    # Send end message
                    await websocket.send_json({"type": "end", "segment_id": segment_id})
                    
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")


@app.post("/v1/audio/speech/stream")
async def http_audio_stream(request: AudioRequest):
    """
    HTTP endpoint for streaming audio as raw PCM bytes.
    Compatible with streaming audio players.
    """
    print(f"Received HTTP streaming request for: '{request.text[:50]}...'")
    
    async def stream_pcm():
        async for chunk in generate_audio_chunks(
            text=request.text,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty
        ):
            yield chunk
    
    return StreamingResponse(
        stream_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS)
        }
    )


@app.get("/")
async def read_root():
    return {
        "message": "Llasa XCodec2 TTS Streaming API",
        "model": MODEL_NAME,
        "codec": XCODEC2_MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": xcodec_model is not None,
        "device": CODEC_DEVICE
    }


if __name__ == "__main__":
    print("Starting Llasa XCodec2 Streaming API...")
    print(f"Model: {MODEL_NAME}")
    print(f"Codec: {XCODEC2_MODEL_NAME}")
    print(f"Device: {CODEC_DEVICE}")
    uvicorn.run("streaming_api_server:app", host="0.0.0.0", port=8001, reload=False)