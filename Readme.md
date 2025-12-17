pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps

pip install omegaconf torchcodec

pip install "fastapi[standard]"

pip install --no-deps trl==0.15.2

pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

pip install frozendict torch transformers torchaudio

vllm serve unsloth/Llasa-1B --dtype bfloat16 --max-model-len 8192 --enforce-eager --disable-custom-all-reduce --max-num-seqs 1 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --trust-remote-code