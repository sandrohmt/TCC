#!/usr/bin/env python3

# ============================================================
# SETUP:
#   pip install torch torchaudio transformers==4.47.0
#   pip install xcodec2
#   pip install torchao==0.6.1
#   pip install torchtune==0.3.1 --no-deps
#   pip install accelerate psutil pynvml scipy numpy soundfile
#
# Versão CPU — otimizada para r8id.4xlarge (128GB RAM)
# ============================================================

import os
import sys
import time
import threading
import re
import numpy as np
import scipy.io.wavfile as wavfile
import psutil

# CPU não tem NVML — desativa silenciosamente
NVML_AVAILABLE = False

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model

# ============================================================
# CONFIGURAÇÕES
# ============================================================
MODEL_ID    = "HKUSTAudio/Llasa-8B"
CODEC_ID    = "HKUSTAudio/xcodec2"
SAMPLE_RATE = 16000

BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_FOLDER          = os.path.join(BASE_DIR, "txts")
OUTPUT_AUDIO_FOLDER = os.path.join(BASE_DIR, "saidas")
OUTPUT_LOG_FOLDER   = os.path.join(BASE_DIR, "logs")

CHUNK_MAX_CHARS  = 400
CROSSFADE_MS     = 80
MONITOR_INTERVAL = 0.1

monitoring_data = []
running = True

# ============================================================
# UTIL: arquivos numerados
# ============================================================
def create_numbered_filename(folder, prefix, ext):
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        filename = os.path.join(folder, f"{prefix}_{i:03d}.{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

# ============================================================
# MONITORAMENTO (CPU + RAM apenas)
# ============================================================
def monitor():
    global running
    process = psutil.Process()
    psutil.cpu_percent(interval=None)

    while running:
        cpu    = psutil.cpu_percent()
        memory = process.memory_info().rss / (1024 ** 2)

        monitoring_data.append((
            time.time(),
            cpu,
            memory,
            0,  # gpu_percent — N/A em CPU
            0,  # vram_mb — N/A em CPU
        ))
        time.sleep(MONITOR_INTERVAL)

# ============================================================
# CHUNKING DO TEXTO
# ============================================================
def chunk_text(text, max_chars=CHUNK_MAX_CHARS):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ============================================================
# CROSSFADE
# ============================================================
def crossfade(audio1, audio2, sr=SAMPLE_RATE, ms=CROSSFADE_MS):
    fade = int(sr * (ms / 1000))
    if fade == 0 or len(audio1) < fade or len(audio2) < fade:
        return np.concatenate([audio1, audio2])
    fadein  = np.linspace(0, 1, fade)
    fadeout = 1 - fadein
    mixed   = audio1[-fade:] * fadeout + audio2[:fade] * fadein
    return np.concatenate([audio1[:-fade], mixed, audio2[fade:]])

# ============================================================
# HELPERS DE TOKEN
# ============================================================
def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            speech_ids.append(int(token_str[4:-2]))
        else:
            print(f"Token inesperado ignorado: {token_str}")
    return speech_ids

# ============================================================
# GERAÇÃO DE CHUNK
# ============================================================
def generate_chunk(text, model, tokenizer, codec_model):
    try:
        formatted_text = (
            f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        )

        chat = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>",
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        )  # sem .to(device) — fica em CPU

        speech_end_id = tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )

        tokenizer.pad_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=2048,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=1,
            )

        generated_ids = outputs[0][input_ids.shape[1]:-1]

        speech_tokens_str = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        speech_ids = extract_speech_ids(speech_tokens_str)

        if len(speech_ids) == 0:
            print("Nenhum token de fala extraído.")
            return np.zeros(0, dtype=np.float32)

        speech_tensor = (
            torch.tensor(speech_ids)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # sem .to(device) — fica em CPU

        gen_wav = codec_model.decode_code(speech_tensor)

        audio = gen_wav[0, 0, :].numpy().astype(np.float32)
        return audio

    except Exception as e:
        print(f"Erro na geração do chunk: {e}")
        return np.zeros(0, dtype=np.float32)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---- lista txts ----
    txts = [f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")]

    if not txts:
        print("Nenhum .txt encontrado em:", TXT_FOLDER)
        sys.exit()

    print("\n=== ARQUIVOS DISPONÍVEIS ===\n")
    for i, f in enumerate(txts, 1):
        print(f"{i}. {f}")

    choice   = int(input("\nEscolha o arquivo: "))
    txt_path = os.path.join(TXT_FOLDER, txts[choice - 1])

    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    print("\nDividindo texto em partes...")
    chunks = chunk_text(full_text)
    print(f"{len(chunks)} partes geradas.")

    # ---- saídas ----
    wav_out = create_numbered_filename(OUTPUT_AUDIO_FOLDER, "saida_llasa_cpu", "wav")
    log_out = create_numbered_filename(OUTPUT_LOG_FOLDER,   "monitor_llasa_cpu", "csv")

    # ---- monitor ----
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # ---- carrega modelos em CPU ----
    print("\nUsando dispositivo: cpu")

    print(f"Carregando tokenizer e modelo: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 para CPU
        device_map="cpu",
    )
    model.eval()

    print(f"Carregando codec: {CODEC_ID}")
    codec_model = XCodec2Model.from_pretrained(CODEC_ID)
    codec_model.eval()  # sem .to(device)

    print("Modelos carregados.")
    print("AVISO: inferência em CPU é muito mais lenta que GPU.")

    # ---- geração ----
    print("\nGerando áudio...")
    full_audio  = np.zeros(0, dtype=np.float32)
    start_total = time.time()

    for idx, c in enumerate(chunks, 1):
        chunk_start = time.time()
        print(f"\n> Parte {idx}/{len(chunks)} ({len(c)} chars)")
        audio = generate_chunk(c, model, tokenizer, codec_model)
        chunk_time = time.time() - chunk_start
        print(f"  Tempo do chunk: {chunk_time:.1f}s")
        if len(audio) > 0:
            full_audio = crossfade(full_audio, audio, sr=SAMPLE_RATE)

    end_total = time.time()
    running   = False
    monitor_thread.join()

    # ---- normaliza ----
    if len(full_audio) > 0 and np.max(np.abs(full_audio)) > 0:
        full_audio = full_audio / np.max(np.abs(full_audio))

    # ---- salva WAV ----
    wavfile.write(
        wav_out,
        SAMPLE_RATE,
        (full_audio * 32767).astype(np.int16),
    )
    print(f"\nÁudio salvo em: {wav_out}")

    # ---- CSV ----
    with open(log_out, "w") as f:
        f.write(
            "timestamp,cpu_percent,memory_mb,"
            "gpu_percent,vram_mb,total_time_sec\n"
        )
        for row in monitoring_data:
            f.write(
                f"{row[0]:.3f},{row[1]:.2f},{row[2]:.2f},"
                f"{row[3]:.2f},{row[4]:.2f},"
                f"{end_total - start_total:.2f}\n"
            )

    print(f"Log salvo em: {log_out}")
    print(f"Tempo total: {end_total - start_total:.2f}s")
