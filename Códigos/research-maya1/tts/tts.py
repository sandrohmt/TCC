#!/usr/bin/env python3

import os
import time
import threading
import psutil
import numpy as np
import scipy.io.wavfile as wavfile
from pynvml import *

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

# ============================================
# CONFIGURAÇÕES
# ============================================
MODEL_ID = "maya-research/maya1"
SNAC_ID = "hubertsiuzdak/snac_24khz"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TXT_FOLDER = os.path.join(BASE_DIR, "..", "txts")
OUTPUT_AUDIO_FOLDER = os.path.join(BASE_DIR, "..", "saidas")
OUTPUT_LOG_FOLDER = os.path.join(BASE_DIR, "..", "logs")

CHUNK_MAX_CHARS = 400
CROSSFADE_MS = 80

MONITOR_INTERVAL = 0.1

monitoring_data = []
running = True

# ============================================
# TOKENS SNAC
# ============================================
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266

SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937

SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

# ============================================
# DEVICE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\nUsando dispositivo: {device}")

# ============================================
# NVML
# ============================================
handle = None

if device == "cuda":

    try:

        nvmlInit()

        handle = nvmlDeviceGetHandleByIndex(0)

        print("NVML inicializado com sucesso")

    except Exception as e:

        print(f"Erro ao inicializar NVML: {e}")

# ============================================
# UTIL: arquivos numerados
# ============================================
def create_numbered_filename(folder, prefix, ext):

    os.makedirs(folder, exist_ok=True)

    i = 1

    while True:

        filename = os.path.join(
            folder,
            f"{prefix}_{i:03d}.{ext}"
        )

        if not os.path.exists(filename):

            return filename

        i += 1

# ============================================
# MONITORAMENTO
# ============================================
def monitor():

    global running

    process = psutil.Process()

    psutil.cpu_percent(interval=None)

    while running:

        cpu = psutil.cpu_percent()

        memory = process.memory_info().rss / (1024 ** 2)

        gpu_usage = 0
        vram_used = 0

        if handle is not None:

            try:

                util = nvmlDeviceGetUtilizationRates(handle)

                mem = nvmlDeviceGetMemoryInfo(handle)

                gpu_usage = util.gpu

                vram_used = mem.used / (1024 ** 2)

            except Exception as e:

                print(f"Erro NVML: {e}")

        monitoring_data.append((
            time.time(),
            cpu,
            memory,
            gpu_usage,
            vram_used
        ))

        time.sleep(MONITOR_INTERVAL)

# ============================================
# CHUNKING DO TEXTO
# ============================================
def chunk_text(text, max_chars=CHUNK_MAX_CHARS):

    import re

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

# ============================================
# PROMPT MAYA1
# ============================================
def build_prompt(
    tokenizer,
    text,
    description=(
        "Dark villain character, Female voice in their 40s "
        "with a British accent. low pitch, gravelly timbre, "
        "slow pacing, angry tone at high intensity."
    )
):

    soh = tokenizer.decode([SOH_ID])

    eoh = tokenizer.decode([EOH_ID])

    soa = tokenizer.decode([SOA_ID])

    sos = tokenizer.decode([CODE_START_TOKEN_ID])

    eot = tokenizer.decode([TEXT_EOT_ID])

    bos = tokenizer.bos_token

    formatted = f'<description="{description}"> {text}'

    return soh + bos + formatted + eot + eoh + soa + sos

# ============================================
# EXTRAÇÃO SNAC
# ============================================
def extract_snac_codes(token_ids):

    try:

        eos_idx = token_ids.index(CODE_END_TOKEN_ID)

    except ValueError:

        eos_idx = len(token_ids)

    return [
        tid for tid in token_ids[:eos_idx]
        if SNAC_MIN_ID <= tid <= SNAC_MAX_ID
    ]

# ============================================
# UNPACK SNAC
# ============================================
def unpack_snac(snac_tokens):

    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:

        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // 7

    snac_tokens = snac_tokens[:frames * 7]

    if frames == 0:

        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):

        s = snac_tokens[i * 7:(i + 1) * 7]

        l1.append((s[0] - CODE_TOKEN_OFFSET) % 4096)

        l2.extend([
            (s[1] - CODE_TOKEN_OFFSET) % 4096,
            (s[4] - CODE_TOKEN_OFFSET) % 4096
        ])

        l3.extend([
            (s[2] - CODE_TOKEN_OFFSET) % 4096,
            (s[3] - CODE_TOKEN_OFFSET) % 4096,
            (s[5] - CODE_TOKEN_OFFSET) % 4096,
            (s[6] - CODE_TOKEN_OFFSET) % 4096
        ])

    return [l1, l2, l3]

# ============================================
# DECODIFICAÇÃO SNAC
# ============================================
def decode_snac(snac_model, levels, device):

    with torch.inference_mode():

        codes = [
            torch.tensor(level, device=device).unsqueeze(0)
            for level in levels
        ]

        z_q = snac_model.quantizer.from_codes(codes)

        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

    # remove warmup
    if len(audio) > 2048:

        audio = audio[2048:]

    return audio

# ============================================
# CROSSFADE
# ============================================
def crossfade(audio1, audio2, sr=24000, ms=CROSSFADE_MS):

    fade = int(sr * (ms / 1000))

    if (
        fade == 0 or
        len(audio1) < fade or
        len(audio2) < fade
    ):

        return np.concatenate([audio1, audio2])

    fadein = np.linspace(0, 1, fade)

    fadeout = 1 - fadein

    mixed = (
        audio1[-fade:] * fadeout +
        audio2[:fade] * fadein
    )

    return np.concatenate([
        audio1[:-fade],
        mixed,
        audio2[fade:]
    ])

# ============================================
# GERAÇÃO DE CHUNK
# ============================================
def generate_chunk(
    text,
    tokenizer,
    model,
    snac_model,
    device
):

    prompt = build_prompt(tokenizer, text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    try:

        with torch.inference_mode():

            out = model.generate(
                **inputs,
                max_new_tokens=1792,
                temperature=0.35,
                top_p=0.9,
                repetition_penalty=1.12,
                eos_token_id=CODE_END_TOKEN_ID,
                do_sample=True,
            )

    except Exception as e:

        print(f"Erro ao gerar chunk: {e}")

        return np.zeros(0, dtype=np.float32)

    ids = out[0, inputs['input_ids'].shape[1]:].tolist()

    snac_tokens = extract_snac_codes(ids)

    if len(snac_tokens) == 0:

        print("Nenhum token SNAC encontrado.")

        return np.zeros(0, dtype=np.float32)

    levels = unpack_snac(snac_tokens)

    audio = decode_snac(
        snac_model,
        levels,
        device
    )

    if device == "cuda":

        torch.cuda.empty_cache()

    return audio

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    txts = [
        f for f in os.listdir(TXT_FOLDER)
        if f.endswith(".txt")
    ]

    if not txts:

        print("Nenhum .txt encontrado.")

        exit()

    print("\n=== ARQUIVOS DISPONÍVEIS ===\n")

    for i, f in enumerate(txts, 1):

        print(f"{i}. {f}")

    choice = int(input("\nEscolha o arquivo: "))

    txt_path = os.path.join(
        TXT_FOLDER,
        txts[choice - 1]
    )

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        full_text = f.read()

    print("\nDividindo texto em partes...")

    chunks = chunk_text(full_text)

    print(f"{len(chunks)} partes geradas.")

    # ============================================
    # SAÍDAS
    # ============================================
    wav_out = create_numbered_filename(
        OUTPUT_AUDIO_FOLDER,
        "saida",
        "wav"
    )

    log_out = create_numbered_filename(
        OUTPUT_LOG_FOLDER,
        "monitor_maya",
        "csv"
    )

    # ============================================
    # MONITOR
    # ============================================
    monitor_thread = threading.Thread(
        target=monitor
    )

    monitor_thread.start()

    # ============================================
    # MODELOS
    # ============================================
    print("\nCarregando Maya1...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    print("Carregando SNAC...")

    snac_model = SNAC.from_pretrained(
        SNAC_ID
    ).eval()

    snac_model = snac_model.to(device)

    # ============================================
    # GERAÇÃO
    # ============================================
    print("\nGerando áudio...")

    full_audio = np.zeros(
        0,
        dtype=np.float32
    )

    start_total = time.time()

    for idx, c in enumerate(chunks, 1):

        print(
            f"\n> Parte {idx}/{len(chunks)} "
            f"({len(c)} chars)"
        )

        audio = generate_chunk(
            c,
            tokenizer,
            model,
            snac_model,
            device
        )

        if len(audio) > 0:

            full_audio = crossfade(
                full_audio,
                audio
            )

    end_total = time.time()

    running = False

    monitor_thread.join()

    # ============================================
    # NORMALIZA
    # ============================================
    if len(full_audio) > 0:

        full_audio = (
            full_audio /
            np.max(np.abs(full_audio))
        )

    # ============================================
    # SALVA WAV
    # ============================================
    wavfile.write(
        wav_out,
        24000,
        (full_audio * 32767).astype(np.int16)
    )

    print(f"\nÁudio salvo em: {wav_out}")

    # ============================================
    # CSV
    # ============================================
    with open(log_out, "w") as f:

        f.write(
            "timestamp,"
            "cpu_percent,"
            "memory_mb,"
            "gpu_percent,"
            "vram_mb,"
            "total_time_sec\n"
        )

        for row in monitoring_data:

            f.write(
                f"{row[0]:.3f},"
                f"{row[1]:.2f},"
                f"{row[2]:.2f},"
                f"{row[3]:.2f},"
                f"{row[4]:.2f},"
                f"{end_total-start_total:.2f}\n"
            )

    print(f"Log salvo em: {log_out}")

    print(
        f"Tempo total: "
        f"{end_total-start_total:.2f}s"
    )

    # ============================================
    # FINALIZA NVML
    # ============================================
    if handle is not None:

        nvmlShutdown()