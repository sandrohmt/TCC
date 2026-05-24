#!/usr/bin/env python3

import time
import threading
import psutil
import numpy as np
import scipy.io.wavfile
import os

import torch
from transformers import VitsModel, AutoTokenizer

from pynvml import *

# ============================================
# CONFIGURAÇÕES
# ============================================
MODEL_ID = "facebook/mms-tts-por"

MONITOR_INTERVAL = 0.1

monitoring_data = []
running = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TXT_DIR = os.path.join(BASE_DIR, "..", "txts")
SAIDAS_DIR = os.path.join(BASE_DIR, "..", "saidas")
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

# ============================================
# DEVICE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32

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
# GARANTE PASTAS
# ============================================
for folder in [TXT_DIR, SAIDAS_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ============================================
# CARREGAR TEXTO
# ============================================
def load_text_from_file():

    arquivos = [
        f for f in os.listdir(TXT_DIR)
        if f.endswith(".txt")
    ]

    if not arquivos:
        raise Exception(
            f"Nenhum arquivo .txt encontrado em '{TXT_DIR}'"
        )

    print("\nArquivos disponíveis:\n")

    for i, arq in enumerate(arquivos, 1):
        print(f"{i} - {arq}")

    escolha = int(input("\nDigite o número do arquivo: "))

    filename = arquivos[escolha - 1]

    caminho = os.path.join(TXT_DIR, filename)

    print(f"\nUsando texto de: {caminho}")

    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()

# ============================================
# NOME NUMERADO
# ============================================
def create_numbered_filename(prefix, ext, folder):

    i = 1

    while True:

        filename = f"{prefix}_{i:03d}.{ext}"

        full_path = os.path.join(folder, filename)

        if not os.path.exists(full_path):
            return full_path

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
# INFERÊNCIA
# ============================================
def run_inference(wav_file, text):

    global running

    print("\nCarregando modelo...")

    model = VitsModel.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Gerando áudio...")

    inputs = tokenizer(text, return_tensors="pt")

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    start = time.time()

    with torch.no_grad():

        output = model(**inputs).waveform

    end = time.time()

    running = False

    # ============================================
    # PROCESSA ÁUDIO
    # ============================================
    audio = output.squeeze().cpu().numpy()

    audio = audio / np.max(np.abs(audio))

    audio_int16 = (audio * 32767).astype(np.int16)

    scipy.io.wavfile.write(
        wav_file,
        rate=model.config.sampling_rate,
        data=audio_int16
    )

    print(f"\nÁudio salvo em: {wav_file}")

    return end - start

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    print("Iniciando...")

    texto_carregado = load_text_from_file()

    wav_file = create_numbered_filename(
        "saida",
        "wav",
        SAIDAS_DIR
    )

    log_file = create_numbered_filename(
        "monitor_log",
        "csv",
        LOGS_DIR
    )

    # ============================================
    # MONITOR
    # ============================================
    monitor_thread = threading.Thread(target=monitor)

    monitor_thread.start()

    # ============================================
    # INFERÊNCIA
    # ============================================
    duration = run_inference(
        wav_file,
        texto_carregado
    )

    monitor_thread.join()

    print(f"\nTempo total de inferência: {duration:.3f} segundos")

    # ============================================
    # CSV
    # ============================================
    with open(log_file, "w") as f:

        f.write(
            "timestamp,"
            "cpu_percent,"
            "memory_mb,"
            "gpu_percent,"
            "vram_mb,"
            "inference_time\n"
        )

        for row in monitoring_data:

            f.write(
                f"{row[0]:.3f},"
                f"{row[1]:.2f},"
                f"{row[2]:.2f},"
                f"{row[3]:.2f},"
                f"{row[4]:.2f},"
                f"{duration:.2f}\n"
            )

    print(f"Log salvo em: {log_file}\n")