#!/usr/bin/env python3

import os
import time
import threading
import psutil
import numpy as np
import scipy.io.wavfile

import torch

from pynvml import *

from melo.api import TTS

# =========================================================
# CONFIGURAÇÕES
# =========================================================
MODEL_LANGUAGE = "EN"

MONITOR_INTERVAL = 0.1

monitoring_data = []

running = True

# =========================================================
# DIRETÓRIOS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TXT_DIR = os.path.join(BASE_DIR, "..", "txts")

SAIDAS_DIR = os.path.join(BASE_DIR, "..", "saidas")

LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

# =========================================================
# GARANTE PASTAS
# =========================================================
for folder in [TXT_DIR, SAIDAS_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# ESCOLHA DEVICE
# =========================================================
print("\nEscolha o dispositivo:")

print("1 - CPU")
print("2 - GPU CUDA")

device_option = input("\nDigite a opção: ").strip()

if device_option == "2" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\nUsando dispositivo: {device}")

# =========================================================
# NVML
# =========================================================
handle = None

if device == "cuda":

    try:

        nvmlInit()

        handle = nvmlDeviceGetHandleByIndex(0)

        print("NVML inicializado com sucesso")

    except Exception as e:

        print(f"Erro ao inicializar NVML: {e}")

# =========================================================
# CARREGAR TEXTO
# =========================================================
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

# =========================================================
# NOME NUMERADO
# =========================================================
def create_numbered_filename(prefix, ext, folder):

    i = 1

    while True:

        filename = f"{prefix}_{i:03d}.{ext}"

        full_path = os.path.join(folder, filename)

        if not os.path.exists(full_path):

            return full_path

        i += 1

# =========================================================
# MONITORAMENTO
# =========================================================
def monitor():

    global running

    process = psutil.Process()

    psutil.cpu_percent(interval=None)

    while running:

        timestamp = time.time()

        cpu_percent = psutil.cpu_percent()

        ram_used_mb = process.memory_info().rss / (1024 ** 2)

        gpu_usage = 0

        vram_used = 0

        gpu_temperature = 0

        gpu_power = 0

        if handle is not None:

            try:

                util = nvmlDeviceGetUtilizationRates(handle)

                mem = nvmlDeviceGetMemoryInfo(handle)

                temp = nvmlDeviceGetTemperature(
                    handle,
                    NVML_TEMPERATURE_GPU
                )

                power = nvmlDeviceGetPowerUsage(handle) / 1000

                gpu_usage = util.gpu

                vram_used = mem.used / (1024 ** 2)

                gpu_temperature = temp

                gpu_power = power

            except Exception as e:

                print(f"Erro NVML: {e}")

        monitoring_data.append([
            timestamp,
            cpu_percent,
            ram_used_mb,
            gpu_usage,
            vram_used,
            gpu_temperature,
            gpu_power
        ])

        time.sleep(MONITOR_INTERVAL)

# =========================================================
# INFERÊNCIA
# =========================================================
def run_inference(wav_file, text):

    global running

    print("\nCarregando modelo MeloTTS...")

    speed = 1.0

    model = TTS(
        language=MODEL_LANGUAGE,
        device=device
    )

    print("Modelo carregado")

    speaker_ids = model.hps.data.spk2id

    print("\nSpeakers disponíveis:\n")

    speakers = list(speaker_ids.keys())

    for i, speaker in enumerate(speakers, 1):

        print(f"{i} - {speaker}")

    speaker_choice = int(
        input("\nEscolha o speaker: ")
    )

    speaker_name = speakers[speaker_choice - 1]

    speaker_id = speaker_ids[speaker_name]

    print(f"\nSpeaker escolhido: {speaker_name}")

    print("\nGerando áudio...")

    start = time.time()

    model.tts_to_file(
        text=text,
        speaker_id=speaker_id,
        output_path=wav_file,
        speed=speed
    )

    end = time.time()

    running = False

    print(f"\nÁudio salvo em: {wav_file}")

    return end - start

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("\nIniciando MeloTTS Benchmark...\n")

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

    # =====================================================
    # THREAD MONITOR
    # =====================================================
    monitor_thread = threading.Thread(
        target=monitor
    )

    monitor_thread.start()

    # =====================================================
    # INFERÊNCIA
    # =====================================================
    duration = run_inference(
        wav_file,
        texto_carregado
    )

    monitor_thread.join()

    print(
        f"\nTempo total de inferência: "
        f"{duration:.3f} segundos"
    )

    # =====================================================
    # GERA CSV
    # =====================================================
    with open(log_file, "w") as f:

        f.write(
            "timestamp,"
            "cpu_percent,"
            "ram_used_mb,"
            "gpu_percent,"
            "vram_used_mb,"
            "gpu_temp_c,"
            "gpu_power_w,"
            "inference_time\n"
        )

        for row in monitoring_data:

            f.write(
                f"{row[0]:.3f},"
                f"{row[1]:.2f},"
                f"{row[2]:.2f},"
                f"{row[3]:.2f},"
                f"{row[4]:.2f},"
                f"{row[5]:.2f},"
                f"{row[6]:.2f},"
                f"{duration:.3f}\n"
            )

    print(f"\nLog salvo em: {log_file}")

    # =====================================================
    # FINALIZA NVML
    # =====================================================
    if handle is not None:

        nvmlShutdown()

    print("\nExecução finalizada.\n")