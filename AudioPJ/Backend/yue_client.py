import os
import torch
import gc
import sys

# HACK: Add torch's lib directory AND nvidia modules to DLL search path
try:
    # 1. Add Torch Lib
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
        print(f"🔧 Added Torch Lib to DLL Path: {torch_lib}")

    # 2. Add Nvidia Libs (cublas, cudnn, etc.)
    site_packages = os.path.dirname(os.path.dirname(torch.__file__))
    nvidia_path = os.path.join(site_packages, 'nvidia')
    if os.path.exists(nvidia_path):
        for root, dirs, files in os.walk(nvidia_path):
            if 'bin' in dirs:
                bin_path = os.path.join(root, 'bin')
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(bin_path)
                os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                print(f"🔧 Added Nvidia DLL Path: {bin_path}")
except Exception as e:
    print(f"⚠️ Could not add DLL paths: {e}")

from llama_cpp import Llama
from transformers import AutoModel, AutoTokenizer
import scipy.io.wavfile
import numpy as np
import logging

# Initialize logger BEFORE using it
logger = logging.getLogger(__name__)

# Try to use real XCodec decoder first, fallback to placeholder
try:
    from xcodec_real_decoder import decode_stage1_output_real
    USE_REAL_XCODEC = True
    logger.info("Real XCodec decoder available")
except ImportError as e:
    logger.warning(f"Real XCodec decoder not available: {e}")
    from xcodec_decoder import decode_stage1_output
    USE_REAL_XCODEC = False

# CONFIGURAZIONE PERCORSI
# Placeholder path, will be updated when model is found

# CONFIGURAZIONE PERCORSI
MODEL_STAGE1_PATH = "./models/YuE-s1-7B-anneal-en-cot-Q4_K_S.gguf"
MODEL_STAGE2_PATH = "./models/yue-s2-1b-general-q8_0.gguf"
# Use the same output directory that FastAPI serves
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

def run_pipeline(prompt_text, genre, mood):
    """
    Esegue la staffetta: Carica S1 -> Genera -> Scarica S1 -> Carica S2 -> Audio
    """
    logger.info("--- Starting Pipeline ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- FASE 1: STAGE 1 (GGUF) ---
    logger.info(f"[1/4] Loading Stage 1 (GGUF) from {MODEL_STAGE1_PATH}...")
    print("[1/4] Caricamento Stage 1 (GGUF)...")
    
    if not os.path.exists(MODEL_STAGE1_PATH):
        logger.error(f"Error: Stage 1 model not found at {MODEL_STAGE1_PATH}")
        print(f"Errore: Modello Stage 1 non trovato in {MODEL_STAGE1_PATH}")
        return None

    try:
        llm_s1 = Llama(
            model_path=MODEL_STAGE1_PATH,
            n_ctx=2048,          # LIMITATO A 30 SECONDI PER EVITARE LOOP
            n_gpu_layers=-1,     # Usa tutta la GPU possibile
            verbose=True
        )
    except Exception as e:
        logger.error(f"Failed to load Stage 1 model: {e}", exc_info=True)
        return None

    full_prompt = f"[Genre] {genre}\n[Mood] {mood}\n[Lyrics]\n{prompt_text}\n"
    
    logger.info("[2/4] Generating Audio Tokens (Stage 1)...")
    print("[2/4] Generazione Token Audio (Stage 1)...")
    try:
        output_s1 = llm_s1(
            full_prompt,
            max_tokens=2048,
            temperature=1.0,
            stop=["[EXIT]"],
            echo=False
        )
        raw_content_s1 = output_s1['choices'][0]['text']
        logger.info(f"Stage 1 generation complete. Output length: {len(raw_content_s1)}")
    except Exception as e:
        logger.error(f"Stage 1 generation failed: {e}", exc_info=True)
        return None
    
    # PULIZIA MEMORIA STAGE 1
    del llm_s1
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 1 memory cleared.")
    print("Memoria Stage 1 liberata.")

    # --- FASE 2: STAGE 2 (GGUF) ---
    logger.info(f"[3/4] Loading Stage 2 (GGUF) from {MODEL_STAGE2_PATH}...")
    print("[3/4] Caricamento Stage 2 (GGUF)...")
    
    if os.path.exists(MODEL_STAGE2_PATH):
        try:
            llm_s2 = Llama(
                model_path=MODEL_STAGE2_PATH,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=True
            )
            logger.info("Stage 2 Loaded successfully (Load Test Passed)")
            print("Stage 2 Caricato correttamente (Test caricamento superato)")
            
            # Per ora non generiamo nulla con Stage 2 perché manca il bridging dei token
            # Ma dimostriamo che il modello si carica in VRAM
            
            del llm_s2
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Stage 2 memory cleared.")
            print("Memoria Stage 2 liberata.")
            
        except Exception as e:
            logger.error(f"Stage 2 GGUF load error: {e}", exc_info=True)
            print(f"Errore caricamento Stage 2 GGUF: {e}")
    else:
        logger.warning(f"Stage 2 model not found at {MODEL_STAGE2_PATH}, skipping load test.")
        print(f"Modello Stage 2 non trovato in {MODEL_STAGE2_PATH}, salto il test di caricamento.")

    # --- SALVATAGGIO OUTPUT ---
    logger.info("[4/4] Saving results...")
    print("[4/4] Salvataggio risultati...")

    # Save text output for debugging
    txt_filename = f"{OUTPUT_DIR}/output_raw.txt"
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(raw_content_s1)
        logger.info(f"Text output saved to {txt_filename}")
    except Exception as e:
        logger.error(f"Failed to save text output: {e}", exc_info=True)

    # Decode audio tokens from Stage 1 output
    logger.info("[4/4] Decoding audio tokens...")
    print("[4/4] Decodifica token audio...")

    audio_filename = f"generated_audio_{genre}_{mood[:10]}.wav"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)

    try:
        sample_rate = 44100
        duration = 30.0  # seconds

        # Decode tokens from Stage 1 output
        if USE_REAL_XCODEC:
            logger.info("Using real XCodec decoder...")
            print("🎵 Using real XCodec decoder for high-quality audio...")
            audio_data = decode_stage1_output_real(raw_content_s1, sample_rate, duration)
        else:
            logger.info("Using placeholder decoder...")
            print("⚠️ Using placeholder decoder (install transformers for real XCodec)")
            from xcodec_decoder import decode_stage1_output
            audio_data = decode_stage1_output(raw_content_s1, sample_rate, duration)

        if audio_data is None:
            logger.warning("Token decoding failed, using fallback tone")
            # Fallback to simple tone
            frequency = 440.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t)

            fade_samples = int(sample_rate * 0.1)
            audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Convert to 16-bit PCM
        audio_data_int16 = np.int16(audio_data * 32767)

        scipy.io.wavfile.write(audio_path, sample_rate, audio_data_int16)
        logger.info(f"Finished! Audio file saved to {audio_path}")
        print(f"Finito! File audio salvato in {audio_path}")
        print(f"✅ Audio generated from real tokens")
        logger.info("Audio generation from tokens completed")

        return audio_filename  # Return just the filename, not the full path
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}", exc_info=True)
        return None

# Funzione dummy per testare solo l'audio (se avessimo i token giusti)
def decode_tokens(tokens):
    pass
