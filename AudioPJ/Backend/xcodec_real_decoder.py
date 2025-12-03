"""
Real XCodec Decoder for YuE
Uses the actual XCodec model from Hugging Face to decode audio tokens
"""
import logging
import numpy as np
import torch
from typing import List, Optional
import re

logger = logging.getLogger(__name__)

# Global codec model (loaded once)
_xcodec_model = None
_xcodec_processor = None


def load_xcodec_model():
    """Load the XCodec model from Hugging Face"""
    global _xcodec_model, _xcodec_processor

    if _xcodec_model is not None:
        return _xcodec_model, _xcodec_processor

    try:
        logger.info("Loading XCodec2 model from Hugging Face (with custom code)...")

        # XCodec2 requires loading the custom modeling code
        # We need to use the model's custom class directly
        import sys
        from pathlib import Path
        from huggingface_hub import hf_hub_download, snapshot_download

        # Download the model repository
        model_path = snapshot_download("HKUSTAudio/xcodec2")
        logger.info(f"Model downloaded to: {model_path}")

        # Add to Python path so we can import the custom code
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        # Import the custom XCodec2 model class
        from modeling_xcodec2 import XCodec2Model

        # Load the model
        _xcodec_model = XCodec2Model.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            _xcodec_model = _xcodec_model.cuda()

        _xcodec_model.eval()

        # XCodec2 uses 16kHz sampling rate
        _xcodec_processor = {"sampling_rate": 16000}

        logger.info("✅ XCodec2 model loaded successfully")
        return _xcodec_model, _xcodec_processor

    except Exception as e:
        logger.error(f"Failed to load XCodec2 model: {e}", exc_info=True)
        return None, None


def extract_audio_tokens(text: str) -> List[int]:
    """Extract xcodec tokens from Stage 1 text output"""
    pattern = r'<xcodec/0/(\d+)>'
    matches = re.findall(pattern, text)

    if not matches:
        logger.warning("No xcodec tokens found in output")
        return []

    tokens = [int(match) for match in matches]
    logger.info(f"Extracted {len(tokens)} audio tokens (range: {min(tokens)}-{max(tokens)})")

    return tokens


def decode_with_xcodec(tokens: List[int], sample_rate: int = 44100) -> Optional[np.ndarray]:
    """
    Decode audio tokens using the real XCodec model

    Args:
        tokens: List of audio token IDs from Stage 1
        sample_rate: Target sample rate

    Returns:
        Audio waveform as numpy array or None if failed
    """
    model, processor = load_xcodec_model()

    if model is None or processor is None:
        logger.error("XCodec model not available")
        return None

    try:
        # Convert tokens to tensor
        # XCodec2 expects tokens in shape (batch, 1, sequence_length)
        token_tensor = torch.tensor([tokens], dtype=torch.long).unsqueeze(1)

        if torch.cuda.is_available():
            token_tensor = token_tensor.cuda()

        logger.info(f"Decoding {len(tokens)} tokens with XCodec2...")
        logger.info(f"Token tensor shape: {token_tensor.shape}")

        # Decode using XCodec2's decode_code method
        with torch.no_grad():
            # XCodec2 API: decode_code(vq_code)
            audio_values = model.decode_code(token_tensor)

        # Convert to numpy
        if isinstance(audio_values, torch.Tensor):
            audio_array = audio_values.cpu().numpy()
        else:
            audio_array = np.array(audio_values)

        logger.info(f"Raw audio output shape: {audio_array.shape}")

        # Flatten if needed (XCodec2 outputs (batch, 1, samples))
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()

        # Resample if needed (XCodec2 outputs at 16kHz)
        model_sample_rate = processor.get('sampling_rate', 16000)
        if model_sample_rate != sample_rate:
            logger.info(f"Resampling from {model_sample_rate}Hz to {sample_rate}Hz...")
            from scipy import signal
            num_samples = int(len(audio_array) * sample_rate / model_sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        # Normalize
        if audio_array.max() > 0:
            audio_array = audio_array / np.abs(audio_array).max() * 0.9

        logger.info(f"✅ Successfully decoded {len(audio_array)} audio samples")
        return audio_array

    except Exception as e:
        logger.error(f"XCodec decoding failed: {e}", exc_info=True)
        return None


def decode_stage1_output_real(output_text: str, sample_rate: int = 44100, duration: float = 30.0) -> Optional[np.ndarray]:
    """
    Complete pipeline: extract tokens from text and decode with real XCodec

    Args:
        output_text: Raw text output from Stage 1 (contains xcodec tokens)
        sample_rate: Audio sample rate (default 44.1kHz)
        duration: Target duration in seconds (will pad/trim if needed)

    Returns:
        Audio array or None if failed
    """
    # Extract tokens
    tokens = extract_audio_tokens(output_text)

    if not tokens:
        logger.error("No audio tokens found in Stage 1 output")
        return None

    # Decode with real XCodec
    audio = decode_with_xcodec(tokens, sample_rate)

    if audio is None:
        logger.error("XCodec decoding failed")
        return None

    # Adjust duration
    target_samples = int(sample_rate * duration)
    current_samples = len(audio)

    if current_samples < target_samples:
        # Pad with silence
        logger.info(f"Padding audio from {current_samples/sample_rate:.2f}s to {duration}s")
        padding = np.zeros(target_samples - current_samples)
        audio = np.concatenate([audio, padding])
    elif current_samples > target_samples:
        # Trim
        logger.info(f"Trimming audio from {current_samples/sample_rate:.2f}s to {duration}s")
        audio = audio[:target_samples]

    logger.info(f"✅ Final audio: {len(audio)} samples, {duration:.2f} seconds")

    return audio
