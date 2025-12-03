"""
XCodec Token Decoder for YuE
Extracts and decodes audio tokens from Stage 1 output
"""
import re
import logging
import numpy as np
import torch
from typing import List, Optional

logger = logging.getLogger(__name__)


def extract_audio_tokens(text: str) -> List[int]:
    """
    Extract xcodec tokens from Stage 1 text output
    Format: <xcodec/0/NUMBER>
    Returns: List of token IDs
    """
    # Find all xcodec tokens
    pattern = r'<xcodec/0/(\d+)>'
    matches = re.findall(pattern, text)

    if not matches:
        logger.warning("No xcodec tokens found in output")
        return []

    tokens = [int(match) for match in matches]
    logger.info(f"Extracted {len(tokens)} audio tokens")
    logger.info(f"Token range: {min(tokens)} to {max(tokens)}")

    return tokens


def tokens_to_audio_simple(tokens: List[int], sample_rate: int = 44100, duration: float = 30.0) -> np.ndarray:
    """
    Simple token-to-audio conversion using token values as frequency modulation
    This is a placeholder that creates melodic audio based on the token sequence

    For real decoding, this would use the Stage 2 model or xcodec vocoder
    """
    if not tokens:
        logger.error("No tokens provided for audio generation")
        return None

    logger.info(f"Generating audio from {len(tokens)} tokens...")

    # Calculate samples
    total_samples = int(sample_rate * duration)

    # Map tokens to frequencies (normalize token range to musical scale)
    token_array = np.array(tokens, dtype=float)

    # Normalize tokens to 0-1 range
    token_min = token_array.min()
    token_max = token_array.max()
    if token_max > token_min:
        normalized = (token_array - token_min) / (token_max - token_min)
    else:
        normalized = np.zeros_like(token_array)

    # Map to musical frequency range (A3 to A5: 220-880 Hz)
    frequencies = 220 + normalized * 660

    # Create time array
    t = np.linspace(0, duration, total_samples, False)

    # Generate audio by interpolating between token frequencies
    audio = np.zeros(total_samples)

    # Each token controls a time segment
    samples_per_token = total_samples // len(tokens)

    for i, freq in enumerate(frequencies):
        start_idx = i * samples_per_token
        end_idx = min((i + 1) * samples_per_token, total_samples)

        if start_idx >= total_samples:
            break

        segment_t = t[start_idx:end_idx]
        # Generate sine wave for this segment with smooth transitions
        phase_offset = 0 if i == 0 else audio[start_idx - 1]
        audio[start_idx:end_idx] = np.sin(2 * np.pi * freq * segment_t + phase_offset)

        # Apply envelope to smooth transitions between tokens
        if i > 0:
            # Crossfade with previous segment
            fade_samples = min(100, samples_per_token // 4)
            fade_in = np.linspace(0, 1, fade_samples)
            audio[start_idx:start_idx + fade_samples] *= fade_in

    # Apply overall envelope (fade in/out)
    fade_duration = 0.1  # seconds
    fade_samples = int(sample_rate * fade_duration)

    # Fade in
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)

    # Fade out
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Add subtle harmonics for richness
    harmonic = np.sin(4 * np.pi * 440 * t) * 0.1  # Octave harmonic
    audio = audio * 0.9 + harmonic[:len(audio)] * 0.1

    # Normalize
    if audio.max() > 0:
        audio = audio / np.abs(audio).max() * 0.8

    logger.info(f"Generated audio: {len(audio)} samples, {duration:.2f} seconds")

    return audio


def decode_stage1_output(output_text: str, sample_rate: int = 44100, duration: float = 30.0) -> Optional[np.ndarray]:
    """
    Complete pipeline: extract tokens from text and generate audio

    Args:
        output_text: Raw text output from Stage 1 (contains xcodec tokens)
        sample_rate: Audio sample rate (default 44.1kHz)
        duration: Target duration in seconds

    Returns:
        Audio array or None if failed
    """
    # Extract tokens
    tokens = extract_audio_tokens(output_text)

    if not tokens:
        logger.error("No audio tokens found in Stage 1 output")
        return None

    # Log statistics
    logger.info(f"Token statistics:")
    logger.info(f"  Count: {len(tokens)}")
    logger.info(f"  Unique tokens: {len(set(tokens))}")
    logger.info(f"  Most common: {max(set(tokens), key=tokens.count)}")

    # Generate audio from tokens
    audio = tokens_to_audio_simple(tokens, sample_rate, duration)

    if audio is not None:
        logger.info("✅ Token-to-audio decoding successful")
    else:
        logger.error("❌ Token-to-audio decoding failed")

    return audio
