from dataclasses import dataclass
import logging
import os
from typing import Optional
import wave

import ffmpeg
from pydub.utils import mediainfo
import webrtcvad

# Default audio format constants for speech recognition
# 16kHz: Standard sample rate for speech recognition (ASR)
# 1 channel: Mono audio is required by most VAD/ASR engines
# 2 bytes: 16-bit PCM is the most common bit depth for speech
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 2 bytes = 16 bits

# Default chunking parameters for VAD-based splitting
DEFAULT_MIN_CHUNK_DURATION = 300000  # 5 minutes
DEFAULT_MAX_CHUNK_DURATION = 600000  # 10 minutes
DEFAULT_VAD_AGGRESSIVENESS = 2
DEFAULT_MAX_WAIT_FOR_SILENCE = 120000  # 2 minutes

logger = logging.getLogger(__name__)


@dataclass
class ChunkMeta:
    """
    Metadata for a speech chunk generated from audio splitting.

    Attributes
    ----------
    index : int
        The sequential index of the chunk in the original audio.
    wav_path : str
        Absolute path to the chunk wav file.
    mp3_path : Optional[str]
        Absolute path to the chunk mp3 file (set after conversion).
    start_time_in_ms : int
        Start time of the chunk in the original audio, in milliseconds.
    end_time_in_ms : int
        End time of the chunk in the original audio, in milliseconds.
    duration_in_ms : int
        Duration of the chunk, in milliseconds.
    wav_size : Optional[int]
        Size of the wav file in bytes.
    mp3_size : Optional[int]
        Size of the mp3 file in bytes.
    cached : bool
        Whether the chunk was loaded from cache.
    transcript : Optional[str]
        Transcription result for this chunk.
    """
    index: int
    wav_path: str
    mp3_path: Optional[str] = None
    start_time_in_ms: int = 0
    end_time_in_ms: int = 0
    duration_in_ms: int = 0
    wav_size: Optional[int] = None
    mp3_size: Optional[int] = None
    cached: bool = False
    transcript: Optional[str] = None


def is_valid_wav(
    input_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH
) -> bool:
    # Check if the input wav file matches the given format
    # Returns True if matches, False otherwise
    # sample_rate: target sample rate (default 16kHz for ASR)
    # channels: target channel count (default mono)
    # sample_width: target bit depth in bytes (default 2 for 16bit)
    try:
        with wave.open(input_path, 'rb') as wf:
            if (
                wf.getframerate() == sample_rate and
                wf.getnchannels() == channels and
                wf.getsampwidth() == sample_width
            ):
                return True
    except Exception as e:
        # If the file cannot be opened or is not a valid wav, return False
        # This may happen if the file is corrupted, not a wav, or unreadable
        # Exception info: {}
        pass
    # Return False if format does not match or exception occurs
    return False


def convert_audio_ffmpeg(
    input_path: str,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    sample_fmt: str = 's16'
) -> None:
    """
    Convert audio file to specified format using ffmpeg.

    Args:
        input_path (str): Source audio file path.
        output_path (str): Target audio file path. Format is inferred from
            extension.
        sample_rate (int, optional): Target sample rate. Default is 16000.
        channels (int, optional): Target channel count. Default is 1 (mono).
        sample_width (int, optional): Bit depth in bytes. Default is 2 (16bit).
        sample_fmt (str, optional): PCM sample format for ffmpeg. Default is
            's16'. Common values: 's16' (16bit), 's32' (32bit), 'flt' (float32),
            etc.

    Returns:
        None
    """
    logger.info(
        f"Converting audio: {input_path} -> {output_path}, "
        f"sample_rate={sample_rate}, channels={channels}, "
        f"sample_fmt={sample_fmt}"
    )
    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                ar=sample_rate,
                ac=channels,
                sample_fmt=sample_fmt
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"Audio conversion finished: {output_path}")
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"ffmpeg error during conversion: {err_msg}")
        raise


def split_speech_chunks(
    input_path: str,
    output_dir: str,
    min_chunk_duration: int = DEFAULT_MIN_CHUNK_DURATION,
    max_chunk_duration: int = DEFAULT_MAX_CHUNK_DURATION,
    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS,
    max_wait_for_silence: int = DEFAULT_MAX_WAIT_FOR_SILENCE
) -> list:
    """
    Split audio file into chunks based on silence using VAD and min/max duration.

    Parameters
    ----------
    input_path : str
        Path to the input audio file. Any format supported by ffmpeg.
        Will be converted to standard wav if needed.
    output_dir : str
        Directory to save the resulting wav chunks.
    min_chunk_duration : int, optional
        Minimum duration (ms) for each chunk. Default is 300000 ms (5 min).
    max_chunk_duration : int, optional
        Maximum duration (ms) for each chunk. Default is 600000 ms (10 min).
    vad_aggressiveness : int, optional
        VAD aggressiveness mode. 0=least, 3=most aggressive. Default is 2.
    max_wait_for_silence : int, optional
        Absolute max overrun after max_chunk_duration (ms). Default 120000 ms.

    Returns
    -------
    list of ChunkMeta
        List of chunk metadata objects, each describing a chunk file and its
        position in the original audio.

    Notes
    -----
    - Output files are always 16kHz, 16bit, mono wav.
    - If input is not a valid wav with required parameters, it will be
      converted automatically.
    - Uses webrtcvad for speech/silence detection.
    - Chunks will not be shorter than min_chunk_duration, nor longer than
      max_chunk_duration + max_wait_for_silence.
    - Forced splits will prefer silence points for natural boundaries.
    """
    logger.info(
        f"[split_speech_chunks] Start splitting: {input_path} -> "
        f"{output_dir}"
    )
    logger.info(
        f"[split_speech_chunks] Parameters: min_chunk_duration="
        f"{min_chunk_duration}, max_chunk_duration={max_chunk_duration}, "
        f"vad_aggressiveness={vad_aggressiveness}, "
        f"max_wait_for_silence={max_wait_for_silence}"
    )

    # Ensure input is valid wav
    if not is_valid_wav(input_path):
        temp_wav = os.path.join(output_dir, 'vad_input.wav')
        logger.info(
            f"[split_speech_chunks] Input is not valid wav, converting to "
            f"{temp_wav}"
        )
        convert_audio_ffmpeg(input_path, temp_wav)
        wav_path = temp_wav
    else:
        wav_path = input_path
        logger.info(
            f"[split_speech_chunks] Input is valid wav: {wav_path}"
        )

    # Read frames from wav file
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        frame_bytes = int(sample_rate * 30 / 1000 * 2)
        frames = []
        while True:
            pcm = wf.readframes(frame_bytes // 2)
            if len(pcm) < frame_bytes:
                break
            frames.append(pcm)
    logger.info(
        f"[split_speech_chunks] Total frames read: {len(frames)}"
    )

    # VAD detection
    vad = webrtcvad.Vad(vad_aggressiveness)
    speech_flags = [vad.is_speech(f, 16000) for f in frames]
    logger.info(
        f"[split_speech_chunks] VAD detection complete."
    )

    # Chunking logic
    current_chunk = []
    current_chunk_duration = 0
    current_chunk_start_time = 0
    waiting_for_silence = False
    overrun_duration = 0
    frame_duration = 30  # ms
    chunk_meta_tuples = []

    for i, (flag, frame) in enumerate(zip(speech_flags, frames)):
        # Always add frame to current chunk
        current_chunk.append(frame)
        current_chunk_duration += frame_duration

        # If in waiting mode, accumulate overrun duration
        if flag and waiting_for_silence:
            overrun_duration += frame_duration

        # Silence point logic
        if not flag:
            if (
                current_chunk_duration >= min_chunk_duration and
                waiting_for_silence
            ):
                chunk_end_time = (
                    current_chunk_start_time + current_chunk_duration
                )
                chunk_meta_tuples.append(
                    (current_chunk, current_chunk_start_time, chunk_end_time)
                )
                logger.info(
                    f"[split_speech_chunks] Chunk forced by max duration and "
                    f"silence: {len(current_chunk)} frames, duration="
                    f"{current_chunk_duration}ms, overrun={overrun_duration}ms, "
                    f"frame_idx={i}"
                )
                current_chunk = []
                current_chunk_start_time = chunk_end_time
                current_chunk_duration = 0
                waiting_for_silence = False
                overrun_duration = 0
                continue
            if (
                current_chunk_duration >= min_chunk_duration and
                not waiting_for_silence
            ):
                chunk_end_time = (
                    current_chunk_start_time + current_chunk_duration
                )
                chunk_meta_tuples.append(
                    (current_chunk, current_chunk_start_time, chunk_end_time)
                )
                logger.info(
                    f"[split_speech_chunks] Chunk ended by silence: "
                    f"{len(current_chunk)} frames, duration="
                    f"{current_chunk_duration}ms, frame_idx={i}"
                )
                current_chunk = []
                current_chunk_start_time = chunk_end_time
                current_chunk_duration = 0
                continue

        if (
            current_chunk_duration >= max_chunk_duration and
            not waiting_for_silence
        ):
            waiting_for_silence = True
            overrun_duration = 0
            logger.info(
                f"[split_speech_chunks] Enter waiting for silence mode at "
                f"frame {i}, current_chunk_duration="
                f"{current_chunk_duration}ms"
            )

        if (
            waiting_for_silence and
            overrun_duration >= max_wait_for_silence
        ):
            chunk_end_time = (
                current_chunk_start_time + current_chunk_duration
            )
            chunk_meta_tuples.append(
                (current_chunk, current_chunk_start_time, chunk_end_time)
            )
            logger.info(
                f"[split_speech_chunks] Chunk forced by absolute max overrun: "
                f"{len(current_chunk)} frames, duration="
                f"{current_chunk_duration}ms, overrun={overrun_duration}ms, "
                f"frame_idx={i}"
            )
            current_chunk = []
            current_chunk_start_time = chunk_end_time
            current_chunk_duration = 0
            waiting_for_silence = False
            overrun_duration = 0

    # Final chunk
    if current_chunk:
        chunk_end_time = current_chunk_start_time + current_chunk_duration
        chunk_meta_tuples.append(
            (current_chunk, current_chunk_start_time, chunk_end_time)
        )
        logger.info(
            f"[split_speech_chunks] Final chunk: {len(current_chunk)} frames, "
            f"duration={current_chunk_duration}ms"
        )

    # Write chunk files and build metadata
    chunk_meta_objs = []
    for idx, (chunk_frames, start_time, end_time) in enumerate(chunk_meta_tuples):
        chunk_path = os.path.abspath(
            os.path.join(output_dir, f'chunk_{idx}.wav')
        )
        with wave.open(chunk_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b''.join(chunk_frames))
        duration = end_time - start_time
        meta = ChunkMeta(
            index=idx,
            wav_path=chunk_path,
            start_time_in_ms=start_time,
            end_time_in_ms=end_time,
            duration_in_ms=duration
        )
        chunk_meta_objs.append(meta)
        logger.info(
            f"[split_speech_chunks] Saved chunk {idx}: {chunk_path}, "
            f"size={os.path.getsize(chunk_path)} bytes, start={start_time}ms, "
            f"end={end_time}ms, duration={duration}ms"
        )

    logger.info(
        f"[split_speech_chunks] Total chunks generated: {len(chunk_meta_objs)}"
    )
    return chunk_meta_objs

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    This function supports most common audio formats (wav, mp3, flac, ogg, etc.)
    via ffmpeg and pydub. It extracts the duration metadata using pydub's
    mediainfo utility.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        float: Duration of the audio file in seconds.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        KeyError: If duration information is not found in the file metadata.
        ValueError: If the duration value cannot be converted to float.
    """
    info = mediainfo(audio_path)
    return float(info['duration'])