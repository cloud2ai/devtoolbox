import os
import sys
import logging
from devtoolbox.speech.utils import split_speech_chunks, ChunkMeta

# Manual test for split_speech_chunks
# Usage: python test_split_speech_chunks.py <input_audio> <output_dir>

def main():
    """
    Manual test for split_speech_chunks.
    Pass input audio file and output directory as arguments.
    Prints the generated chunk metadata for each chunk.
    """
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("[test_split_speech_chunks] Logging initialized at DEBUG level.")
    if len(sys.argv) < 3:
        print("Usage: python test_split_speech_chunks.py <input_audio> <output_dir>")
        sys.exit(1)
    input_audio = sys.argv[1]
    output_dir = sys.argv[2]
    logging.debug(f"[test_split_speech_chunks] Input audio: {input_audio}")
    logging.debug(f"[test_split_speech_chunks] Output dir: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.debug(f"[test_split_speech_chunks] Created output dir: {output_dir}")
    # Use parameters to generate chunks between 5 and 10 minutes
    chunk_metas = split_speech_chunks(
        input_path=input_audio,
        output_dir=output_dir,
        min_chunk_duration=300000,      # 5 minutes
        max_chunk_duration=600000,      # 10 minutes
        vad_aggressiveness=2,
        max_wait_for_silence=120000     # 2 minutes
    )
    print(f"Total chunks: {len(chunk_metas)}")
    for meta in chunk_metas:
        print(f"Chunk {meta.index}: {meta.file_path}")
        print(f"  Start: {meta.start_time} ms")
        print(f"  End:   {meta.end_time} ms")
        print(f"  Duration: {meta.duration} ms")
        print()
        logging.debug(f"[test_split_speech_chunks] Chunk {meta.index}: {meta.file_path}, "
                      f"start={meta.start_time}, end={meta.end_time}, duration={meta.duration}")

if __name__ == "__main__":
    main()