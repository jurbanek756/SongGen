import torch
import os
import soundfile as sf
from songgen import (
    SongGenMixedForConditionalGeneration,
    SongGenProcessor
)

# --- Configuration Constants ---
# Path to the pretrained model checkpoint.
# This model is specifically for the "mixed mode" output, which combines vocals and accompaniment.[1]
CKPT_PATH = "LiuZH-19/SongGen_mixed_pro"

# Directory for input prompt files.
# Ensure this directory exists and contains your markdown files.
PROMPTS_DIR = os.path.join("assets", "prompts")

# Filenames for lyrics and textual description.
# The textual description can control musical attributes like instrumentation, genre, mood, and timbre.[2, 3, 1]
LYRICS_FILENAME = "baby_keem_lyrics.md"
TEXT_FILENAME = "baby_keem_description.md"

# Output filename for the generated audio.
OUTPUT_FILENAME = "songgen_out.wav"

def read_markdown_file(filepath: str) -> str:
    """
    Reads the content of a Markdown file from the specified filepath.
    Ensures the file exists before attempting to read.

    Args:
        filepath (str): The full path to the Markdown file.

    Returns:
        str: The content of the Markdown file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file at '{filepath}' was not found. Please ensure it exists.")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_songgen_model(ckpt_path: str, device: str):
    """
    Loads the SongGen model and processor.

    Args:
        ckpt_path (str): Path or name of the pretrained model.
        device (str): The device to load the model onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    print(f"Loading model from {ckpt_path} and moving to device: {device}")
    model = SongGenMixedForConditionalGeneration.from_pretrained(
        ckpt_path,
        attn_implementation='sdpa' # Using SDPA for potential performance optimization on compatible hardware.
    ).to(device)
    processor = SongGenProcessor(ckpt_path, device)
    print("Model and processor loaded successfully.")
    return model, processor

def generate_song(model, processor, text: str, lyrics: str, device: str):
    """
    Prepares model inputs and generates audio.

    Args:
        model: The loaded SongGen model.
        processor: The loaded SongGen processor.
        text (str): Textual description for musical attributes (e.g., genre, mood, instrumentation).
        lyrics (str): Lyrical content for the song.
        device (str): The device the model is on.

    Returns:
        numpy.ndarray: The generated audio array.
    """
    print("Preparing model inputs...")
    # The processor takes 'text' for general description (genre, mood, instrumentation)
    # and 'lyrics' for the lyrical content.[2, 3, 1]
    model_inputs = processor(text=text, lyrics=lyrics)

    print("Generating audio (this may take some time)...")
    # Use torch.no_grad() for inference to reduce memory usage and speed up computation.
    with torch.no_grad():
        generation = model.generate(
            **model_inputs,
            do_sample=True, # Enables sampling for more diverse outputs.
            # The 'ref_voice_path' and 'separate' parameters are not included here as per previous request in the original code.
            # SongGen also supports optional voice cloning with a 3-second reference clip [2, 3, 1]
            # and a 'dual-track mode' for separate vocal and accompaniment outputs.[2, 3, 1]
            # These features can be enabled by modifying the processor inputs and generation parameters if desired.
        )

    # Move generated tensor to CPU and convert to numpy array.
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr

def save_audio(audio_array, sampling_rate, output_path: str):
    """
    Saves the generated audio array to a WAV file.

    Args:
        audio_array (numpy.ndarray): The audio data to save.
        sampling_rate (int): The sampling rate of the audio.
        output_path (str): The full path to save the WAV file.
    """
    sf.write(output_path, audio_array, sampling_rate)
    print(f"Audio generated and saved to '{output_path}'")

if __name__ == "__main__":
    # Determine the device to use (CUDA if available, otherwise CPU).
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the model and processor.
    model, processor = load_songgen_model(CKPT_PATH, DEVICE)

    # Construct full file paths for lyrics and text.
    lyrics_filepath = os.path.join(PROMPTS_DIR, LYRICS_FILENAME)
    text_filepath = os.path.join(PROMPTS_DIR, TEXT_FILENAME)
    output_filepath = OUTPUT_FILENAME

    # Read lyrics and text from Markdown files.
    try:
        print(f"Attempting to read lyrics from: {lyrics_filepath}")
        lyrics_content = read_markdown_file(lyrics_filepath)
        print("Lyrics read successfully.")

        print(f"Attempting to read description from: {text_filepath}")
        text_description = read_markdown_file(text_filepath)
        print("Description read successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure your '{PROMPTS_DIR}' directory exists and contains the required markdown files.")
        exit(1) # Exit with an error code
    except Exception as e:
        print(f"An unexpected error occurred while reading files: {e}")
        exit(1) # Exit with an error code

    # Generate audio.
    generated_audio = generate_song(model, processor, text_description, lyrics_content, DEVICE)

    # Save the generated audio.
    save_audio(generated_audio, model.config.sampling_rate, output_filepath)
