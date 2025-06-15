import os
import torch
import soundfile as sf
from songgen import (
    SongGenMixedForConditionalGeneration,
    SongGenProcessor
)

# --- Configuration Constants ---
# Set these environment variables before running the script:
# export SONGGEN_LYRICS_FILENAME="your_lyrics.md"
# export SONGGEN_DESCRIPTION_FILENAME="your_description.md"
LYRICS_FILENAME_ENV = "SONGGEN_LYRICS_FILENAME"
DESCRIPTION_FILENAME_ENV = "SONGGEN_DESCRIPTION_FILENAME"

# Path to the pretrained model checkpoint
CKPT_PATH = "LiuZH-19/SongGen_mixed_pro"

# Directory where your prompt markdown files are located
PROMPTS_DIR = os.path.join("assets", "prompts")

# Output audio filename
OUTPUT_AUDIO_FILENAME = "songgen_mixed_out.wav"

# --- Utility Functions ---

def read_markdown_file(filepath: str) -> str:
    """
    Reads the content of a Markdown file from the specified filepath.
    Ensures the file exists before attempting to read.

    Args:
        filepath (str): The full path to the Markdown file.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: For other issues during file reading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file at '{filepath}' was not found. Please ensure it exists.")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise IOError(f"An error occurred while reading file '{filepath}': {e}")

def load_songgen_model(model_path: str, device: str):
    """
    Loads the SongGenMixedForConditionalGeneration model and its processor.

    Args:
        model_path (str): Path to the pretrained model checkpoint.
        device (str): Device to load the model onto (e.g., "cuda:0" or "cpu").

    Returns:
        tuple: (model, processor) loaded objects.
    """
    print(f"Loading model from {model_path} and moving to device: {device}")
    model = SongGenMixedForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation='sdpa'
    ).to(device)
    processor = SongGenProcessor(model_path, device)
    print("Model and processor loaded successfully.")
    return model, processor

def get_file_path_from_env(env_var_name: str, base_dir: str) -> str:
    """
    Retrieves a filename from an environment variable and constructs its full path.

    Args:
        env_var_name (str): The name of the environment variable.
        base_dir (str): The base directory where the file is expected.

    Returns:
        str: The full path to the file.

    Raises:
        ValueError: If the environment variable is not set.
    """
    filename = os.getenv(env_var_name)
    if not filename:
        raise ValueError(
            f"Environment variable '{env_var_name}' not set. "
            f"Please set it to the name of your markdown file (e.g., 'your_lyrics.md')."
        )
    return os.path.join(base_dir, filename)

# --- Main Execution ---
def main():
    """
    Main function to load model, read inputs, generate song, and save audio.
    """
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    model, processor = load_songgen_model(CKPT_PATH, device)

    # Read lyrics and text from markdown files using environment variables
    try:
        lyrics_filepath = get_file_path_from_env(LYRICS_FILENAME_ENV, PROMPTS_DIR)
        text_filepath = get_file_path_from_env(DESCRIPTION_FILENAME_ENV, PROMPTS_DIR)

        print(f"Attempting to read lyrics from: {lyrics_filepath}")
        lyrics = read_markdown_file(lyrics_filepath)
        print("Lyrics read successfully.")

        print(f"Attempting to read description from: {text_filepath}")
        text = read_markdown_file(text_filepath)
        print("Description read successfully.")

    except (ValueError, FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        print(
            "Please ensure your 'assets/prompts/' directory exists, "
            "contains the required markdown files, and that the environment "
            "variables are correctly set."
        )
        return # Exit if file reading fails
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        return

    # Prepare model inputs
    # For SongGenMixedForConditionalGeneration, ref_voice_path and separate are not used.
    print("Preparing model inputs...")
    model_inputs = processor(text=text, lyrics=lyrics)

    # Generate audio
    print("Generating audio (this may take some time)...")
    with torch.no_grad(): # Disable gradient calculations for inference
        generation = model.generate(**model_inputs, do_sample=True)

    # Save generated audio
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(OUTPUT_AUDIO_FILENAME, audio_arr, model.config.sampling_rate)
    print(f"Audio generated and saved to '{OUTPUT_AUDIO_FILENAME}'")

if __name__ == "__main__":
    main()
