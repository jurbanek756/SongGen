"""
lyrics/Text/Voice processor class for MusicGen
"""
import os
import torch
import librosa
import soundfile as sf
from transformers import PreTrainedProcessor, AutoTokenizer
from songgen import VoiceBpeTokenizer  
from transformers import Wav2Vec2FeatureExtractor
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio



class SongGenProcessor(PreTrainedProcessor):
    def __init__(self, ckpt_path):
        """
        Initializes the SongGenProcessor 
        """
        self.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side='right')
        self.lyrics_tokenizer = VoiceBpeTokenizer() 
        mert_path = 'm-a-p/MERT-v1-330M'
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, 
                                cache_dir=None,    
                                trust_remote_code=True)
        self.demucs = pretrained.get_model("htdemucs")
    
    def __call__(self, text, lyrics, ref_voice_path=None, start=0, separate=False, padding=True, return_tensors="pt"):
        """
        Processes the input text, lyrics, and audio file, and returns the tensors suitable for model input.

        :param text: A list of text descriptions for music generation.
        :param lyrics: Lyrics text for the music generation.
        :param ref_voice_path: Optional path to the reference voice for conditioning the generation.
        :param start: The starting time for the reference voice slice.
        :param separate: Whether to perform audio separation (if applicable).
        :param audio_prompt_path: Optional path to an additional audio prompt.
        :param astart, aend: Time range for audio slicing.
        :param padding: Whether to apply padding to inputs.
        :param return_tensors: Whether to return the tensors as PyTorch tensors.
        :return: A dictionary with the model's inputs, ready for inference.
        """
        # Process lyrics and convert them into token IDs (with padding and other settings)
        prompt_input_ids = [261] + self.voicebpe_tokenizer.encode(lyrics.strip().replace('\n', '.'), lang='en') + [0]
        print('Processed lyrics:', self.voicebpe_tokenizer.decode(prompt_input_ids))
        
        # Tokenize the lyrics and pad to max length
        lyrics_inputs = self.text_tokenizer.pad(
            [{"input_ids": prompt_input_ids}],
            return_tensors=return_tensors,
            padding="max_length",
            max_length=512
        ).to('cuda')  # Move the tensor to GPU

        # Tokenize the text descriptions (e.g., music genre or style)
        text_inputs = self.text_tokenizer(
            text,
            return_tensors=return_tensors,
            padding="max_length",
            max_length=512
        ).to('cuda')  # Move the tensor to GPU

        # Combine both the text and lyrics inputs
        batch = {
            **text_inputs,
            "prompt_input_ids": lyrics_inputs.input_ids,
            "prompt_attention_mask": lyrics_inputs.attention_mask
        }

        # Process reference voice (if provided)
        if ref_voice_path is not None:
            # Read the reference voice audio file
            wav, sr = sf.read(ref_voice_path)
            wav = wav.T  # Transpose for single channel audio
            wav = librosa.to_mono(wav)  # Convert to mono if stereo
            # Slice the audio according to the start and end times
            lidx = int(start * sr)
            ridx = lidx + int(3 * sr)  # Slice a 3-second segment
            wav = wav[lidx:ridx]

            if separate:
                # If audio separation is needed, process the audio (this can be customized)
                demucs_wav = self.convert_audio(
                    torch.tensor(wav[None], device='cuda').to(torch.float32),  # Move tensor to GPU
                    sr,
                    44100,  # Assuming target sampling rate
                    2  # Assuming 2 audio channels (stereo)
                )
                stems = self.apply_model(demucs_wav.unsqueeze(0))  # Apply the separation model
                wav = stems[0][-1:].sum(0).mean(0).cpu().numpy()  # Process the audio

            # Resample the audio to target sample rate if needed
            if sr != 16000:  # Assuming 16000 Hz as the target sampling rate
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                sr = 16000

            print('Reference voice processed:')
            # Optionally display or process the audio further (e.g., visualize it)
            input_wavs = [wav]
            # Using MertProcessor to handle audio inputs (assuming it exists)
            mert_processor = MertProcessor.from_pretrained('mert-model')  # Assuming this is an existing processor
            mert_inputs = mert_processor(
                input_wavs, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=48000
            )
            # Add reference voice values and attention mask to the batch
            batch['ref_voice_values'] = mert_inputs['input_values'].to('cuda')
            batch['ref_voice_attention_mask'] = mert_inputs['attention_mask'].to('cuda')

        return batch

    def convert_audio(self, audio_tensor, sr, target_sr, channels):
        """
        Converts audio (e.g., resample or apply audio processing).
        :param audio_tensor: The audio data as a tensor.
        :param sr: The sample rate of the input audio.
        :param target_sr: The target sample rate.
        :param channels: The number of audio channels (e.g., stereo or mono).
        :return: Processed audio tensor.
        """
        # This method can be expanded to include actual audio transformations (e.g., resampling)
        audio_tensor_resampled = audio_tensor  # No actual resampling here; replace with actual logic if needed
        return audio_tensor_resampled

    def apply_model(self, demucs_wav):
        """
        Applies the audio separation model (this is a placeholder for an actual model).
        :param demucs_wav: The waveform to apply the separation model to.
        :return: A list of separated audio tracks.
        """
        # Example: Apply audio separation (this can be customized with the actual model)
        return [demucs_wav]  # Return the same audio in this placeholder (replace with actual logic)

# Example usage:

# Assuming you already have a VoiceBpeTokenizer instance
voicebpe_tokenizer = VoiceBpeTokenizer()

# Initialize SongGenProcessor with the path to the tokenizer and VoiceBpeTokenizer
processor = SongGenProcessor(description_tokenizer_name_or_path="your-description-tokenizer-path", voicebpe_tokenizer=voicebpe_tokenizer)

# Use the processor to process text and lyrics
text_input = processor.process_text(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt"
)
