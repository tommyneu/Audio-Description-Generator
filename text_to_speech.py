""" Module for converting text to an audio file """
import torch
from TTS.api import TTS

def generate_audio(text:str, audio_output:str, model:str = "tts_models/en/vctk/vits", speaker:str = "p244"):
    """ Takes in a string of text and saves an audio file of the narration of that text """

    # Get device && init TTS
    tts_device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_obj = TTS(model).to(tts_device)

    tts_obj.tts_to_file(text=text, file_path=audio_output, speaker=speaker)
