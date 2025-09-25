""" Module for converting text to an audio file """
import argparse
import torch
from TTS.api import TTS

def generate_audio(text:str, audio_output:str, model:str = 'tts_models/en/vctk/vits', language:str|None = None, speaker:str|None = 'p244'):
    """ Takes in a string of text and saves an audio file of the narration of that text """

    # Get device && init TTS
    tts_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_obj = TTS(model).to(tts_device)

    tts_obj.tts_to_file(text=text, file_path=audio_output, language=language, speaker=speaker)

# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect audio blocks from audio input')
    parser.add_argument('-t',
                        '--text',
                        required=True,
                        help='Text to be converted to speech')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Output file path')
    parser.add_argument('-m',
                        '--model',
                        default='tts_models/en/vctk/vits',
                        help='TTS model to run')
    parser.add_argument('-l',
                        '--language',
                        default='none',
                        help='TTS model language')
    parser.add_argument('-s',
                        '--speaker',
                        default='p244',
                        help='TTS model speaker')
    args = parser.parse_args()

    if args.language.lower() == 'none':
        args.language = None

    if args.speaker.lower() == 'none':
        args.speaker = None

    generate_audio(args.text, args.output, args.model, args.language, args.speaker)
