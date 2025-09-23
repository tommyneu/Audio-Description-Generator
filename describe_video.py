""" Module will take in a video and save a new audio description video """

import argparse
import os
import atexit

import torch
from TTS.api import TTS
from scenedetect import detect, ContentDetector
import ollama
import numpy as np

import ffmpeg_helper

DEBUG = False
SAVE_FILES = False

MODEL = 'gemma3:12b'

# pylint: disable=line-too-long
PROMPT = 'Describe this image. Keep your response short and to the point. Your response should be general do not give too much detail. Your response should contain only the description of the image with no extra text, explanations, or conversational phrases.'
SCENE_DETECT_THRESHOLD = 27.0
FREEZE_FRAME_PADDING = 0.5
SIMILARLY_SCORE = 0.75

FILES = []

TTS_DEVICE = None
TTS_OBJ = None

def exit_handler():
    """ Cleans up files before exiting """

    if SAVE_FILES:
        return
    for file_path in FILES:
        if os.path.exists(file_path):
            os.remove(file_path)

atexit.register(exit_handler)

def detect_scenes(video_input:str) -> str:
    """ Takes in a video file path and returns videos scenes """
    output = []
    if DEBUG:
        print('Detecting Scenes')
        print(f'- Scene Detect Threshold: {SCENE_DETECT_THRESHOLD}')

    # Using PySceneDetect to detect different scenes
    scene_list = detect(video_input, ContentDetector(threshold=SCENE_DETECT_THRESHOLD))
    for i, scene in enumerate(scene_list):
        # Formatting scene data to something useable
        output.append({
            'scene_number': i,
            'start_timecode': scene[0].get_timecode(),
            'start_frame': scene[0].frame_num,
            'end_timecode': scene[1].get_timecode(),
            'end_frame': scene[1].frame_num,
        })
        if DEBUG:
            print(f"""- Scene {i:03}:Start {scene[0].get_timecode()} /Frame {scene[0].frame_num:05}, End {scene[1].get_timecode()} /Frame {scene[1].frame_num:05}""")

    return output

def describe_image(image_input: str, retries: int = 0) -> str:
    """Takes in an image path and returns a description of that image"""
    if DEBUG:
        print('Describing Image')
        print(f'- Model: {MODEL}')
        print(f'- Image Path: {image_input}')

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                'role': 'user',
                'content': PROMPT,
                'images': [image_input]  # this attaches the image
            }]
        )

        if DEBUG:
            print(f'- Raw Response: {response}')

        # Extract the text from the response
        description = response['message']['content'].strip()
        return description or ''

    # pylint: disable=broad-exception-caught
    except Exception as e:
        if DEBUG:
            print(f'- Error: {e}')
            print(f'- Retrying: {retries + 1}')
        if retries < 3:
            return describe_image(image_input, retries + 1)
        return ''

def semantic_similarity(text1: str, text2: str) -> float:
    """ Using Ollama embeds returns a score for how similar two strings are """
    # Get embeddings from Ollama
    e1 = ollama.embeddings(model="nomic-embed-text", prompt=text1)["embedding"]
    e2 = ollama.embeddings(model="nomic-embed-text", prompt=text2)["embedding"]

    vec1 = np.array(e1)
    vec2 = np.array(e2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def should_skip_description(prev: str, curr: str) -> bool:
    """ Checks to see if two strings are too similar """
    score = semantic_similarity(prev, curr)
    print(f"Similarity: {score:.3f}")
    return score >= SIMILARLY_SCORE

def text_to_audio_file(text:str, audio_output:str):
    """ Takes in a string of text and saves an audio file of the narration of that text """
    if DEBUG:
        print('Text To Speech')

    # tts = gTTS(text=text, lang='en')
    # tts.save(audio_output)

    TTS_OBJ.tts_to_file(text=text, file_path=audio_output, speaker="p340")

def process_video(video_input:str, video_output:str):
    """ Takes in video and saves an audio description version of the video """
    if DEBUG:
        print('Processing Video')
        print(f'- Video Input: {video_input}')
        print('- Normalizing Video')

    # Make tmp dir if not exists
    os.makedirs('./tmp', exist_ok=True)
    normalized_video_path = './tmp/normalized_video.mp4'
    FILES.append(normalized_video_path)
    ffmpeg_helper.normalize_video(video_input, normalized_video_path)

    clips_file_path = './tmp/clips_file.txt'
    FILES.append(clips_file_path)
    clips = []
    previous_descriptions = []

    # Get the different scenes in the video
    scenes = detect_scenes(video_input)
    for scene in scenes:
        process_scene(normalized_video_path, scene, clips, previous_descriptions)

    if DEBUG:
        print('Exporting Clips')
    ffmpeg_helper.export_clips_to_file(clips_file_path, clips)

    ffmpeg_helper.combine_videos(video_output, clips_file_path)

    exit_handler()

def process_scene(video_input:str, scene:dict, clips:list, previous_descriptions:list):
    """ Processes a specific scene and appends scene video file paths to clips """

    if DEBUG:
        print('Processing Scene')
        print(f'- Scene: {scene["scene_number"]}')
        print('Clipping Scene')

    # Cut the scene from the video and save a copy
    scene_video_path = f'./tmp/scene_{scene["scene_number"]}.mp4'
    FILES.append(scene_video_path)
    ffmpeg_helper.cut_video_into_clip(video_input, scene_video_path, scene['start_timecode'], scene['end_timecode'])

    if DEBUG:
        print('Saving First Frame')
    # Get first frame of the video clip
    first_frame_path = f'./tmp/scene_{scene["scene_number"]}_image.jpeg'
    FILES.append(first_frame_path)
    ffmpeg_helper.save_first_frame_as_image(scene_video_path, first_frame_path)

    # Get the image description
    image_description = describe_image(first_frame_path)
    if DEBUG:
        print(f'- Image Description: {image_description}')

    skipping = False
    if len(previous_descriptions) > 0 and should_skip_description(image_description, previous_descriptions[-1]):
        skipping = True

    if image_description == '' or skipping:
        if DEBUG:
            print('No Description or Skipping Narration')
        clips.append(scene_video_path)
        return

    previous_descriptions.append(image_description)

    # Create an audio file of narration
    tts_path = f'./tmp/scene_{scene["scene_number"]}_tts.aiff'
    FILES.append(tts_path)
    text_to_audio_file(image_description, tts_path)

    if DEBUG:
        print('Creating Still Frame Video')
    # Create a still frame video clip
    still_frame_path = f'./tmp/scene_{scene["scene_number"]}_still_frame.mp4'
    FILES.append(still_frame_path)
    ffmpeg_helper.create_still_frame_narration_clip(tts_path, first_frame_path, still_frame_path)

    clips.append(still_frame_path)
    clips.append(scene_video_path)


# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate audio-described video with pauses.')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to input video file')
    parser.add_argument('-o',
                        '--output',
                        default='./audio_description_video.mp4',
                        help='Path to output video file')
    parser.add_argument('-m',
                        '--model',
                        default=MODEL,
                        help='Ollama model to use for describing images')
    parser.add_argument('-p',
                        '--prompt',
                        default=PROMPT,
                        help='Ollama model to use for describing images')
    parser.add_argument('-ffp',
                        '--freeze_frame_padding',
                        default=FREEZE_FRAME_PADDING,
                        help='Duration padding after narration in freeze frame')
    parser.add_argument('-st',
                        '--scene_threshold',
                        default=SCENE_DETECT_THRESHOLD,
                        help='Scene detect threshold')
    parser.add_argument('-ss',
                        '--similarity_score',
                        default=SIMILARLY_SCORE,
                        help='Similarly score between two scene descriptions')
    parser.add_argument('-ve',
                        '--video_encoding',
                        default=ffmpeg_helper.VIDEO_ENCODING,
                        help='Video encoding for ffmpeg')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug mode')
    parser.add_argument('--ffmpeg_debug',
                        action='store_true',
                        help='FFMPEG debug mode')
    parser.add_argument('--save_files',
                        action='store_true',
                        help='Save tmp files')
    args = parser.parse_args()

    if args.debug is True:
        DEBUG = True

    if args.ffmpeg_debug is True:
        ffmpeg_helper.set_debug(True)

    if args.save_files is True:
        SAVE_FILES = True

    MODEL = args.model
    SCENE_DETECT_THRESHOLD = float(args.scene_threshold)

    ffmpeg_helper.set_video_encoding(args.video_encoding)

    # Get device &&  Init TTS
    TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TTS_OBJ = TTS("tts_models/en/vctk/vits").to(TTS_DEVICE)

    try:
        process_video(args.input, args.output)
    except KeyboardInterrupt:
        exit_handler()
