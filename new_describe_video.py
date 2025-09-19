""" Module will take in a video and save a new audio description video """

import argparse
import os
import subprocess

from gtts import gTTS
from scenedetect import detect, ContentDetector

import ffmpeg_helper

DEBUG = False
SAVE_FILES = False

MODEL = 'gemma3:12b'

# pylint: disable=line-too-long
PROMPT = 'You are an audio description service for generating ADA compliant audio descriptions for this video. Your response should only include the description of what is happening in the image do not include any other text. Make sure the description is short and to the point.'
SCENE_DETECT_THRESHOLD = 27.0
FREEZE_FRAME_PADDING = 0.5

def detect_scenes(video_input):
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

def describe_image(image_input, retries=0):
    """ Takes in an image path and returns a description of that image """
    if DEBUG:
        print('Describing Image')
        print(f'- Model: {MODEL}')
        print(f'- Image Path: {image_input}')

    cmd = ["ollama", "run", MODEL, f"{PROMPT}"]
    # Append image argument if Ollama CLI supports it
    if image_input:
        cmd.extend([image_input])

    if DEBUG:
        print(f'- Command: {cmd}')

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False
    )

    if result.returncode != 0:
        if DEBUG:
            print(f'- Retrying: {retries + 1}')
        if retries < 3:
            return describe_image(image_input, retries + 1)
        else: 
            return ''

    return result.stdout.strip() or ''

def text_to_audio_file(text, audio_output):
    """ Takes in a string of text and saves an audio file of the narration of that text """
    if DEBUG:
        print('Text To Speech:')

    tts = gTTS(text=text, lang='en')
    tts.save(audio_output)

def process_video(video_input, video_output):
    """ Takes in video and saves an audio description version of the video """
    if DEBUG:
        print('Processing Video')
        print(f'- Video Input: {video_input}')
        print('- Normalizing Video')

    # Make tmp dir if not exists
    os.makedirs('./tmp', exist_ok=True)
    normalized_video_path = './tmp/normalized_video.mp4'
    ffmpeg_helper.normalize_video(video_input, normalized_video_path)

    clips_file_path = './tmp/clips_file.txt'
    clips = []

    # Get the different scenes in the video
    scenes = detect_scenes(video_input)
    for scene in scenes:
        if DEBUG:
            print('Processing Scene')
            print(f'- Scene: {scene['scene_number']}')
            print('Clipping Scene')

        # Cut the scene from the video and save a copy
        scene_video_path = f'./tmp/scene_{scene['scene_number']}.mp4'
        ffmpeg_helper.cut_video_into_clip(normalized_video_path, scene_video_path, scene['start_timecode'], scene['end_timecode'])

        if DEBUG:
            print('Saving First Frame')
        # Get first frame of the video clip
        first_frame_path = f'./tmp/scene_{scene['scene_number']}_image.jpeg'
        ffmpeg_helper.save_first_frame_as_image(scene_video_path, first_frame_path)

        if DEBUG:
            print('Describing Image')
        # Get the image description
        image_description = describe_image(first_frame_path)

        if image_description == '':
            if DEBUG:
                print('No Description Skipping Narration')
            clips.append(scene_video_path)
            continue

        if DEBUG:
            print('Creating Narration')
        # Create an audio file of narration
        tts_path = f'./tmp/scene_{scene['scene_number']}_tts.mp3'
        text_to_audio_file(image_description, tts_path)

        if DEBUG:
            print('Creating Still Frame Video')
        # Create a still frame video clip
        still_frame_path = f'./tmp/scene_{scene['scene_number']}_still_frame.mp4'
        ffmpeg_helper.create_still_frame_narration_clip(tts_path, first_frame_path, still_frame_path)

        clips.append(still_frame_path)
        clips.append(scene_video_path)

    if DEBUG:
        print('Exporting Clips')
    ffmpeg_helper.export_clips_to_file(clips_file_path, clips)

    ffmpeg_helper.combine_videos(video_output, clips_file_path)

        # Load the video from `video_input`
        # Create clips list for all the various clips
        # Loop through the `scenes`
            # Cut original video into the current scene
            # Get the middle frame from the scene
            # Use `Ollama` to describe image
            # Use `GTTS` to make narration audio
            # Load narration audio
            # Create image frame with duration of audio
            # Save image clip with audio to clips list
            # Save scene to clips list
            # Release and delete audio file
        # Concatinate all the clips into final video file and save it to video_output

    # print(scenes)
    # print(video_input, video_output)


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
                        help='ollama model to use for describing images')
    parser.add_argument('-p',
                        '--prompt',
                        default=PROMPT,
                        help='ollama model to use for describing images')
    parser.add_argument('-ffp',
                        '--freeze_frame_padding',
                        default=FREEZE_FRAME_PADDING,
                        help='duration padding after narration in freeze frame')
    parser.add_argument('-st',
                        '--scene_threshold',
                        default=SCENE_DETECT_THRESHOLD,
                        help='scene detect threshold')
    parser.add_argument('-ve',
                        '--video_encoding',
                        default=ffmpeg_helper.VIDEO_ENCODING,
                        help='Video encoding for ffmpeg')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Debug mode')
    parser.add_argument('--save_files',
                        action='store_true',
                        help='Save tmp files')
    args = parser.parse_args()

    if args.debug is True:
        DEBUG = True

    if args.save_files is True:
        SAVE_FILES = True

    MODEL = args.model
    SCENE_DETECT_THRESHOLD = float(args.scene_threshold)

    ffmpeg_helper.set_video_encoding(args.video_encoding)

    process_video(args.input, args.output)
