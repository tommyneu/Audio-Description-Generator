""" Module will take in a video and save a new audio description video """

import argparse
import os
import atexit

import audio_block_detect
import describe_scene
import ffmpeg_helper
import text_to_speech
# import visual_scene_detect

DEBUG = False
SAVE_FILES = False

MODEL = 'gemma3:12b'

# pylint: disable=line-too-long
PROMPT = 'You are a video audio description service. These images are frames from a scene, describe the what is happening in the scene. Make sure your response is one coherent thought. Your response should contain only the description with no extra text, explanations, or conversational phrases.'
SIMILARLY_SCORE = 0.75
FRAMES_PER_CLIP=5

FILES = []

def _exit_handler():
    """ Cleans up files before exiting """

    if SAVE_FILES:
        return
    for file_path in FILES:
        if os.path.exists(file_path):
            os.remove(file_path)

atexit.register(_exit_handler)

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

    audio_file_path = "./tmp/video_audio_conversion.wav"
    FILES.append(audio_file_path)
    ffmpeg_helper.video_to_audio_wav(normalized_video_path, audio_file_path)

    clips_file_path = './tmp/clips_file.txt'
    FILES.append(clips_file_path)
    clips = []
    previous_descriptions = []

    # Get the different scenes in the video
    scenes = audio_block_detect.get_audio_blocks(audio_file_path)
    if DEBUG:
        print(scenes)
    for scene in scenes:
        _process_scene(normalized_video_path, scene, clips, previous_descriptions)

    if DEBUG:
        print('Exporting Clips')
    ffmpeg_helper.export_clips_to_file(clips_file_path, clips)

    ffmpeg_helper.combine_videos(video_output, clips_file_path)

    _exit_handler()

def _process_scene(video_input:str, scene:dict, clips:list, previous_descriptions:list):
    """ Processes a specific scene and appends scene video file paths to clips """

    if DEBUG:
        print('Processing Scene')
        print(f'- Scene: {scene["scene_number"]}')
        print('Clipping Scene')

    # Cut the scene from the video and save a copy
    scene_video_path = f'./tmp/scene_{scene["scene_number"]}.mp4'
    FILES.append(scene_video_path)
    ffmpeg_helper.cut_video_into_clip(video_input, scene_video_path, scene['start_timecode'], scene['end_timecode'])

    clip_duration = ffmpeg_helper.get_duration(scene_video_path)

    if DEBUG:
        print('Saving First Frame')
    # Get first frame of the video clip
    first_frame_path = f'./tmp/scene_{scene["scene_number"]}_first_frame.png'
    FILES.append(first_frame_path)
    ffmpeg_helper.save_first_frame_as_image(scene_video_path, first_frame_path)

    if DEBUG:
        print(f'Saving {FRAMES_PER_CLIP} Frames From Scene')
    frame_images = []
    frame_step = clip_duration / FRAMES_PER_CLIP
    for i in range(FRAMES_PER_CLIP):
        current_frame_time = frame_step * i
        current_frame_path = f'./tmp/scene_{scene["scene_number"]}_frame_{i}.jpeg'
        FILES.append(current_frame_path)
        ffmpeg_helper.save_frame_at_time_as_image(scene_video_path, current_frame_time, current_frame_path)
        frame_images.append(current_frame_path)

    # Get the image description
    if DEBUG:
        print('Describing Scene')
    image_description = describe_scene.generate_description(frame_images, PROMPT, MODEL)
    if DEBUG:
        print(f'- Image Description: {image_description}')

    skipping = False
    if len(previous_descriptions) > 0 and describe_scene.should_skip_description(image_description, previous_descriptions[-1], SIMILARLY_SCORE):
        skipping = True

    if image_description == '' or skipping:
        if DEBUG:
            print('No Description or Skipping Narration')
        clips.append(scene_video_path)
        return

    previous_descriptions.append(image_description)

    # Create an audio file of narration
    if DEBUG:
        print('Generating Narration')
    tts_path = f'./tmp/scene_{scene["scene_number"]}_tts.wav'
    FILES.append(tts_path)
    text_to_speech.generate_audio(image_description, tts_path)

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

    ffmpeg_helper.set_video_encoding(args.video_encoding)

    try:
        process_video(args.input, args.output)
    except KeyboardInterrupt:
        _exit_handler()
