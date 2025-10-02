""" Module will take in a video and save a new audio description video """

import argparse
import os
import atexit
import uuid
import time

import audio_block_detect
import describe_scene
import ffmpeg_helper
import text_to_speech
import visual_scene_detect

DEBUG = False
SAVE_FILES = False

MODEL = 'gemma3:12b'

# pylint: disable=line-too-long
PROMPT = 'You are a video audio description service. These images are frames from a scene, describe the what is happening in the scene. Make sure to describe all text in any of the images. Make sure your response is one coherent thought and never say "the scene", "the video", "the video frames", or "the images". Your response should contain only the description with no extra text, explanations, or conversational phrases.'
SIMILARLY_SCORE = 0.75
FRAMES_PER_CLIP=5

FILES = []

def _exit_handler():
    """ Cleans up files before exiting """

    for file_path in FILES:
        delete_tmp_file(file_path)

atexit.register(_exit_handler)

def delete_tmp_file(file_path):
    """ Deletes file """
    if SAVE_FILES:
        return

    if not os.path.exists(file_path):
        return

    if not os.path.isfile(file_path):
        raise ValueError('Trying to delete a non-file')

    file_path = os.path.abspath(file_path)
    directory_path = os.path.abspath('./tmp')
    if not os.path.dirname(file_path) == directory_path:
        raise ValueError('Trying to delete a file outside tmp folder')

    os.remove(file_path)

def timecode_to_seconds(timecode: str) -> float:
    """ Convert a timecode string (hh:mm:ss.ms) to total seconds as a float """
    hours, minutes, seconds = timecode.split(":")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds

def format_elapsed_time(elapsed_seconds: float) -> str:
    """ Returns a formatted time string for printing"""
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{minutes}m {seconds}s"

def debug_print(message):
    """ Print message is DEBUG is true """
    if DEBUG:
        print(message)

def process_video(video_input:str, video_output:str):
    """ Takes in video and saves an audio description version of the video """

    video_uuid = uuid.uuid4()

    debug_print('Processing Video')
    debug_print(f'- Video UUID: {video_uuid}')
    debug_print(f'- Video Input: {video_input}')
    debug_print('- Normalizing Video')

    # Make tmp dir if not exists
    os.makedirs('./tmp', exist_ok=True)
    tmp_path = f'./tmp/{video_uuid}_'

    normalized_video_path = f'{tmp_path}normalized_video.mp4'
    FILES.append(normalized_video_path)
    ffmpeg_helper.normalize_video(video_input, normalized_video_path)

    clips_file_path = f'{tmp_path}clips_file.txt'
    FILES.append(clips_file_path)

    # This is where the final video clip paths will be stored
    clips = []

    debug_print('- Converting to Audio')
    audio_file_path = f'{tmp_path}video_audio_conversion.wav'
    FILES.append(audio_file_path)
    ffmpeg_helper.video_to_audio_wav(normalized_video_path, audio_file_path)

    # Get the different scenes in the video
    debug_print('- Detecting Audio Blocks')
    audio_blocks = audio_block_detect.get_audio_blocks(audio_file_path)
    for audio_block in audio_blocks:
        debug_print(f'-- Audio Block: {audio_block["scene_number"]:03}, Start: {audio_block["start_timecode"]}, End: {audio_block["end_timecode"]}')

    debug_print('- Deleting audio conversion')
    delete_tmp_file(audio_file_path)

    # Get the different scenes in the video
    debug_print('- Detecting Video Blocks')
    video_blocks = visual_scene_detect.get_visual_scenes(normalized_video_path)
    for video_block in video_blocks:
        debug_print(f'-- Video Block: {video_block["scene_number"]:03}, Start: {video_block["start_timecode"]}, End: {video_block["end_timecode"]}')

    current_video_index = 0
    previous_description = None

    # Loop through audio blocks
    for audio_block in audio_blocks:
        scene_descriptions = []
        debug_print(f'- Processing Audio Block {audio_block["scene_number"]} / {len(audio_blocks) - 1}')

        debug_print('-- Clipping Audio Block')
        # Clip video at audio block start and end
        audio_block_clip_path = f'{tmp_path}audio_block_{audio_block["scene_number"]}.mp4'
        FILES.append(audio_block_clip_path)
        ffmpeg_helper.cut_video_into_clip(normalized_video_path, audio_block_clip_path, audio_block["start_timecode"], audio_block["end_timecode"])

        # Loop through the video loop from current_video_index
        for video_block in video_blocks[current_video_index:]:
            if timecode_to_seconds(video_block['start_timecode']) > timecode_to_seconds(audio_block['end_timecode']):
                break

            debug_print(f'-- Processing Video Block {video_block["scene_number"]} / {len(video_blocks) - 1}')
            # Increment current_video_index
            current_video_index += 1

            debug_print('--- Clipping Video Block')
            # Clip video at scene start and end
            video_block_clip_path = f'{tmp_path}video_block_{video_block["scene_number"]}.mp4'
            FILES.append(video_block_clip_path)
            ffmpeg_helper.cut_video_into_clip(normalized_video_path, video_block_clip_path, video_block["start_timecode"], video_block["end_timecode"])

            # Get duration of clip
            video_block_duration = ffmpeg_helper.get_duration(video_block_clip_path)

            debug_print(f'--- Saving {FRAMES_PER_CLIP} of video block')
            # Get `FRAMES_PER_CLIP` evenly spaced frames from clip
            frame_images = []
            frame_step = video_block_duration / FRAMES_PER_CLIP
            for i in range(FRAMES_PER_CLIP):
                current_frame_time = frame_step * i
                current_frame_path = f'{tmp_path}video_block_{video_block["scene_number"]}_frame_{i}.jpeg'
                FILES.append(current_frame_path)
                ffmpeg_helper.save_frame_at_time_as_image(video_block_clip_path, current_frame_time, current_frame_path)
                frame_images.append(current_frame_path)

            debug_print('--- Deleting video block clip')
            delete_tmp_file(video_block_clip_path)

            debug_print('--- Describing video block')
            debug_print(f'---- Model: {MODEL}')
            video_block_description = describe_scene.generate_description(frame_images, PROMPT, MODEL)
            debug_print(f'---- Description: {video_block_description}')

            debug_print('--- Deleting video block frames')
            for file_path in frame_images:
                delete_tmp_file(file_path)

            if video_block_description == '':
                debug_print('---- Missing Description')
                continue

            if previous_description is not None:
                if describe_scene.should_skip_description(previous_description, video_block_description, SIMILARLY_SCORE):
                    debug_print('---- Description too similar to last one ... Skipping')
                    continue

            previous_description = video_block_description
            scene_descriptions.append(video_block_description)

        if len(scene_descriptions) > 0:

            # Combine the descriptions somehow
            combined_description = ' '.join(scene_descriptions)

            debug_print('-- Saving first frame of audio block')
            # Get the first frame from this block
            audio_block_first_frame_path = f'{tmp_path}audio_block_{audio_block["scene_number"]}_first_frame.png'
            FILES.append(audio_block_first_frame_path)
            ffmpeg_helper.save_first_frame_as_image(audio_block_clip_path, audio_block_first_frame_path)

            debug_print('-- Generating narration track')
            # Get TTS from descriptions
            narration_track_path = f'{tmp_path}audio_block_{audio_block["scene_number"]}_narration_track.wav'
            FILES.append(narration_track_path)
            text_to_speech.generate_audio(combined_description, narration_track_path)

            debug_print('-- Exporting narration clip')
            # Create still narration clip
            audio_block_narration_clip_path = f'{tmp_path}audio_block_{audio_block["scene_number"]}_narration_clip.mp4'
            FILES.append(audio_block_narration_clip_path)
            ffmpeg_helper.create_still_frame_narration_clip(narration_track_path, audio_block_first_frame_path, audio_block_narration_clip_path)

            # Append narration clip to clips
            clips.append(audio_block_narration_clip_path)

            debug_print('-- Deleting narration track & first frame of audio block')
            delete_tmp_file(audio_block_first_frame_path)
            delete_tmp_file(narration_track_path)

        # Append audio block clip to clips
        clips.append(audio_block_clip_path)

    debug_print('- Exporting Clips')
    ffmpeg_helper.export_clips_to_file(clips_file_path, clips)

    debug_print('- Combining Clips')
    ffmpeg_helper.combine_videos(video_output, clips_file_path)

    debug_print('- Finished and cleaning up files')
    _exit_handler()

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

    # Record the start time
    start_time = time.time()

    try:
        process_video(args.input, args.output)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        debug_print(format_elapsed_time(elapsed_time))

    except KeyboardInterrupt:
        _exit_handler()
