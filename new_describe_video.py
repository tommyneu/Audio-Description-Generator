from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
from scenedetect import detect, ContentDetector
import argparse
import os
import subprocess

DEBUG = False
SAVE_FILES = False

MODEL = 'gemma3:12b'
PROMPT = 'You are an audio description service for generating ADA compliant audio descriptions for this video. Your response should only include the description of what is happening in the image do not include any other text. Make sure the description is short and to the point.'
SCENE_DETECT_THRESHOLD = 27.0
FREEZE_FRAME_PADDING = 0.5

def detect_scenes(video_input):
    output = []
    if DEBUG:
        print('Detecting Scenes')
        print(f'- Scene Detect Threshold: {SCENE_DETECT_THRESHOLD}')

    # Using PySceneDetect to detect different scenes
    scene_list = detect(args.input, ContentDetector(threshold=SCENE_DETECT_THRESHOLD))
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
            print(f'- Scene {i:03}: Start {scene[0].get_timecode()} / Frame {scene[0].frame_num:05}, End {scene[1].get_timecode()} / Frame {scene[1].frame_num:05}')

    return output

def describe_image(image_input):
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
        capture_output=True
    )

    if result.returncode != 0:
        print("Ollama error:", result.stderr)
        return "No description available."

    return result.stdout.strip() or "No description available."

def text_to_audio_file(text, audio_output):
    if DEBUG:
        print('Text To Speech:')

    try:
        tts = gTTS(text=text, lang='en')
        tts.save(audio_output)
        return True
    except Exception as e:
        print(f"Error generating audio for '{text}': {e}")
        # Create a placeholder silent audio file if gTTS fails
        # A 2-second silent audio file
        subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '2', '-q:a', '9', '-acodec', 'libmp3lame', filename], check=True)
        return False

def process_video(video_input, video_output):
    if DEBUG:
        print('Processing Video')
        print(f'- Video Input: {video_input}')

    # Make tmp dir if not exists
    os.makedirs('./tmp', exist_ok=True)

    # Get the different scenes in the video
    scenes = detect_scenes(video_input)

    with VideoFileClip(video_input, audio=True) as whole_video:
        clips = []

        for scene in scenes:
            current_scene_clip = whole_video.subclipped(scene['start_timecode'], scene['end_timecode'])

            scene_middle_frame_file_name = f'./tmp/frame_{scene['scene_number']}.jpeg'
            middle_time = current_scene_clip.duration / 2;
            current_scene_clip.save_frame(scene_middle_frame_file_name, middle_time)

            image_description = describe_image(scene_middle_frame_file_name)
            if DEBUG:
                print(f'- Image Description: {image_description}')

            scene_description_narration_file_name = f'./tmp/frame_tts_{scene['scene_number']}.wav'
            text_to_audio_file(image_description, scene_description_narration_file_name)

            with AudioFileClip(scene_description_narration_file_name) as narration_track:
                freeze_frame_duration = narration_track.duration+FREEZE_FRAME_PADDING

                with ImageClip(scene_middle_frame_file_name, duration=freeze_frame_duration) as still_frame:
                    still_frame_clip = still_frame.with_audio(narration_track)

                    print(still_frame_clip.audio)

                    clips.append(still_frame_clip)

            if not SAVE_FILES and os.path.exists(scene_middle_frame_file_name):
                if DEBUG:
                    print(f'Deleting Frame Image: {scene_middle_frame_file_name}')
                os.remove(scene_middle_frame_file_name)
            if not SAVE_FILES and os.path.exists(scene_description_narration_file_name):
                if DEBUG:
                    print(f'Deleting Frame TTS: {scene_description_narration_file_name}')
                os.remove(scene_description_narration_file_name)

            clips.append(current_scene_clip)

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(video_output, codec="libx264", audio_codec="aac")

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
    parser.add_argument('-i', '--input', required=True, help='Path to input video file')
    parser.add_argument('-o', '--output', default='./described_paused.mp4', help='Path to output video file')
    parser.add_argument('-m', '--model', default=MODEL, help='ollama model to use for describing images')
    parser.add_argument('-p', '--prompt', default=PROMPT, help='ollama model to use for describing images')
    parser.add_argument('-ffp', '--freeze_frame_padding', default=FREEZE_FRAME_PADDING, help='duration padding after narration in freeze frame')
    parser.add_argument('-st', '--scene_threshold', default=SCENE_DETECT_THRESHOLD, help='scene detect threshold')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--save_files', action='store_true', help='Save tmp files')
    args = parser.parse_args()

    if args.debug == True:
        DEBUG = True

    if args.save_files == True:
        SAVE_FILES = True

    MODEL = args.model
    SCENE_DETECT_THRESHOLD = float(args.scene_threshold)

    process_video(args.input, args.output)
