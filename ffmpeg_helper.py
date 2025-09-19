""" FFMPEG helper functions for running common commands """
import subprocess

VIDEO_ENCODING = 'libx264'
ENCODER_PRESET = 'fast'
CONSTANT_RATE_FACTOR = '23'
PIXEL_FORMAT = 'yuv420p'
FRAME_RATE = '30'

AUDIO_ENCODING = 'aac'
BITRATE = '192k'
SAMPLE_RATE = '48000'
AUDIO_CHANNELS = '2'

STILL_FRAME_AUDIO_PADDING_SECONDS = '0.5'

def set_video_encoding(new_encoding):
    VIDEO_ENCODING = new_encoding

def normalize_video(video_input:str, video_output:str):
    """ Taking the input video and normalizing it to standard video parameters """
    cmd = [
        'ffmpeg', '-y', '-i', video_input,
        # Video settings
        '-c:v', VIDEO_ENCODING, '-preset', ENCODER_PRESET,
        '-crf', CONSTANT_RATE_FACTOR, '-pix_fmt', PIXEL_FORMAT, '-r', FRAME_RATE,
        # Audio settings
        '-c:a', AUDIO_ENCODING, '-b:a', BITRATE, '-ar', SAMPLE_RATE, '-ac', AUDIO_CHANNELS,
        video_output
    ]
    subprocess.run(cmd, check=True, stdout = subprocess.DEVNULL)

def cut_video_into_clip(video_input:str, video_output:str, start_time:str, end_time:str):
    """ Based on the start time and end time it will cut the video and save a copy """
    # Re-encode the clip to avoid issues when it comes to key frames
    cmd = [
        'ffmpeg', '-y',
        # Video input
        '-i', video_input,
        # Seek to start time and cut to end time
        '-ss', start_time, '-to', end_time,
        # Video settings
        '-c:v', VIDEO_ENCODING, '-preset', ENCODER_PRESET,
        '-crf', CONSTANT_RATE_FACTOR, '-pix_fmt', PIXEL_FORMAT, '-r', FRAME_RATE,
        # Audio settings
        '-c:a', AUDIO_ENCODING, '-b:a', BITRATE, '-ar', SAMPLE_RATE, '-ac', AUDIO_CHANNELS,
        video_output
    ]
    subprocess.run(cmd, check=True, stdout = subprocess.DEVNULL)

def save_first_frame_as_image(video_input:str, image_output:str):
    """ Saves the first frame from a video file and saves the image"""
    cmd = [
        'ffmpeg', '-y', '-i', video_input, '-frames:v', '1', '-update', '1',
        image_output
    ]
    subprocess.run(cmd, check=True, stdout = subprocess.DEVNULL)

def create_still_frame_narration_clip(audio_input:str, image_input:str, video_output:str):
    """ Using the image and audio file it will create a still frame video (with a bit of scilence at the end of the clip) """
    cmd = [
        'ffmpeg', '-y',
        # Loop image for the whole video
        '-loop', '1', '-i', image_input,
        # Add narration track
        '-i', audio_input,
        # Video settings
        '-c:v', VIDEO_ENCODING, '-tune', 'stillimage', '-preset', ENCODER_PRESET,
        '-crf', CONSTANT_RATE_FACTOR, '-pix_fmt', PIXEL_FORMAT, '-r', FRAME_RATE,
        # Audio settings
        '-c:a', AUDIO_ENCODING, '-b:a', BITRATE, '-ar', SAMPLE_RATE, '-ac', AUDIO_CHANNELS,
        # Make sure the output matches the shortest stream length
        '-shortest',
        video_output
    ]
    subprocess.run(cmd, check=True, stdout = subprocess.DEVNULL)

def export_clips_to_file(clips_file_output:str, clips: list):
    """ Taking a list of clips (file paths) it will write it to a file for ffmpeg """
    with open(clips_file_output, 'w', encoding='utf-8') as clips_file:
        for clip in clips:
            clips_file.write(f'file {clip.replace('./tmp/', '')}\n')

def combine_videos(video_output:str, clips_file_input: str):
    """ Taking a clips file it will concat all the videos together """
    cmd = [
        'ffmpeg', '-y',
        # Concat videos into one video clip
        '-f', 'concat', '-safe', '0', '-i', clips_file_input,
        # Video settings
        '-c:v', VIDEO_ENCODING, '-preset', ENCODER_PRESET,
        '-crf', CONSTANT_RATE_FACTOR, '-pix_fmt', PIXEL_FORMAT, '-r', FRAME_RATE,
        # Audio settings
        '-c:a', AUDIO_ENCODING, '-b:a', BITRATE, '-ar', SAMPLE_RATE, '-ac', AUDIO_CHANNELS,
        # Force FFMPEG to keep short bits of silent audio
        '-fflags', '+genpts',
        video_output
    ]
    subprocess.run(cmd, check=True, stdout = subprocess.DEVNULL)
