import cv2
import subprocess
import json
from gtts import gTTS
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from rapidfuzz import fuzz
import argparse
import os

# -------------------------------
# Config
# -------------------------------
OLLAMA_MODEL = "gemma3:12b"
NARRATION_STYLE = "Describe this still frame from a video for a blind person. This text will be used for generating a audio description video for ADA compliance. Do not include any extra text in your response just the description. Description should be short and to the point."
SIMILARITY_THRESHOLD = 85  # skip narration if too similar to previous


# -------------------------------
# Ollama Helpers
# -------------------------------
def ollama_describe_image(image_path, prompt=NARRATION_STYLE):
    cmd = ["ollama", "run", OLLAMA_MODEL, f"{prompt}"]
    # Append image argument if Ollama CLI supports it
    if image_path:
        cmd.extend(['./' + image_path])

    print(cmd)
    print(image_path)

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print("Ollama error:", result.stderr)
        return "No description available."

    return result.stdout.strip() or "No description available."


# -------------------------------
# Scene Detection
# -------------------------------
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]


# -------------------------------
# TTS
# -------------------------------
def create_audio_from_text(text, filename):
    """
    Converts a given text into a WAV audio file using the gTTS library.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return True
    except Exception as e:
        print(f"Error generating audio for '{text}': {e}")
        # Create a placeholder silent audio file if gTTS fails
        # A 2-second silent audio file
        subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '2', '-q:a', '9', '-acodec', 'libmp3lame', filename], check=True)
        return False


# -------------------------------
# Main Pipeline
# -------------------------------
def process_video(video_path, output_path="described_paused.mp4"):
    clip = VideoFileClip(video_path)
    scenes = detect_scenes(video_path)
    print(f"Detected {len(scenes)} scenes")

    final_clips = []
    last_desc = None

    for i, (start, end) in enumerate(scenes):
        scene_clip = clip.subclip(start, end)

        # Extract middle frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) / 2) * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            final_clips.append(scene_clip)
            continue

        frame_path = f"frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)

        # Generate description
        desc = ollama_describe_image(frame_path).strip()
        print(f"[Scene {i}] {desc}")

        # Skip if too similar to previous
        if last_desc and fuzz.ratio(desc.lower(), last_desc.lower()) > SIMILARITY_THRESHOLD:
            print("⚠️ Skipping narration (too similar to previous)")
            final_clips.append(scene_clip)
            continue

        last_desc = desc

        # Narration audio
        tts_path = f"tts_{i}.wav"
        create_audio_from_text(desc, tts_path)
        narration_audio = AudioFileClip(tts_path)
        freeze_duration = narration_audio.duration + 0.5

        # Freeze-frame with narration
        freeze_frame = ImageClip(frame_path, duration=freeze_duration)
        freeze_frame = freeze_frame.set_audio(narration_audio)

        # Add freeze-frame + scene
        final_clips.extend([freeze_frame, scene_clip])

        # Optional: cleanup temp files
        # os.remove(frame_path)
        # os.remove(tts_path)

    # Combine everything
    final = concatenate_videoclips(final_clips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"✅ Done! Output saved to {output_path}")


# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio-described video with pauses.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="described_paused.mp4", help="Path to output video file")
    args = parser.parse_args()

    process_video(args.input, args.output)
