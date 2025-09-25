""" Module for detecting and parsing visual scenes in video """
import argparse
from scenedetect import detect, ContentDetector

def get_visual_scenes(video_input:str, scene_detect_threshold=27.0) -> list:
    """ Takes in a video file path and returns videos scenes """
    output = []

    # Using PySceneDetect to detect different scenes
    scene_list = detect(video_input, ContentDetector(threshold=scene_detect_threshold))
    for i, scene in enumerate(scene_list):
        # Formatting scene data to something useable
        output.append({
            'scene_number': i,
            'start_timecode': scene[0].get_timecode(),
            'end_timecode': scene[1].get_timecode(),
        })

    return output

# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect visual scenes from video input')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to input video file')
    args = parser.parse_args()

    print(get_visual_scenes(args.input))
