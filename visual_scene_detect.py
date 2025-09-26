""" Module for detecting and parsing visual scenes in video """
import argparse
from scenedetect import detect, ContentDetector

def get_visual_scenes(video_input:str, scene_detect_threshold:float = 27.0, min_length:float = 1.0) -> list:
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

    output = _merge_short_scenes(output, min_length)

    return output

def _timecode_to_seconds(timecode: str) -> float:
    """ Convert a timecode string (hh:mm:ss.ms) to total seconds as a float """
    hours, minutes, seconds = timecode.split(":")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds

def _merge_short_scenes(scenes: list, min_duration: float = 1.0) -> list:
    """Merge consecutive scenes if duration < min_duration seconds"""
    if not scenes:
        return []

    merged = []
    current_scene = scenes[0]

    for next_scene in scenes[1:]:
        # Duration of current scene
        start = _timecode_to_seconds(current_scene['start_timecode'])
        end = _timecode_to_seconds(current_scene['end_timecode'])
        duration = end - start

        if duration < min_duration:
            # Merge forward (extend current scene to next scene's end)
            current_scene['end_timecode'] = next_scene['end_timecode']
        else:
            # Lock current scene and move on
            merged.append(current_scene)
            current_scene = next_scene

    # Handle last scene
    start = _timecode_to_seconds(current_scene['start_timecode'])
    end = _timecode_to_seconds(current_scene['end_timecode'])
    duration = end - start

    if duration < min_duration and merged:
        # Merge backward: extend previous scene’s end to this scene’s end
        merged[-1]['end_timecode'] = current_scene['end_timecode']
    else:
        merged.append(current_scene)

    # Reassign scene numbers
    for i, scene in enumerate(merged):
        scene['scene_number'] = i

    return merged

# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect visual scenes from video input')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to input video file')
    parser.add_argument('-st',
                        '--scene_threshold',
                        default=27.0,
                        help='Threshold for scene detect')
    parser.add_argument('-ml',
                        '--min_length',
                        default=1.0,
                        help='Min length for a visual scene')
    args = parser.parse_args()

    video_blocks = get_visual_scenes(args.input, float(args.scene_threshold), float(args.min_length))
    for video_block in video_blocks:
        print(f'Video Block: {video_block["scene_number"]:03}, Start: {video_block["start_timecode"]}, End: {video_block["end_timecode"]}')
