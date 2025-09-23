""" Module for detecting and parsing visual scenes in video """
from scenedetect import detect, ContentDetector

def get_visual_scenes(video_input:str, scene_detect_threshold=27.0) -> str:
    """ Takes in a video file path and returns videos scenes """
    output = []

    # Using PySceneDetect to detect different scenes
    scene_list = detect(video_input, ContentDetector(threshold=scene_detect_threshold))
    for i, scene in enumerate(scene_list):
        # Formatting scene data to something useable
        output.append({
            'scene_number': i,
            'start_timecode': scene[0].get_timecode(),
            'start_frame': scene[0].frame_num,
            'end_timecode': scene[1].get_timecode(),
            'end_frame': scene[1].frame_num,
        })

    return output
