""" Simple single-pass scene detection and optimal frame selection using CLIP """

import os
import argparse
import atexit
import json

import torch
import clip
from PIL import Image

import ffmpeg_helper

SAVE_FILES = False
FILES = []
DEBUG = False

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

def debug_print(message, extra_new_line:bool = False):
    """ Print message is DEBUG is true """
    if DEBUG:
        if extra_new_line:
            print(message, end=os.linesep+os.linesep)
        else:
            print(message)

def _get_embedding(frame_path, model, preprocess, device):
    """ Convert an image file to a normalized CLIP embedding """
    image = Image.open(frame_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding

def _compute_cosine_similarity(a, b):
    """ Determines how similar two tensors are """
    # pylint: disable=not-callable
    return torch.nn.functional.cosine_similarity(a, b).item()

def _get_frame_embedding_at_time(video_path, timestamp, model, preprocess, device):
    """ Helper to get embedding for a specific timestamp """
    tmpdir = f'./tmp/frame_{timestamp}.png'
    FILES.append(tmpdir)
    ffmpeg_helper.save_frame_at_time_as_image(video_path, timestamp, tmpdir, 224)
    emb = _get_embedding(tmpdir, model, preprocess, device)
    delete_tmp_file(tmpdir)
    return emb

def analyze_video_single_pass(video_path: str, sample_interval: float = 1.0) -> list:
    """
    Single pass through video to compute frame-to-frame similarities

    Args:
        video_path: Path to video file
        sample_interval: Seconds between sampled frames

    Returns:
        List of dictionaries with timestamp and similarity to previous frame
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    duration = ffmpeg_helper.get_duration(video_path)
    debug_print(f"Video duration: {duration}s")
    debug_print(f"Sample interval: {sample_interval}s")
    debug_print("")

    similarities = []
    prev_emb = None
    prev_timestamp = None

    current_time = 0
    while current_time < duration:
        debug_print(f'Processing frame at {current_time:.2f}s / {duration:.2f}s')

        emb = _get_frame_embedding_at_time(video_path, current_time, model, preprocess, device)

        if prev_emb is not None:
            similarity = _compute_cosine_similarity(emb, prev_emb)
            similarities.append({
                'timestamp': current_time,
                'prev_timestamp': prev_timestamp,
                'similarity': similarity
            })
            debug_print(f'  Similarity to previous: {similarity:.3f}')

        prev_emb = emb
        prev_timestamp = current_time
        current_time += sample_interval

    _exit_handler()
    return similarities

def detect_cuts_from_similarities(
        similarities: list,
        cut_threshold: float = 0.85,
        min_scene_length: float = 2.0) -> list:
    """
    Detect scene cuts based on similarity drops

    Args:
        similarities: List from analyze_video_single_pass
        cut_threshold: Similarity below this is considered a cut
        min_scene_length: Minimum scene duration (shorter scenes merged)

    Returns:
        List of cut timestamps
    """
    debug_print("\n=== Detecting Cuts ===")

    # Find all potential cuts
    potential_cuts = []
    for item in similarities:
        if item['similarity'] < cut_threshold:
            potential_cuts.append(item['timestamp'])
            debug_print(f"Potential cut at {item['timestamp']:.2f}s (similarity: {item['similarity']:.3f})")

    if not potential_cuts:
        debug_print("No cuts detected")
        return []

    # Merge cuts that are too close together (likely same scene transition)
    merged_cuts = [potential_cuts[0]]
    for cut in potential_cuts[1:]:
        if cut - merged_cuts[-1] >= min_scene_length:
            merged_cuts.append(cut)
        else:
            debug_print(f"  Merging cut at {cut:.2f}s (too close to {merged_cuts[-1]:.2f}s)")

    debug_print(f"\nFinal cuts: {len(merged_cuts)}")
    return merged_cuts

def find_optimal_frames_per_scene(
        similarities: list,
        cuts: list,
        duration: float,
        frames_per_scene: int = 3,
        method: str = 'lowest_similarity') -> list:
    """
    Find optimal frames for audio description within each scene

    Args:
        similarities: List from analyze_video_single_pass
        cuts: List of cut timestamps
        duration: Total video duration
        frames_per_scene: Number of frames to select per scene
        method: 'lowest_similarity' (most distinct) or 'highest_similarity' (most typical)

    Returns:
        List of scenes with optimal frames
    """
    debug_print("\n=== Finding Optimal Frames ===")

    # Create scene boundaries
    scene_boundaries = [0] + cuts + [duration]
    scenes = []

    for i in range(len(scene_boundaries) - 1):
        scene_start = scene_boundaries[i]
        scene_end = scene_boundaries[i + 1]
        scene_duration = scene_end - scene_start

        debug_print(f"\nScene {i+1}: {scene_start:.2f}s - {scene_end:.2f}s (duration: {scene_duration:.2f}s)")

        # Get all frames within this scene
        scene_frames = []
        for item in similarities:
            # Include frames that fall within scene bounds
            if scene_start <= item['prev_timestamp'] < scene_end:
                scene_frames.append({
                    'timestamp': item['prev_timestamp'],
                    'similarity': item['similarity']
                })
            if scene_start <= item['timestamp'] < scene_end:
                # Also consider the current timestamp
                if not any(f['timestamp'] == item['timestamp'] for f in scene_frames):
                    scene_frames.append({
                        'timestamp': item['timestamp'],
                        'similarity': item['similarity']
                    })

        # Sort frames by similarity (ascending for distinct, descending for typical)
        if method == 'lowest_similarity':
            # Most distinct frames (different from neighbors)
            scene_frames.sort(key=lambda x: x['similarity'])
            score_label = 'distinctiveness'
        else:  # 'highest_similarity'
            # Most typical frames (similar to neighbors)
            scene_frames.sort(key=lambda x: x['similarity'], reverse=True)
            score_label = 'typicality'

        # Select top N frames and sort by timestamp for readability
        num_frames = min(frames_per_scene, len(scene_frames))
        optimal = scene_frames[:num_frames]
        optimal.sort(key=lambda x: x['timestamp'])

        debug_print(f"  Optimal frames ({score_label}):")
        for frame in optimal:
            debug_print(f"    - {frame['timestamp']:.2f}s (similarity: {frame['similarity']:.3f})")

        scenes.append({
            'scene_number': i + 1,
            'start': scene_start,
            'end': scene_end,
            'duration': scene_duration,
            'optimal_frames': optimal
        })

    return scenes

def process_video(
        video_path: str,
        sample_interval: float = 1.0,
        cut_threshold: float = 0.85,
        min_scene_length: float = 2.0,
        frames_per_scene: int = 3,
        optimal_frame_method: str = 'lowest_similarity') -> dict:
    """
    Complete video analysis: detect cuts and find optimal frames

    Args:
        video_path: Path to video file
        sample_interval: Seconds between sampled frames
        cut_threshold: Similarity threshold for detecting cuts
        min_scene_length: Minimum scene duration
        frames_per_scene: Number of optimal frames per scene
        optimal_frame_method: 'lowest_similarity' (distinct) or 'highest_similarity' (typical)

    Returns:
        Dictionary with complete analysis
    """
    duration = ffmpeg_helper.get_duration(video_path)

    # Single pass to get all similarities
    debug_print("=== Phase 1: Single Pass Analysis ===")
    similarities = analyze_video_single_pass(video_path, sample_interval)

    # Detect cuts from similarity data
    cuts = detect_cuts_from_similarities(similarities, cut_threshold, min_scene_length)

    # Find optimal frames within each scene
    scenes = find_optimal_frames_per_scene(
        similarities,
        cuts,
        duration,
        frames_per_scene,
        optimal_frame_method
    )

    return {
        'cuts': cuts,
        'scenes': scenes,
        'total_duration': duration,
        'num_scenes': len(scenes),
        'sample_interval': sample_interval,
        'cut_threshold': cut_threshold,
        'similarity_data': similarities  # Include raw data for further analysis
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple single-pass scene detection with optimal frame selection."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "-si", "--sample_interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for detecting cuts (default: 0.85)"
    )
    parser.add_argument(
        "-msl", "--min_scene_length",
        type=float,
        default=2.0,
        help="Minimum scene length in seconds (default: 2.0)"
    )
    parser.add_argument(
        "-of", "--optimal_frames",
        type=int,
        default=3,
        help="Number of optimal frames per scene (default: 3)"
    )
    parser.add_argument(
        "-om", "--optimal_method",
        type=str,
        choices=['lowest_similarity', 'highest_similarity'],
        default='lowest_similarity',
        help="Method for selecting optimal frames (default: lowest_similarity)"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode'
    )
    parser.add_argument(
        '--save-files',
        action='store_true',
        help='Keep temporary frame files'
    )
    parser.add_argument(
        '--save-raw-data',
        action='store_true',
        help='Include raw similarity data in output'
    )

    args = parser.parse_args()

    if args.debug:
        DEBUG = True
    if args.save_files:
        SAVE_FILES = True

    try:
        debug_print('Starting single-pass video analysis...\n')
        result = process_video(
            args.input,
            args.sample_interval,
            args.threshold,
            args.min_scene_length,
            args.optimal_frames,
            args.optimal_method
        )

        # Optionally remove raw similarity data for cleaner output
        if not args.save_raw_data:
            del result['similarity_data']

        debug_print('\n=== Results ===')
        print(json.dumps(result, indent=2))

    except KeyboardInterrupt:
        _exit_handler()
