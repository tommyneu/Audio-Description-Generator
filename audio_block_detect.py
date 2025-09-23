""" Module for detecting audio blocks """
import whisper
import ffmpeg_helper

def get_audio_blocks(audio_input:str) -> list:
    """ Takes in a audio file path and returns  """
    output = []

    total_duration = ffmpeg_helper.get_duration(audio_input)

    model = whisper.load_model("base")
    result = model.transcribe(audio_input, fp16=False)

    filled_gaps_results = _fill_gaps(result["segments"], total_duration)
    merged_segments = _merge_short_segments(filled_gaps_results)

    for i, segment in enumerate(merged_segments):
        output.append({
            'scene_number': i,
            'start_timecode': _format_seconds(segment["start"]),
            'end_timecode': _format_seconds(segment["end"]),
        })

    return output

def _format_seconds(seconds: float) -> str:
    """Convert seconds to hh:mm:ss.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def _fill_gaps(segments, total_dur):
    """ Fills in gaps in whisper segments """
    filled = []
    prev_end = 0.0

    for seg in segments:
        if seg["start"] > prev_end:  # gap
            filled.append({
                "start": prev_end,
                "end": seg["start"],
                "text": ""  # silence
            })
        filled.append(seg)
        prev_end = seg["end"]

    # Handle trailing silence
    if prev_end < total_dur:
        filled.append({
            "start": prev_end,
            "end": total_dur,
            "text": ""  # silence
        })

    return filled

def _merge_short_segments(segments, min_length=5.0):
    """
    Merge segments shorter than `min_length` seconds into neighbors.

    segments: list of dicts with {"start": float, "end": float, "text": str}
    Returns: new list of merged segments
    """
    if not segments:
        return []

    merged = []
    buffer = segments[0]

    for seg in segments[1:]:
        seg_length = buffer["end"] - buffer["start"]

        if seg_length < min_length:
            # Merge buffer into this segment
            buffer = {
                "start": buffer["start"],
                "end": seg["end"],
                "text": (buffer["text"] + " " + seg["text"]).strip()
            }
        else:
            merged.append(buffer)
            buffer = seg

    merged.append(buffer)
    return merged
