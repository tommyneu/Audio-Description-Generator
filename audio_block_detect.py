""" Module for detecting audio blocks """
import argparse
import whisper
import ffmpeg_helper

def get_audio_blocks(audio_input: str, min_pause: float = 0.6):
    """
    Returns both speech and pause blocks based on word-level timestamps.
    Uses Whisper's own segmentation (not amplitude-based silence).
    """

    total_duration = ffmpeg_helper.get_duration(audio_input)

    model = whisper.load_model("base")
    result = model.transcribe(
        audio_input,
        fp16=False,
        word_timestamps=True,
        condition_on_previous_text=False
    )

    # Flatten all words from all segments
    words = []
    for seg in result["segments"]:
        if "words" in seg:
            words.extend(seg["words"])

    if not words:
        return []

    blocks = []
    block_count = 0
    last_end = 0.0

    text_buffer = ''
    text_buffer_start = None

    # Iterate through all recognized words
    for index, single_word in enumerate(words):
        start = float(single_word["start"])
        gap = start - last_end
        last_end = float(single_word["end"])

        # We found a big enough gap in the speech so we should save everything before this as a block
        if gap > min_pause:
            if text_buffer_start is None:
                blocks.append({
                    'scene_number': block_count,
                    'text' : text_buffer.strip(),
                    'start_timecode': ffmpeg_helper.seconds_to_timecode(0.0),
                    'end_timecode': ffmpeg_helper.seconds_to_timecode(float(single_word["end"]))
                })
            else:
                blocks.append({
                    'scene_number': block_count,
                    'text' : text_buffer.strip(),
                    'start_timecode': ffmpeg_helper.seconds_to_timecode(text_buffer_start),
                    'end_timecode': ffmpeg_helper.seconds_to_timecode(float(single_word["end"]))
                })
                text_buffer_start = None
            text_buffer = ''
            block_count += 1

        # We are on the last word so we need to add this text buffer to the blocks
        elif index == len(words) - 1:
            # We need to get that last word
            text_buffer += single_word['word']
            blocks.append({
                'scene_number': block_count,
                'text' : text_buffer,
                'start_timecode': ffmpeg_helper.seconds_to_timecode(text_buffer_start),
                'end_timecode': ffmpeg_helper.seconds_to_timecode(float(single_word["end"]))
            })
            text_buffer = ''
            # We still increment in case we have extra space at the end of the video
            block_count += 1

        # No gap so we can extend the text buffer
        else:
            if text_buffer_start is None:
                text_buffer_start = start

        # We are looking to see if the previous word and this word has a gap
        # So we are always adding this word to the next block
        text_buffer += single_word['word']

    # We need to fill in the remaining time
    if last_end < total_duration:
        blocks.append({
            'scene_number': block_count,
            'text' : '',
            'start_timecode': ffmpeg_helper.seconds_to_timecode(last_end),
            'end_timecode': ffmpeg_helper.seconds_to_timecode(total_duration)
        })

    return blocks

# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect audio blocks from audio input')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to input audio file')
    parser.add_argument('-ml',
                        '--min_length',
                        default='0.6',
                        help='Minimum length for audio block')
    args = parser.parse_args()

    print(get_audio_blocks(args.input, float(args.min_length)))
