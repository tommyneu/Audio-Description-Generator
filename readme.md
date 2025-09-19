# Audio Description Project

1. Install FFMPEG and Ollama
2. Run `ollama pull gemma3:12b` and `ollama pull nomic-embed-text`
3. Run `python3 -m venv ./venv`
4. Run `venv/bin/pip3 install -r requirements.txt`
5. Run `venv/bin/python3 describe_video.py --input ./my_video.mp4 --output ./my_video_audio_description.mp4`

## Pylint

To run pylint, run`pylint describe_video.py`. Pylint configuration is located in `./pyproject.toml`
