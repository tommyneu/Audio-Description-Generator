# Audio Description Project

1. Install pyenv for python 3.11
    1. Run `brew install pyenv`
    2. Add these commands to your `~/.zshrc` or `~/bashrc`

        ```BASH
            export PYENV_ROOT="$HOME/.pyenv"
            [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init - zsh)"
        ```

    3. Restart your terminal and run `pyenv install 3.11`
2. Install FFMPEG and Ollama and espeak
    - Run `brew install ffmpeg` for video editing
    - Run `brew install espeak` for some AI Text-To-Speech
    - Install Ollama from their site [https://ollama.com/download](https://ollama.com/download)
3. Run `ollama pull gemma3:12b` and `ollama pull nomic-embed-text`
4. Run `python3 -m venv ./venv` && `source ./venv/bin/activate`
    - Every time you start a new terminal for the project run `source ./venv/bin/activate`
    - [Python Environments](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-python-envs) will run that for you
5. Run `pip3 install -r requirements.txt`
6. Run `python3 describe_video.py --input ./my_video.mp4 --output ./my_video_audio_description.mp4`

## Pylint

To run pylint, run`pylint describe_video.py`. Pylint configuration is located in `./pyproject.toml`
