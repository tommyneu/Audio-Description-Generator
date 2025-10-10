""" Module for describing a video scene based on a list of images """
import argparse
import ollama
import numpy as np

def generate_description(images: list, prompt:str, model:str = "gemma3:12b", retries: int = 0) -> str:
    """Takes in an image path and returns a description of that image"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': images  # this attaches the image
            }]
        )

        # Extract the text from the response
        description = response['message']['content'].strip()
        return description or ''

    # pylint: disable=broad-exception-caught
    except Exception:
        if retries < 3:
            return generate_description(images, prompt, model, retries + 1)
        return ''

def semantic_similarity(text1: str, text2: str) -> float:
    """ Using Ollama embeds returns a score for how similar two strings are """
    # Get embeddings from Ollama
    e1 = ollama.embeddings(model="nomic-embed-text", prompt=text1)["embedding"]
    e2 = ollama.embeddings(model="nomic-embed-text", prompt=text2)["embedding"]

    vec1 = np.array(e1)
    vec2 = np.array(e2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect audio blocks from audio input')
    parser.add_argument('-f',
                        '--function',
                        required=True,
                        help='Function you wish to run')
    parser.add_argument('-i',
                        '--images',
                        nargs='*',
                        help='image file path')
    parser.add_argument('-p',
                        '--prompt',
                        default='none',
                        help='Prompt for model')
    parser.add_argument('-m',
                        '--model',
                        default='gemma3:12b',
                        help='Model to run')
    parser.add_argument('-s1',
                        '--string_1',
                        default='none',
                        help='String to compare')
    parser.add_argument('-s2',
                        '--string_2',
                        default='none',
                        help='String to compare')
    args = parser.parse_args()

    if args.function == 'generate_description':
        if args.images is None:
            raise ValueError('At least one image is needed')
        if args.prompt is None:
            raise ValueError('Prompt is needed')
        print(generate_description(args.images, args.prompt, args.model))

    elif args.function == 'semantic_similarity':
        if args.string_1 is None:
            raise ValueError('Missing string 1')
        if args.string_2 is None:
            raise ValueError('Missing string 2')

        print(semantic_similarity(args.string_1, args.string_2))
