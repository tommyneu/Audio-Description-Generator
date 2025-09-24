""" Module for describing a video scene based on a list of images """
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

def _semantic_similarity(text1: str, text2: str) -> float:
    """ Using Ollama embeds returns a score for how similar two strings are """
    # Get embeddings from Ollama
    e1 = ollama.embeddings(model="nomic-embed-text", prompt=text1)["embedding"]
    e2 = ollama.embeddings(model="nomic-embed-text", prompt=text2)["embedding"]

    vec1 = np.array(e1)
    vec2 = np.array(e2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def should_skip_description(prev: str, curr: str, similary_threshold:str = 0.75) -> bool:
    """ Checks to see if two strings are too similar """
    score = _semantic_similarity(prev, curr)
    print(f"Similarity: {score:.3f}")
    return score >= similary_threshold
