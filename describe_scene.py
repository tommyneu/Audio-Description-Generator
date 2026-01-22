""" Module for describing a video scene based on a list of images """
import argparse
import ollama
import numpy as np

def generate_description(images: list, model:str = "gemma3:12b", retries: int = 0) -> str:
    """Takes in an image path and returns a description of that image"""

    try:
        response = ollama.generate(
            model=model,
            prompt="""Analyze the provided sequence of images and generate the single, required audio description.""",
            images=images,
            system='''You are an expert Audio Description Generator. Your sole function is to create clear,
objective, and professionally formatted audio descriptions for visually impaired audiences. Analyze the
sequence of input images, which represent a single visual scene from a video.

**Strict Constraints:**
1.  **Do not** use any conversational language, greetings, or acknowledgments.
2.  **Do not** speculate, interpret, or describe sounds, music, or dialogue.
3.  **Do not** use pronouns (I, you, we, etc.).
4.  **Do not** describe the camera work (e.g., "The camera pans," "A close-up shows").
5.  The output must be a single, concise paragraph.
6.  Focus on identifying key visual elements: people, actions, locations, and essential on-screen text.
7.  Prioritize actions and changes across the image sequence.
8.  Maintain a neutral, objective tone.
9.  **Do not** use any text formatting or emojis.

**Task:** Synthesize the visual information from the input images into a single, cohesive,
action-oriented audio description that is ready to be voiced.'''
        )

        # Extract the text from the response
        description = response['response'].strip()
        return description or ''

    # pylint: disable=broad-exception-caught
    except Exception:
        if retries < 3:
            return generate_description(images, model, retries + 1)
        return ''

def semantic_similarity(text1: str, text2: str) -> float:
    """ Using Ollama embeds returns a score for how similar two strings are """
    # Get embeddings from Ollama
    e1 = ollama.embeddings(model="nomic-embed-text", prompt=text1)["embedding"]
    e2 = ollama.embeddings(model="nomic-embed-text", prompt=text2)["embedding"]

    vec1 = np.array(e1)
    vec2 = np.array(e2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def merge_scene_descriptions(descriptions_input: list[str], model:str = "gemma3:12b", retries:int=0)-> str:
    """ Merges multiple descriptions into one """

    prompt = 'Combine the following chronological scene descriptions into a single, cohesive block:'
    prompt += '\n'.join([f'- {desc}' for desc in descriptions_input])

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            system="""You are an expert Audio Description Editor. Your task is to combine two or more separate,
consecutive scene descriptions into a single, cohesive, finalized narration block. The output must be
ready to be spoken immediately.

**Input:** A sequential list of preliminary audio descriptions, each representing a single visual scene.

**Strict Constraints & Requirements:**
1.  **Do not** add any conversational language, introductory phrases, or closing remarks.
2.  **Do not** describe the transition between scenes (e.g., "The scene changes to..."). Simply blend the content.
3.  **Eliminate all redundancy.** If a person, object, or location is described in consecutive inputs, mention it only in the first scene description where it appears.
4.  **Prioritize Action and Change.** Maintain the focus on the most important actions and visual changes across the combined scenes.
5.  **Maintain Chronological Order.** The final description must accurately reflect the sequence of events as described in the input list.
6.  The final output must be a single, smooth, cohesive paragraph.
7.  Maintain the neutral, objective tone of the input descriptions.

**Task:** Edit and combine the provided sequence of scene descriptions into a single, flowing, ready-to-voice audio block.
"""
        )

        # Extract the text from the response
        description = response['response'].strip()
        return description or ''

    # pylint: disable=broad-exception-caught
    except Exception:
        if retries < 3:
            return generate_description(descriptions_input, model, retries + 1)
        return ''

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
        print(generate_description(args.images, args.model))

    elif args.function == 'semantic_similarity':
        if args.string_1 is None:
            raise ValueError('Missing string 1')
        if args.string_2 is None:
            raise ValueError('Missing string 2')

        print(semantic_similarity(args.string_1, args.string_2))
