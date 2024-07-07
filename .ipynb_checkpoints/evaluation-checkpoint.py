import numpy as np
import dashscope
from dotenv import dotenv_values

config = dotenv_values('.env')
dashscope.api_key = config['qwen_key']


def get_embedding(embed_text):
    respond = ''
    try:
        respond = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v2,
            input=embed_text)
        embedding = respond.output['embeddings'][0]['embedding']
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
    return embedding


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    :param vec1: First vector (numpy array).
    :param vec2: Second vector (numpy array).
    :return: Cosine similarity score (float).
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity