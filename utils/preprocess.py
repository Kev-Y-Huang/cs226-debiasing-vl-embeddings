import json

import numpy as np
import requests
from tqdm import tqdm


def project_onto_subspace(wv, w2i, y):
    """
    Projects the output embeddings onto the gender subspace.

    Args:
        wv (np.ndarray): Word embeddings
        w2i (dict): Word to index mapping
        y (np.ndarray): The output embeddings

    Returns:
        z (np.ndarray): Projection of output embeddings onto gender subspace.
        gender_vector (np.ndarray): Vectors spanning the gender subspace.
    """
    # Calculate "gender direction"
    # Female-Male word pairs
    pairs = [
        ("woman", "man"),
        ("her", "his"),
        ("she", "he"),
        ("aunt", "uncle"),
        ("niece", "nephew"),
        ("daughters", "sons"),
        ("mother", "father"),
        ("daughter", "son"),
        ("granddaughter", "grandson"),
        ("girl", "boy"),
        ("stepdaughter", "stepson"),
        ("mom", "dad"),
    ]
    # Calculate the difference between female and male word embeddings
    diff = np.array([wv[w2i[wf]] - wv[w2i[wm]] for wf, wm in pairs])

    # Bias subspace is defined by top principal component of the differences
    cov = np.cov(np.array(diff).T)
    evals, evecs = np.linalg.eig(cov)
    dir = np.real(evecs[:, np.argmax(evals)])
    gender_vector = dir / np.linalg.norm(dir)

    # Get projection of output embeddings onto gender subspace
    z = np.array([np.dot(y_, gender_vector) for y_ in y])
    return z, gender_vector


def transform_analogies(wv, w2i, analogies):
    """
    Transforms the analogy data into input and output data.

    Args:
        wv (np.ndarray): Word embeddings
        w2i (dict): Word to index mapping
        analogies (list): List of analogies

    Returns:
        X: Input data as embeddings of the first three words in each analogy
        y: Output data as embeddings of the fourth word in each analogy 
    """
    X = []
    y = []
    for analogy in tqdm(analogies):
        try:
            # First three words are input, last word is output
            x = []
            for word in analogy[:-1]:
                x.append(wv[w2i[word.lower()]])
            X.append(np.stack(x))
            y.append(wv[w2i[analogy[-1].lower()]])
        except:
            # Skip adding analogy any of the words are not in the vocabulary
            pass
    X = np.stack(X)
    y = np.stack(y)

    return X, y


def preprocess():
    """
    Preprocesses the analogy data to produce input, output, and protected attribute data.

    Returns:
        analogies: List of analogies
    """
    # Load analogy data
    url = "http://download.tensorflow.org/data/questions-words.txt"
    # Family category includes gender dynamics
    category = "family"
    r = requests.get(url, allow_redirects=False)
    lines = r.text.split("\n")
    gender_pairs = set()
    valid_category = False
    for line in lines:
        sp = line.split(" ")
        # Start of category will be preceded by the line ": family"
        if not valid_category:
            valid_category = sp[1] == category
        else:
            # Each analogy will be formatted as "a b c d" where a:b::c:d
            if len(sp) == 4:
                gender_pairs.add((sp[0], sp[1]))
                gender_pairs.add((sp[2], sp[3]))
            # End of category will be a new category declaration ": category"
            else:
                break

    path = "data/lists/equalize_pairs.json"
    with open(path, "r") as f:
        equalize_pairs = json.load(f)
        for pair in equalize_pairs:
            gender_pairs.add((pair[0], pair[1]))

    gender_pairs = list(gender_pairs)
    print(f"{len(gender_pairs)} total pairs")

    analogies = []
    for i, pair1 in enumerate(gender_pairs):
        for pair2 in gender_pairs[:i] + gender_pairs[i + 1 :]:
            analogies.append(pair1 + pair2)

    print(f"{len(analogies)} analogies!")

    return analogies


def preprocess_v2(categories=[]):
    """
    Preprocesses the analogy data to produce input, output, and protected attribute data.
    More closely aligns with original Zhang et al. (2018) implementation.

    Returns:
        analogies: List of analogies
    """
    # Load analogy data
    url = "http://download.tensorflow.org/data/questions-words.txt"
    r = requests.get(url, allow_redirects=False)
    lines = r.text.split("\n")
    analogies = []
    valid_category = False
    for line in lines:
        sp = line.split(" ")
        # Start of category will be preceded by the line ": category"
        if len(sp) == 2:
            valid_category = not categories or sp[1] in categories
        elif valid_category:
            analogies.append(sp)

    print(f"{len(analogies)} analogies!")

    return analogies
