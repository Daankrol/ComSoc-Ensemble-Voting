import numpy as np


def index_to_label(indices: list):
    return [chr(ord('A') + x) for x in indices]


def plurality(rankings):
    # rankings[classifier][test_example][ranked idx]
    result = []
    num_examples = len(rankings[0])
    num_classifiers = len(rankings)
    for exp in range(num_examples):
        histo = [0] * 26
        for classifier in range(num_classifiers):
            # top_vote is the idx of the highest
            top_vote = rankings[classifier][exp][0]
            histo[top_vote] += 1

        # gets index of letter with most votes
        # In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        winner = np.argmax(histo)
        result.append(winner)
    return index_to_label(result)
