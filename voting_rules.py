import numpy as np
import math


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
    # print('Final result plurality: ', result)
    return index_to_label(result)


def positional_scoring(rankings, weights=None):
    num_classifiers = len(rankings)
    num_examples = len(rankings[0])
    num_classes = len(rankings[0][0])
    result = []
    if weights is None:
        weights = []
        for i in range(num_classes):
            weights.append(num_classes - (i + 1))
    if len(weights) != 26:
        raise Exception(
            f'Weight vector has length {len(weights)} but should be {num_classes}')
    for exp in range(num_examples):
        scores = [0] * 26
        for classifier in range(num_classifiers):
            for alt in range(26):
                current_alternative = rankings[classifier][exp][alt]
                scores[current_alternative] += weights[alt]
        max_vote = np.argmax(scores)
        result.append(max_vote)
    return index_to_label(result)


def STV(rankings):
    num_classifiers = len(rankings)
    num_examples = len(rankings[0])
    num_classes = len(rankings[0][0])
    result = []
    for exp in range(num_examples):
        rankings_modified = np.copy(rankings)
        winner = [0] * num_classes
        for i in range(len(winner)):
            winner[i] = i
        losers = []
        # remove classes from winner array until only winner remains
        while len(winner) > 1:
            score = [0] * num_classes
            # add a point for the top choice of every classifier
            for classifier in range(num_classifiers):
                elected = -1
                for i in range(num_classes):
                    if rankings_modified[classifier][exp][i] != -1:
                        elected = rankings_modified[classifier][exp][i]
                        break
                score[elected] += 1
            loser_score = num_classifiers + 1
            loser = 0
            # find class with fewest votes, mark as loser
            for idx in range(num_classes):
                if score[idx] <= loser_score:
                    if idx not in losers:
                        loser_score = score[idx]
                        loser = idx
#           remove loser from winner, add to losers
            losers.append(loser)
            winner.remove(loser)
#           Update the rankings of the classifiers
            for classifier in range(num_classifiers):
                idx = np.where(rankings_modified[classifier][exp] == loser)
                rankings_modified[classifier][exp][idx[0][0]] = -1
        result.append(winner[0])
    # print('Final result SVT: ', result)
    return index_to_label(result)
