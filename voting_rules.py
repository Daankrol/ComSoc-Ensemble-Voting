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


def STV(rankings):
    result = []
    num_examples = len(rankings[0])
    num_classifiers = len(rankings)
    num_classes = len(rankings[0][0])
    for exp in range(num_examples):
        print('exp: ', exp)
        rankings_modified = rankings
        winner = [0] * num_classes
        for i in range(len(winner)):
            winner[i] = i
        losers = []
        while len(winner) > 1:
            score = [0] * num_classes
            for classifier in range(num_classifiers):
                score[rankings_modified[classifier][exp][0]] += 1
            loser_score = num_classifiers + 1
            loser = 0
            for idx in range(num_classes):
                if score[idx] < loser_score:
                    if idx not in losers:
                        loser_score = score[idx]
                        loser = idx
#           remove loser from winner, add to losers
            losers.append(loser)
            winner.remove(loser)
#           Update the rankings of the classifiers (of niet want ik faal)
            idx = np.where(rankings_modified[classifier][exp] == loser)
            np.delete(rankings_modified[classifier][exp], idx[0][0])
            print('rankings_modified: (of toch niet) ', rankings_modified[classifier][exp])
        result.append(winner[0])
    return index_to_label(result)

