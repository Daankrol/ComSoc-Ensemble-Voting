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
    return index_to_label(result)


def STV(rankings):
    # rankings[classifier][test_example][ranked idx]
    num_examples = len(rankings[0])
    num_classifiers = len(rankings)
    num_classes = len(rankings[0][0])
    result = []
    for exp in range(num_examples):
        losers = []
        options = [x for x in range(26)]
        while len(losers) < 25:
            histo = [0] * 26
            for classifier in range(num_classifiers):
                top_vote_idx = 0 
                while (rankings[classifier][top_vote_idx] in losers):
                    # Set loser's histogram score to arbitrary high
                    histo[top_vote_idx]  = num_classifiers * 1000
                    top_vote_idx += 1
                histo[top_vote_idx] += 1 
                

            loser  = [x for x in histo if histo[x] == min(histo) ]
            #  np.where(histo == np.min(histo))
            # If there is a tie, none should be removed. 
            if len(loser) > 1:
                break
            # TODO: let best classifier (based on individual f1) 
            for x in loser: losers.append(x) 
    
        print(options)
        print(losers)
        options = set(options)
        losers = set(losers)
        print(type(options))
        print(type( options-  losers))
        exit(9)
        winners = list(set(options) - set(losers))
        # if len(winners) > 1:
            #TODO let the best classifier solve the tie
        winners = winners[0]
        result.append(winners)

    return index_to_label(result)

#         while len(winner) > 1:
#             score = [0] * num_classes
#             for classifier in range(num_classifiers):
#                 score[rankings[classifier][exp][0]] += 1
#             loser_score = num_classifiers + 1
#             loser = 0
#             for idx in range(num_classes):
#                 if score[idx] < loser_score:
#                     if idx not in losers:
#                         loser_score = score[idx]
#                         loser = idx
# #           remove loser from winner, add to losers
#             losers.append(loser)
#             winner.remove(loser)
# #           Update the rankings of the classifiers (of niet want ik faal)
#             idx = np.where(rankings[classifier][exp] == loser)
#             np.delete(rankings[classifier][exp], idx[0][0])
#             # print('rankings_modified: (of toch niet) ', rankings_modified[classifier][exp])
#         result.append(winner[0])
#     return index_to_label(result)

