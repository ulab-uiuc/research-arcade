"""
This program stores the commonly seen ML evaluation metrics.
"""

from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore


def bleu_score(generated_answers, correct_answers, n_loop):
    ans = []
    bleu_scorers = [BLEUScore(n_gram=i) for i in range(1, n_loop+1, 1)]

    for scorer in bleu_scorers:
        score = scorer(generated_answers, correct_answers)
        ans.append(score)
    return ans

def rouge_score(generated_answers, correct_answers):
    rouge_scorer = ROUGEScore()
    rouge_scores = rouge_scorer(generated_answers, correct_answers)

    return rouge_scores