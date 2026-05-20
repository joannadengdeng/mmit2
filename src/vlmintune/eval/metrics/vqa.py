"""Minimal VQA metrics for the initial TextVQA release.

References:
  - VQA v2: https://visualqa.org/evaluation.html
"""
from __future__ import annotations

import re
from typing import Dict, List


_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "were": "we're", "weve": "we've", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


# 1. Validate the configured metric.
def validate_metric(raw_metric: str) -> str:
    """Validate the explicit metric for the current release."""
    if raw_metric != "vqa_accuracy":
        raise ValueError("eval.metric must be exactly 'vqa_accuracy'.")
    return raw_metric


# 2. Normalize a single answer string before comparison.
def normalize_answer(answer: str) -> str:
    """Apply VQA v2 standard normalisation to an answer string."""
    answer = answer.lower()

    # Expand contractions
    words = answer.split()
    words = [_CONTRACTIONS.get(w, w) for w in words]
    answer = " ".join(words)

    # Remove punctuation
    answer = "".join(ch for ch in answer if ch not in _PUNCT)

    # Remove articles
    words = answer.split()
    words = [w for w in words if w not in _ARTICLES]
    answer = " ".join(words)

    # Normalise whitespace + trailing/leading spaces
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


# 3. Coerce the raw ground truth into answer strings.
def coerce_ground_truths(ground_truth: object) -> List[str]:
    """Convert raw ground truth into a non-empty list of answer strings."""
    if ground_truth is None or ground_truth == "":
        return []
    if isinstance(ground_truth, list):
        return [str(gt) for gt in ground_truth if str(gt).strip()]
    return [str(ground_truth)]


# 4. Compute VQA accuracy from the normalized prediction and ground truths.
def vqa_accuracy(
    prediction: str,
    ground_truths: List[str],
) -> float:
    """Compute soft VQA accuracy for a single prediction.

    VQA v2 metric: min(#humans_agreeing / 3, 1.0) for exact-match.
    ``ground_truths`` is the list of up to 10 human answers.
    """
    norm_pred = normalize_answer(prediction)
    count = sum(1 for gt in ground_truths if normalize_answer(gt) == norm_pred)
    return min(count / 3.0, 1.0)


def score_textvqa_prediction(
    prediction: str,
    ground_truth: object,
) -> Dict[str, float]:
    """Score one TextVQA prediction with the single supported metric."""
    ground_truths = coerce_ground_truths(ground_truth)
    if not ground_truths:
        return {}
    return {"vqa_accuracy": vqa_accuracy(prediction, ground_truths)}
