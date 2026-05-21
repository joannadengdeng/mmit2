"""Official VQA v2-style metric helpers for the initial TextVQA release."""
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
_MANUAL_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PUNCT_TO_PROCESS = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]
_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def validate_metric(raw_metric: str) -> str:
    if raw_metric != "vqa_accuracy":
        raise ValueError("eval.metric must be exactly 'vqa_accuracy'.")
    return raw_metric


def normalize_answer(answer: str) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()

    processed = answer
    for punct in _PUNCT_TO_PROCESS:
        if (
            f"{punct} " in answer
            or f" {punct}" in answer
            or _COMMA_STRIP.search(answer) is not None
        ):
            processed = processed.replace(punct, "")
        else:
            processed = processed.replace(punct, " ")
    processed = _PERIOD_STRIP.sub("", processed, re.UNICODE)

    normalized_words = []
    for word in processed.lower().split():
        word = _MANUAL_NUMBER_MAP.get(word, word)
        if word not in _ARTICLES:
            normalized_words.append(word)

    for idx, word in enumerate(normalized_words):
        if word in _CONTRACTIONS:
            normalized_words[idx] = _CONTRACTIONS[word]

    return " ".join(normalized_words)


def coerce_ground_truths(ground_truth: object) -> List[str]:
    if ground_truth is None or ground_truth == "":
        return []
    if isinstance(ground_truth, list):
        return [str(gt) for gt in ground_truth if str(gt).strip()]
    return [str(ground_truth)]


def vqa_accuracy(
    prediction: str,
    ground_truths: List[str],
) -> float:
    norm_pred = normalize_answer(prediction)
    norm_ground_truths = [normalize_answer(gt) for gt in ground_truths]
    if not norm_ground_truths:
        return 0.0

    scores = []
    for idx, _ in enumerate(norm_ground_truths):
        matches = sum(
            1
            for other_idx, ground_truth in enumerate(norm_ground_truths)
            if other_idx != idx and ground_truth == norm_pred
        )
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def score_textvqa_prediction(
    prediction: str,
    ground_truth: object,
) -> Dict[str, float]:
    ground_truths = coerce_ground_truths(ground_truth)
    if not ground_truths:
        return {}
    return {"vqa_accuracy": vqa_accuracy(prediction, ground_truths)}


__all__ = [
    "coerce_ground_truths",
    "normalize_answer",
    "score_textvqa_prediction",
    "validate_metric",
    "vqa_accuracy",
]
