import string
from typing import Collection, Union

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def tokenize(
    text: str,
    stop_words=set(stopwords.words("english")),
    punctuation=string.punctuation,
):
    text = text.replace("_", " ")
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        w for w in tokens if w not in stop_words and w not in punctuation
    ]

    return filtered_tokens


def stem(texts: Union[str, Collection[str]]):
    if isinstance(texts, str):
        texts = [texts]

    result = []
    for text in texts:
        filtered_tokens = tokenize(text)

        stems = []
        for item in filtered_tokens:
            # stems.append(PorterStemmer().stem(item))
            stems.append(item)

        result.append(" ".join(stems))

    return result
