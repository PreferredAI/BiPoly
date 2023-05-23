import itertools
import random
from random import sample
from typing import Collection

import numpy as np
from entity.candidate import Candidate
from helper import logger
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _get_group_titles(
    candidate_pair: Collection[Candidate],
    recursive: bool = False,
    add_node_name: bool = True,
    full_node_name: bool = True,
    p_sample: float = 1.0,
    delimiter: str = ":",
    seed: int = None,
):
    group_titles = []
    for candidate in candidate_pair:
        logger.debug(
            "{} candidate has {} attributes".format(
                candidate.name, len(candidate._attributes)
            )
        )

        if recursive:
            get_keys = candidate.get_recursive_attributes_keys
        else:
            get_keys = candidate.get_attributes_keys

        titles = []
        for node_name in candidate.G:
            cat_titles = [
                candidate._stemmed_attributes[x]
                for x in set(list(get_keys(node_name)))
            ]

            if p_sample < 1 and len(cat_titles) > 0:
                samples = int(len(cat_titles) * p_sample)
                if samples == 0:
                    samples = 1
                cat_titles.sort()
                if seed is not None:
                    random.seed(str(int(seed)) + node_name)
                cat_titles = sample(cat_titles, samples)

            if add_node_name:
                if full_node_name or delimiter not in node_name:
                    cat_titles.append(node_name)
                else:
                    last_name = node_name.rsplit(delimiter, 1)[1].strip()
                    cat_titles.append(last_name)

            titles.append(" ".join(cat_titles))
        group_titles.append(titles)

    return group_titles


def _assert_max_density(mat):
    density = mat.getnnz() / np.prod(mat.shape)

    if density > 0.34:
        return mat.todense()

    return mat.tocsr()


def intersect(
    candidate_pair: Collection[Candidate],
    truth_pairs: set,
    recursive: bool = False,
):
    logger.info("Calculating using ground truth...")
    candidate: Candidate
    similarity = np.zeros(
        (len(candidate_pair[0].G), len(candidate_pair[1].G)), dtype=int,
    )

    for x, x_node in enumerate(candidate_pair[0].G):
        for y, y_node in enumerate(candidate_pair[1].G):
            if recursive:
                a = candidate_pair[0].get_recursive_attributes_keys(x_node)
                b = candidate_pair[1].get_recursive_attributes_keys(y_node)
            else:
                a = candidate_pair[0].get_attributes_keys(x_node)
                b = candidate_pair[1].get_attributes_keys(y_node)
            intersects = 0
            for pair in itertools.product(a, b):
                if pair in truth_pairs:
                    intersects += 1

            similarity[x][y] = intersects

    logger.debug(similarity.shape)
    logger.info("Intersect similarity calculated.")

    return similarity, "intersect"


def random_similarity(candidate_pair: Collection[Candidate]):
    logger.info("Using random...")
    candidate: Candidate
    similarity = np.random.rand(
        len(candidate_pair[0].G), len(candidate_pair[1].G)
    )

    logger.debug(similarity.shape)
    logger.info("Random similarity generated.")

    return similarity, "random"


def similarity_entity_idf(
    candidate_pair: Collection[Candidate],
    recursive: bool = False,
    add_node_name: bool = True,
    full_node_name: bool = True,
    p_sample: float = 1.0,
    seed: int = None,
    vocabulary: dict = None,
    norm="l2",
):
    logger.info(
        "Calculating TF/IDF similarity (norm={}, idf=entity)...".format(norm)
    )
    candidate: Candidate
    group_titles = _get_group_titles(
        candidate_pair,
        recursive=recursive,
        add_node_name=add_node_name,
        full_node_name=full_node_name,
        p_sample=p_sample,
        seed=seed,
    )

    vectorizer = TfidfVectorizer(vocabulary=vocabulary, norm=norm)
    vocab = []
    vocab.extend(candidate_pair[0]._stemmed_attributes.values())
    vocab.extend(candidate_pair[1]._stemmed_attributes.values())
    vocab.extend(candidate_pair[0].G)
    vocab.extend(candidate_pair[1].G)
    vectorizer.fit(vocab)

    tfidf = vectorizer.transform(group_titles[0] + group_titles[1])
    logger.debug(tfidf.shape)

    split = len(candidate_pair[0].G)
    similarity = linear_kernel(
        tfidf[:split], tfidf[split:], dense_output=False
    )
    logger.debug(similarity.shape)
    logger.info(
        "TF/IDF similarity (norm={}, idf=entity) calculated.".format(norm)
    )

    return _assert_max_density(similarity), "{}_entity".format(norm)


def similarity_category_idf(
    candidate_pair: Collection[Candidate],
    recursive: bool = False,
    add_node_name: bool = True,
    full_node_name: bool = True,
    p_sample: float = 1.0,
    seed: int = None,
    vocabulary: dict = None,
    norm="l2",
):
    logger.info(
        "Calculating TF/IDF similarity (norm={}, idf=category)...".format(norm)
    )
    candidate: Candidate
    group_titles = _get_group_titles(
        candidate_pair,
        recursive=recursive,
        add_node_name=add_node_name,
        full_node_name=full_node_name,
        p_sample=p_sample,
        seed=seed,
    )

    vectorizer = TfidfVectorizer(vocabulary=vocabulary, norm=norm)
    vocab = []
    vocab.extend(group_titles[0])
    vocab.extend(group_titles[1])
    vectorizer.fit(vocab)

    tfidf = vectorizer.transform(group_titles[0] + group_titles[1])
    logger.debug(tfidf.shape)

    split = len(candidate_pair[0].G)
    similarity = linear_kernel(
        tfidf[:split], tfidf[split:], dense_output=False
    )
    logger.debug(similarity.shape)
    logger.info(
        "TF/IDF similarity (norm={}, idf=category) calculated.".format(norm)
    )

    return _assert_max_density(similarity), "{}_category".format(norm)


def similarity_category_idf_size(
    candidate_pair: Collection[Candidate],
    recursive: bool = False,
    add_node_name: bool = True,
    full_node_name: bool = True,
    p_sample: float = 1.0,
    seed: int = None,
    vocabulary: dict = None,
    norm="l2",
):
    logger.info(
        "Calculating TF/IDF similarity (norm={}, idf=category) with \
            size...".format(
            norm
        )
    )
    candidate: Candidate
    group_titles = _get_group_titles(
        candidate_pair,
        recursive=recursive,
        add_node_name=add_node_name,
        full_node_name=full_node_name,
        p_sample=p_sample,
        seed=seed,
    )

    vectorizer = TfidfVectorizer(vocabulary=vocabulary, norm=norm)
    vocab = []
    vocab.extend(group_titles[0])
    vocab.extend(group_titles[1])
    vectorizer.fit(vocab)

    tfidf = vectorizer.transform(group_titles[0] + group_titles[1])
    logger.debug(tfidf.shape)

    split = len(candidate_pair[0].G)
    similarity = linear_kernel(
        tfidf[:split], tfidf[split:], dense_output=False
    )
    logger.debug(similarity.shape)

    size = np.empty(similarity.shape, dtype=int)
    for (xi, x), (yi, y) in itertools.product(
        enumerate(candidate_pair[0].G), enumerate(candidate_pair[1].G)
    ):
        size[xi, yi] = min(
            len(candidate_pair[0].get_attributes_keys(x)),
            len(candidate_pair[1].get_attributes_keys(y)),
        )

    similarity = similarity.multiply(np.log(size + 1))

    logger.info(
        "TF/IDF similarity (norm={}, idf=category) with size \
            calculated.".format(
            norm
        )
    )

    return _assert_max_density(similarity), "{}_category_size".format(norm)


def similarity_overlap(
    candidate_pair: Collection[Candidate],
    recursive: bool = False,
    add_node_name: bool = True,
    full_node_name: bool = True,
    p_sample: float = 1.0,
    seed: int = None,
):
    logger.info("Calculating overlap similarity...")

    candidate: Candidate
    group_titles = _get_group_titles(
        candidate_pair,
        recursive=recursive,
        add_node_name=add_node_name,
        full_node_name=full_node_name,
        p_sample=p_sample,
        seed=seed,
    )

    group_titles_set_list = []
    for candidate_group in group_titles:
        titles_set_list = []
        for titles in candidate_group:
            titles_set_list.append(set(titles.split()))
        group_titles_set_list.append(titles_set_list)

    row = []
    col = []
    data = []

    for (r, x), (c, y) in itertools.product(
        enumerate(group_titles_set_list[0]),
        enumerate(group_titles_set_list[1]),
    ):
        intersect = len(x & y)
        if intersect:
            row.append(r)
            col.append(c)
            data.append(intersect / min(len(x), len(y)))

    similarity = coo_matrix(
        (data, (row, col)),
        shape=(len(group_titles_set_list[0]), len(group_titles_set_list[1])),
    )
    similarity = similarity.tocsr()

    logger.info("Overlap similarity calculated.")

    return _assert_max_density(similarity), "overlap"
