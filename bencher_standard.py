import functools
import itertools
import operator
import os
import pickle
import re
from argparse import ArgumentParser, FileType
from collections import defaultdict
from typing import Callable, Collection, Mapping

import dask
import networkx as nx
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed, wait
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

import config as cfg
import helper
from entity.candidate import Candidate
from entity.solution import Solution
from helper import logger
from module import candidatesimilarity as cs
from module import solving
from module.drawing import draw, draw_circular
from module.graphing import recursive_add_edge
from module.matching import Matcher


def get_max_stats(candidate_pair: Collection[Candidate], truth_pairs: set):
    truth_left = truth_pairs.copy()
    for node0 in candidate_pair[0].G:
        for node1 in candidate_pair[1].G:
            a = candidate_pair[0].get_attributes_keys(node0)
            b = candidate_pair[1].get_attributes_keys(node1)
            for pair in itertools.product(a, b):
                if pair in truth_left:
                    truth_left.remove(pair)

    matches = len(truth_pairs) - len(truth_left)

    comparisons = len(candidate_pair[0]._attributes) * len(
        candidate_pair[1]._attributes
    )

    return matches, comparisons


def print_indiv_stats(
    candidate_pair: Collection[Candidate],
    truth_pairs: set,
    suffix,
    recursive: bool,
):
    matches = defaultdict(int)
    comparisons = {}
    for node0 in candidate_pair[0].G:
        for node1 in candidate_pair[1].G:
            if recursive:
                a = list(
                    candidate_pair[0].get_recursive_attributes_keys(node0)
                )
                b = list(
                    candidate_pair[1].get_recursive_attributes_keys(node1)
                )
            else:
                a = candidate_pair[0].get_attributes_keys(node0)
                b = candidate_pair[1].get_attributes_keys(node1)
            len_a = len(a)
            len_b = len(b)
            comparisons[(node0, node1)] = len_a * len_b
            for pair in itertools.product(a, b):
                if pair in truth_pairs:
                    matches[(node0, node1)] += 1

    printable = []
    for key, m in matches.items():
        c = comparisons[key]
        printable.append(key + (m, c))

    pd.DataFrame(printable, columns=["x", "y", "matches", "compares"]).to_csv(
        os.path.join(
            cfg.OUTPUT_DIR, "matched_solution_indiv_{}.csv".format(suffix)
        )
    )


def get_pair_results_stbl(
    candidate_pair: Collection[Candidate],
    truth_pairs: set,
    node_pair: tuple,
    ret_size: bool = False,
):
    a = dict(candidate_pair[0].get_attributes(node_pair[0]))
    b = dict(candidate_pair[1].get_attributes(node_pair[1]))
    if len(a) == 0 or len(b) == 0:
        if ret_size:
            return set(), 0, 0, (len(a), len(b))
        return set(), 0, 0

    titles = (a, b)
    split = len(a)

    corpus = []
    keys = []
    for title_dict in titles:
        corpus.extend(title_dict.values())
        keys.extend(title_dict.keys())

    vectorizer = CountVectorizer(binary=True)
    m = vectorizer.fit_transform(corpus).tocsr()
    m.sort_indices()

    assert m.has_sorted_indices == 1

    m_arr_csr = (m[:split], m[split:])
    m_arr = tuple(x.tocsc() for x in m_arr_csr)
    key_tup = (keys[:split], keys[split:])

    comparisons = 0
    matches = set()
    false_matches = 0
    for i in range(len(vectorizer.vocabulary_)):
        indices = tuple(
            m_arr[mdx].indices[m_arr[mdx].indptr[i] : m_arr[mdx].indptr[i + 1]]
            for mdx in range(2)
        )

        if len(indices[0]) == 0 or len(indices[1]) == 0:
            continue

        for x, y in itertools.product(*indices):
            comparisons += 1
            pair = (key_tup[0][x], key_tup[1][y])
            if pair in truth_pairs:
                matches.add(pair)
            else:
                false_matches += 1

        logger.debug("Completed vocab {}".format(i))

    if ret_size:
        return matches, false_matches, comparisons, (len(a), len(b))
    return matches, false_matches, comparisons


def get_pair_results(
    candidate_pair: Collection[Candidate],
    truth_pairs: set,
    node_pair: tuple,
    recursive: bool,
    ret_size: bool = False,
):
    matches = set()
    false_matches = 0

    if recursive:
        a = list(candidate_pair[0].get_recursive_attributes_keys(node_pair[0]))
        b = list(candidate_pair[1].get_recursive_attributes_keys(node_pair[1]))
    else:
        a = candidate_pair[0].get_attributes_keys(node_pair[0])
        b = candidate_pair[1].get_attributes_keys(node_pair[1])

    len_a = len(a)
    len_b = len(b)
    comparisons = len_a * len_b

    for pair in itertools.product(a, b):
        if pair in truth_pairs:
            matches.add(pair)
        else:
            false_matches += 1

    if ret_size:
        return matches, false_matches, comparisons, (len_a, len_b)
    return matches, false_matches, comparisons


@dask.delayed
def get_solution_stats(
    candidate_pair: Collection[Candidate],
    truth_pair: set,
    raw_solution,
    recursive: bool,
):
    ret = dict()

    x_nodes = list(candidate_pair[0].G)
    y_nodes = list(candidate_pair[1].G)

    overall_matches = set()
    overall_false_matches = 0
    overall_comparisons = 0
    for x, y in raw_solution["x_route"]:
        node_pair = (x_nodes[x], y_nodes[y])
        matches, false_matches, comparisons = get_pair_results(
            candidate_pair, truth_pair, node_pair, recursive
        )
        overall_matches.update(matches)
        overall_false_matches += false_matches
        overall_comparisons += comparisons

    for y, x in raw_solution["y_route"]:
        node_pair = (x_nodes[x], y_nodes[y])
        matches, false_matches, comparisons = get_pair_results(
            candidate_pair, truth_pair, node_pair, recursive
        )
        overall_matches.update(matches)
        overall_false_matches += false_matches
        overall_comparisons += comparisons

    ret["tp"] = len(overall_matches)
    ret["fp"] = overall_false_matches
    ret["comparisons"] = overall_comparisons

    clusters = len(raw_solution["x_fac"]) + len(raw_solution["y_fac"])

    open_x = set()
    for x, y in raw_solution["x_route"]:
        open_x.add(x)

    open_y = set()
    for y, x in raw_solution["y_route"]:
        open_y.add(y)

    open_fac = len(open_y) + len(open_x)
    if open_fac:
        avg_size = (
            len(raw_solution["x_route"]) + len(raw_solution["y_route"])
        ) / open_fac
    else:
        avg_size = 0

    ret["clusters"] = clusters
    ret["avg_size"] = avg_size
    ret["elasped"] = raw_solution["elasped"]

    return ret


def solve(
    client: Client,
    matching_futures,
    solve_funcs: Collection[Callable],
    candidate_pair_scatters: Mapping,
    iterations: int = 1,
):

    if isinstance(solve_funcs, Callable):
        solve_funcs = [solve_funcs]

    solutions = []
    with tqdm(
        total=len(matching_futures), desc="Submit solve", dynamic_ncols=True
    ) as pbar:
        for future, (similarity, name) in as_completed(
            matching_futures, with_results=True
        ):
            sim_func = future.sim_func
            candidate_pair = future.candidate_pair
            candidate_pair_scatter = candidate_pair_scatters[candidate_pair]
            similarity = client.scatter((similarity,))[0]

            for solve_func in solve_funcs:
                d_solution = dask.delayed(solve_func)(
                    candidate_pair_scatter, similarity
                )
                last_candidate_pair = candidate_pair_scatter
                for r in range(iterations - 1):
                    last_candidate_pair = reorg(
                        last_candidate_pair, d_solution
                    )
                    new_similarity, _ = dask.delayed(sim_func, nout=2)(
                        last_candidate_pair
                    )
                    d_solution = dask.delayed(solve_func)(
                        last_candidate_pair,
                        new_similarity,
                        last_solution=d_solution,
                    )

                solution = Solution(candidate_pair, sim_func, solve_func)
                solution.solution_future = d_solution
                solution._final_candidate_pair = last_candidate_pair
                solutions.append(solution)
            pbar.update()

    return solutions


@dask.delayed
def reorg(
    candidate_pair, raw_solution,
):
    routes = ("x_route", "y_route")
    new_candidate_pair = []
    for i, candidate in enumerate(candidate_pair):
        logger.info("Start {}".format(len(candidate.G)))
        new_candidate = Candidate(candidate.name, candidate.G.copy())
        new_candidate._node_attributes_map = dict(
            candidate._node_attributes_map
        )
        new_candidate._node_recursive_attributes_map = (
            candidate._node_recursive_attributes_map
        )
        new_candidate._attributes = candidate._attributes
        new_candidate._stemmed_attributes = candidate._stemmed_attributes

        nodes = list(candidate.G)
        mergers = defaultdict(list)
        for a, b in raw_solution[routes[i]]:
            mergers[b].append(nodes[a])

        for nodes in mergers.values():
            if len(nodes) == 1:
                continue

            new_node = " + ".join(nodes)
            if new_node in new_candidate._node_attributes_map:
                continue

            new_candidate.G.add_node(new_node)
            new_candidate.G.remove_nodes_from(nodes)

            new_candidate._node_attributes_map[new_node] = set()
            for node in nodes:
                new_candidate._node_attributes_map[new_node].update(
                    new_candidate._node_attributes_map[node]
                )
                del new_candidate._node_attributes_map[node]

            logger.debug(len(new_candidate._node_attributes_map[new_node]))

        logger.debug("End {}".format(len(new_candidate.G)))
        new_candidate_pair.append(new_candidate)

    return new_candidate_pair


def export(
    candidate_pair: Collection[Candidate],
    raw_solution,
    name,
    func: callable = None,
):

    if func:
        suffix = "-" + func.__name__
        args = "-".join(("{}={}".format(*i) for i in func.keywords.items()))
        if args:
            suffix += "-" + args
    else:
        suffix = ""

    x_nodes = list(candidate_pair[0].G)
    y_nodes = list(candidate_pair[1].G)

    printable = []

    for x, y in raw_solution["x_route"]:
        printable.append((x_nodes[x], y_nodes[y]))

    for y, x in raw_solution["y_route"]:
        printable.append((x_nodes[x], y_nodes[y]))

    logger.info("Writing solution {} to CSV.".format(suffix))
    pd.DataFrame(printable, columns=["x", "y"]).to_csv(
        os.path.join(
            cfg.OUTPUT_DIR, "matched_benchmark_{}-{}.csv".format(name, suffix)
        )
    )


def export_similarity(
    candidate_pair, similarities, truth_pairs, suffix, recursive: bool
):
    printable = []
    for ((i, x), (j, y)) in itertools.product(
        enumerate(candidate_pair[0].G), enumerate(candidate_pair[1].G)
    ):
        if recursive:
            a = list(candidate_pair[0].get_recursive_attributes_keys(x))
            b = list(candidate_pair[1].get_recursive_attributes_keys(y))
        else:
            a = candidate_pair[0].get_attributes_keys(x)
            b = candidate_pair[1].get_attributes_keys(y)
        len_a = len(a)
        len_b = len(b)
        comparisons = len_a * len_b
        matches = 0
        for pair in itertools.product(a, b):
            if pair in truth_pairs:
                matches += 1

        printout = (x, y, matches, comparisons)
        for similarity, _ in similarities:
            printout += (similarity[i, j],)

        printable.append(printout)

    columns = ["x", "y", "matches", "compares"]
    for _, name in similarities:
        columns.append(name)

    pd.DataFrame(printable, columns=columns).to_csv(
        os.path.join(
            cfg.OUTPUT_DIR,
            "matched_benchmark_{}-similarities.csv".format(suffix),
        )
    )


def retrieve(
    candidates, dataframes, description: bool = False, fake: bool = False,
):
    matcher = Matcher()
    for i, (candidate, df) in enumerate(zip(candidates, dataframes)):
        logger.info("Adding candidate...")
        if fake:
            candidate._attr_loader = helper._testing_attr_loader
        else:
            logger.debug(
                "Generating listing dict for {}...".format(candidate.name)
            )
            grouped_listing = defaultdict(dict)
            for row in df.iterrows():
                title = str(row[1]["title"])
                if description:
                    title += " " + str(row[1]["description"])
                grouped_listing[row[1]["category"].strip()][
                    row[1]["id"]
                ] = title
            logger.debug(
                "Generating listing dict done {} done.".format(candidate.name)
            )

            def get_immediate_listings(node, node_name, listing_dict):
                return listing_dict[node_name]

            funct = functools.partial(
                get_immediate_listings, listing_dict=grouped_listing
            )

            candidate._attr_loader = funct
        matcher.add(candidate)

    return matcher


def get_candidates(dataframes):
    candidates = []
    for n, df in enumerate(dataframes):
        G = nx.DiGraph()

        for category in set(df["category"].tolist()):
            recursive_add_edge(G, category, "root", ": ")

        candidates.append(Candidate(str(n), G))

    return candidates


def wrapped_partial(func, name: str = None, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    if name:
        partial_func.__name__ = name

    return partial_func


def get_sim_funcs(truth_pairs: set, recursive, p_sample, seed):
    vocabulary = None
    sim_func = [
        wrapped_partial(
            cs.similarity_unique_idf,
            vocabulary=vocabulary,
            add_node_name=False,
            recursive=recursive,
            p_sample=p_sample,
            seed=seed,
        ),
        wrapped_partial(
            cs.similarity_category_idf,
            vocabulary=vocabulary,
            add_node_name=False,
            recursive=recursive,
            p_sample=p_sample,
            seed=seed,
        ),
        wrapped_partial(
            cs.similarity_overlap,
            add_node_name=False,
            recursive=recursive,
            p_sample=p_sample,
            seed=seed,
        ),
        wrapped_partial(
            cs.similarity_category_idf_size,
            vocabulary=vocabulary,
            add_node_name=False,
            recursive=recursive,
            p_sample=p_sample,
            seed=seed,
        ),
    ]

    return sim_func


def get_solve_funcs(args):
    solve_funcs = []

    for w in np.around(np.arange(-1, 1.05, 0.1), 2):
        for n in np.around(np.arange(-1, 1.05, 0.1), 2):
            solve_func = wrapped_partial(solving.solve_bipoly, w=w, n=n)
            solve_funcs.append(solve_func)

    solve_func = wrapped_partial(solving.solve_stable_marriage)
    solve_funcs.append(solve_func)

    solve_func = wrapped_partial(solving.solve_bipartite)
    solve_funcs.append(solve_func)

    return solve_funcs


def expand(candidate: Candidate, eps: float = 4, min_samples: int = 2):
    G: nx.DiGraph
    G = candidate.G
    for node in tqdm(list(G), desc="Node", dynamic_ncols=True):
        if node == "root":
            continue

        keys = list(candidate.get_attributes_keys(node))
        if not keys:
            continue

        corpus = [candidate._stemmed_attributes[k] for k in keys]
        vectorizer = CountVectorizer(binary=True)
        m = vectorizer.fit_transform(corpus)

        successors = list(G.successors(node))

        attributes_map = defaultdict(set)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(m)
        for n, label in enumerate(labels):
            attributes_map["{} {}".format(node, label)].add(keys[n])

        G.add_edges_from(
            [(node, new_node) for new_node in attributes_map.keys()]
        )

        # Create a new virtual link to old nodes
        if successors:
            new_node = "{} existing".format(node)
            attributes_map[new_node]
            G.add_edge(node, new_node)
            G.add_edges_from(
                [(new_node, successor) for successor in successors]
            )

        candidate._node_attributes_map[node] = set()

        candidate._node_attributes_map.update(attributes_map)


def cleanup(candidate: Candidate, rounds: int = 5):
    if rounds <= 0:
        return

    candidate: Candidate
    G = candidate.G

    documents = []
    for node in G:
        cat_titles = [
            candidate._stemmed_attributes[x]
            for x in candidate.get_attributes_keys(node)
        ]
        cat_titles.append(node)
        documents.append(" ".join(cat_titles))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    documents.extend(candidate._stemmed_attributes.values())
    tfidf = vectorizer.transform(documents)
    logger.debug(tfidf.shape)

    key_map = {
        key: i
        for i, key in enumerate(
            candidate._stemmed_attributes.keys(), start=len(G)
        )
    }
    node_map = {node: i for i, node in enumerate(G)}

    for r in tqdm(range(rounds), desc="Round", dynamic_ncols=True):
        transfers = defaultdict(lambda: defaultdict(list))
        for node in tqdm(G, desc="Node", dynamic_ncols=True):
            keys = list(candidate.get_attributes_keys(node))
            if not len(keys):
                continue
            keys_idx = [key_map[x] for x in keys]

            nodes = list(
                itertools.chain(G.predecessors(node), G.successors(node))
            )
            nodes.insert(0, node)
            nodes_idx = [node_map[x] for x in nodes]

            similarity = linear_kernel(tfidf[nodes_idx], tfidf[keys_idx])
            max_idx = np.argmax(similarity, axis=0)

            for k, i in enumerate(max_idx):
                if i > 0:
                    transfers[node][nodes[i]].append(keys[k])

        if len(transfers) == 0:
            break

        logger.debug("Transferring {} items...".format(len(transfers)))

        for out_node, ins in transfers.items():
            for in_node, keys in ins.items():
                candidate._node_attributes_map[in_node].update(keys)
                candidate._node_attributes_map[out_node].difference_update(
                    keys
                )


def preprocess(candidate: Candidate, eps, min_samples, rounds):
    cleanup(candidate, rounds=rounds)
    if min_samples:
        expand(candidate, eps, min_samples)


def main(args):
    truth_df = pd.read_csv(args.truth_file)
    logger.info("Truth collection: {}".format(truth_df.shape))
    truth_pairs = set(truth_df.itertuples(index=False, name=None))

    use_pickled = False
    if args.pickle:
        if os.path.isfile(
            os.path.join(
                cfg.OUTPUT_DIR, "matcher-{}.pickle".format(args.pickle)
            )
        ):
            logger.info("Using pickle")
            with open(
                os.path.join(
                    cfg.OUTPUT_DIR, "matcher-{}.pickle".format(args.pickle)
                ),
                "rb",
            ) as handle:
                matcher = pickle.load(handle)
                use_pickled = True

    if not use_pickled:
        adf = pd.read_csv(args.a_file)
        logger.info("Collection 1: {}".format(adf.shape))
        bdf = pd.read_csv(args.b_file)
        logger.info("Collection 2: {}".format(bdf.shape))
        dataframes = (adf, bdf)
        candidates = get_candidates(dataframes)
        matcher = retrieve(
            candidates,
            dataframes,
            description=args.description,
            fake=args.fake,
        )

    sim_funcs = get_sim_funcs(
        truth_pairs, args.recursive, args.p_sample, args.seed
    )
    solve_funcs = get_solve_funcs(args)

    cluster = helper.get_cluster()
    with Client(cluster) as client:
        helper.ensure_worker(client)

        if not use_pickled:
            matcher.prepare(client, batch_size=cfg.STEM_BATCH)
            logger.info(
                "Total nodes is {} and {}.".format(
                    len(matcher._candidates[0].G),
                    len(matcher._candidates[1].G),
                )
            )

            for candidate in tqdm(
                matcher._candidates, desc="preprocessing", dynamic_ncols=True
            ):
                preprocess(
                    candidate, args.eps / 10.0, args.min_samples, args.rounds
                )

            if args.pickle:
                logger.info("Creating pickle file...")
                with open(
                    os.path.join(
                        cfg.OUTPUT_DIR, "matcher-{}.pickle".format(args.pickle)
                    ),
                    "wb",
                ) as handle:
                    pickle.dump(
                        matcher, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
                logger.info("Created pickle file.")

        logger.info(
            "Total nodes is now {} and {}.".format(
                len(matcher._candidates[0].G), len(matcher._candidates[1].G)
            )
        )

        matching_futures = matcher.run(client, sim_funcs)

        candidate_pair_scatters = {}
        for future in matching_futures:
            candidate_pair = future.candidate_pair
            helper.broadcast_add_and_get(
                client, candidate_pair_scatters, candidate_pair
            )

        # Solve
        solution: Solution
        solutions = solve(
            client,
            matching_futures,
            solve_funcs,
            candidate_pair_scatters,
            args.iterations,
        )

        truth_pairs = client.scatter((truth_pairs,), broadcast=True)[0]

        # Submit max stats
        max_stats_future = []
        with tqdm(
            total=len(candidate_pair_scatters),
            desc="Submit max stats",
            dynamic_ncols=True,
        ) as pbar:
            for (
                candiate_pair,
                candidate_pair_scatter,
            ) in candidate_pair_scatters.items():
                future = client.submit(
                    get_max_stats, candidate_pair_scatter, truth_pairs,
                )
                future._candidate_pair = candiate_pair
                max_stats_future.append(future)
            pbar.update()

        # Draw
        draw_future = []
        if args.draw:
            logger.info("Drawing solutions")
            draws = []
            for solution in solutions:
                draws.append(
                    dask.delayed(draw_circular)(
                        solution._final_candidate_pair,
                        solution.solution_future,
                        func=solution.solve_func,
                        prefix="matched_benchmark_{}-draw".format(args.suffix),
                    )
                )
            draw_future = client.compute(draws, priority=10)

        # Export similarity
        export_sim_future = []
        if not args.no_export_sim:
            logger.info("Exporting similarities")

            exports = []
            similarities = []
            for future in matching_futures:
                similarity, name = future.result()
                sim_func_name = future.sim_func.__name__

                similarities.append((similarity, sim_func_name))

            candidate_pair = future.candidate_pair
            candidate_pair_scatter = candidate_pair_scatters[candidate_pair]
            similarities = client.scatter((similarities,))[0]

            exports.append(
                dask.delayed(export_similarity)(
                    candidate_pair_scatter,
                    similarities,
                    truth_pairs,
                    args.suffix,
                    args.recursive,
                )
            )

            export_sim_future = client.compute(exports, priority=10)

        # Export
        export_future = []
        if args.export:
            logger.info("Exporting solutions")
            exports = []
            for solution in solutions:
                exports.append(
                    dask.delayed(export)(
                        solution._final_candidate_pair,
                        solution.solution_future,
                        args.suffix,
                        func=solution.solve_func,
                    )
                )
            export_future = client.compute(exports, priority=10)

        # Submit solution stats (Delayed)
        for solution in solutions:
            solution.d_stats = get_solution_stats(
                solution._final_candidate_pair,
                truth_pairs,
                solution.solution_future,
                args.recursive,
            )

        d_solution_stats = [solution.d_stats for solution in solutions]
        solution_stats_futures = client.compute(d_solution_stats, priority=20)

        # Get max stats
        max_stats = {}
        with tqdm(
            total=len(max_stats_future),
            desc="Get max stats",
            dynamic_ncols=True,
        ) as pbar:
            for future, stats in as_completed(
                max_stats_future, with_results=True
            ):
                max_stats[future._candidate_pair] = stats
                pbar.update()

        # Get solution stats delayed (Delayed)
        d_future_soln_map = dict(zip(solution_stats_futures, solutions))
        with tqdm(
            total=len(solutions), desc="Counting", dynamic_ncols=True
        ) as pbar:
            for future, stats in as_completed(
                solution_stats_futures, with_results=True
            ):
                solution = d_future_soln_map[future]
                solution.max_matches = max_stats[solution.candidate_pair][0]
                solution.max_comparisons = max_stats[solution.candidate_pair][
                    1
                ]

                for k, v in stats.items():
                    setattr(solution, k, v)

                pbar.update()

        printable = []
        # Recall result
        for solution in tqdm(
            solutions, desc="Benchmarking", dynamic_ncols=True
        ):
            solve_args = "-".join(
                (
                    "{}={}".format(param, round(i, 5))
                    for param, i in solution.solve_func.keywords.items()
                )
            )

            printable.append(
                (
                    solution.candidate_pair[0].name,
                    solution.candidate_pair[1].name,
                    solution.tp,
                    solution.max_matches,
                    solution.comparisons,
                    solution.max_comparisons,
                    solve_args,
                    solution.sim_func.__name__,
                    solution.solve_func.__name__,
                    solution.recall,
                    solution.precision,
                    solution.f1,
                    solution.reduction,
                    solution.tradeoff,
                    solution.cpbur,
                    round(solution.elasped, 5),
                    solution.clusters,
                    round(solution.avg_size, 5),
                )
            )
            logger.debug(
                "Matched {} to {} with {}: R = {}".format(
                    solution.candidate_pair[0].name,
                    solution.candidate_pair[1].name,
                    solve_args,
                    solution.recall,
                )
            )

        printable.sort(key=operator.itemgetter(9, 13), reverse=True)

        # Write
        pd.DataFrame(
            printable,
            columns=[
                "name",
                "name",
                "matches",
                "max matches",
                "comparisons",
                "max comparisons",
                "args",
                "sim function",
                "solve function",
                "recall",
                "precision",
                "f1",
                "reduction",
                "tradeoff",
                "cpbur",
                "wall time",
                "clusters",
                "avg cardinality",
            ],
        ).to_csv(
            os.path.join(
                cfg.OUTPUT_DIR, "matched_benchmark_{}.csv".format(args.suffix)
            )
        )

        # Wait for draws
        logger.info("Waiting for any other tasks to finish...")
        wait(draw_future)
        wait(export_sim_future)
        wait(export_future)
        logger.info("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--fake",
        action="store_true",
        default=False,
        help="Run against fake data",
    )
    parser.add_argument(
        "-d",
        "--draw",
        action="store_true",
        default=False,
        help="Draw solution",
    )
    parser.add_argument(
        "-ns",
        "--no_export_sim",
        action="store_true",
        default=False,
        help="Export similarities",
    )
    parser.add_argument(
        "-x",
        "--export",
        action="store_true",
        default=False,
        help="Export solution",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="suffix",
        help="suffix to name of files",
        default="",
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        help="Epsilon value for DBscan multiplied by 10",
        default=40.0,
    )
    parser.add_argument(
        "-m",
        "--min_samples",
        type=int,
        help="Number of levels to trim leaf",
        default=2,
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        help="Number of rounds to cleanup",
        default=1,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of iterations to solve",
        default=1,
    )
    parser.add_argument(
        "-p", "--p_sample", type=float, help="Downsampling", default=1.0,
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="Random seed value", default=None,
    )
    parser.add_argument(
        "-a", "--a_file", type=FileType("r"), required=True,
    )
    parser.add_argument(
        "-b", "--b_file", type=FileType("r"), required=True,
    )
    parser.add_argument(
        "-t", "--truth_file", type=FileType("r"), required=True,
    )
    parser.add_argument(
        "-c",
        "--description",
        action="store_true",
        default=False,
        help="Add description",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursive mode",
    )
    parser.add_argument(
        "--o1", type=float, help="Omega", default=None,
    )
    parser.add_argument(
        "--o2", type=float, help="Omega", default=None,
    )
    parser.add_argument(
        "--e1", type=float, help="Eta", default=None,
    )
    parser.add_argument(
        "--e2", type=float, help="Eta", default=None,
    )
    parser.add_argument("--pickle", help="Use pickle", default=None)
    args = parser.parse_args()

    main(args)
