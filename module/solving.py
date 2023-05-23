import heapq
import itertools
import re
import sys
from collections import defaultdict, deque
from timeit import default_timer as timer
from typing import Collection

import dask
import networkx as nx
import numpy as np
from docplex.mp.model import Model
from entity.candidate import Candidate
from helper import logger
from scipy import sparse


def _tms_to_sol(tms, elasped):
    solution = dict()
    solution["elasped"] = elasped
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []
    solution["objective"] = tms.get_objective_value()

    for var, val in tms.iter_var_values():
        m = re.match(r"(\w)_(\D+)_(\d+)(_(\d+))?", str(var))
        axis = m.group(1)
        kind = m.group(2)
        d1 = int(m.group(3))

        if kind == "fac":
            if axis == "x":
                solution["x_fac"].append(d1)
            else:
                solution["y_fac"].append(d1)
        elif kind == "route":
            d2 = int(m.group(5))
            if axis == "x":
                solution["x_route"].append((d1, d2))
            elif axis == "y":
                solution["y_route"].append((d1, d2))
            else:
                logger.info("Unknown axis: {}".format(axis))
        else:
            logger.debug("Unknown kind: {}".format(kind))

    return solution


def _normalise(similarity):
    return (similarity - similarity.min()) / (
        similarity.max() - similarity.min()
    )


def _get_obj(solution, distance, fac_cost):
    distance_cost = 0
    for x, y in solution["x_route"]:
        distance_cost += distance[x, y]

    for y, x in solution["y_route"]:
        distance_cost += distance[x, y]

    return (
        len(solution["x_fac"]) + len(solution["y_fac"])
    ) * fac_cost + distance_cost


def _get_obj_use_sim(solution, similarity, fac_cost):
    distance_cost = 0
    for x, y in solution["x_route"]:
        distance_cost += 1 - similarity[x, y]

    for y, x in solution["y_route"]:
        distance_cost += 1 - similarity[x, y]

    return (
        len(solution["x_fac"]) + len(solution["y_fac"])
    ) * fac_cost + distance_cost


def solve_uflp_left(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    logger.info("Running solver for uflp left")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve uflp")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From x to y
    x_route = tm.binary_var_matrix(x_range, y_range, "x_route")
    x_fac = tm.binary_var_list(x_range, name="x_fac")

    # Obj
    x_cost = tm.sum(
        x_route[x, y] * (1 - norm_sim[x, y]) for x in x_range for y in y_range
    )
    f_cost = tm.sum(x_fac[x] for x in x_range) * w
    tm.minimize(tm.sum([x_cost, f_cost]))

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(tm.sum(x_route[x, y] for x in x_range) == 1)

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(x_route[x, y] <= x_fac[x])

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve uflp left")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_uflp_right(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    logger.info("Running solver for uflp left")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve uflp")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From y to x
    y_route = tm.binary_var_matrix(y_range, x_range, "y_route")
    y_fac = tm.binary_var_list(y_range, name="y_fac")

    # Obj
    y_cost = tm.sum(
        y_route[y, x] * (1 - norm_sim[x, y]) for x in x_range for y in y_range
    )
    f_cost = tm.sum(y_fac[y] for y in y_range) * w
    tm.minimize(tm.sum([y_cost, f_cost]))

    # tm.print_information()

    # Constrains
    for x in x_range:
        tm.add_constraint(tm.sum(y_route[y, x] for y in y_range) == 1)

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(y_route[y, x] <= y_fac[y])

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve uflp left")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_poly_left(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.info("Running solver for poly left")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve poly")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From x to y
    x_route = tm.binary_var_matrix(x_range, y_range, "x_route")

    x_fac = tm.binary_var_list(x_range, name="x_fac")
    y_fac = tm.binary_var_list(y_range, name="y_fac")

    # Selfless
    x_z = tm.binary_var_list(x_range, name="x_z")

    # Obj
    x_cost = tm.sum(
        x_route[x, y] * norm_sim[x, y] for x in x_range for y in y_range
    )
    f_cost = tm.sum(
        [tm.sum(x_fac[x] for x in x_range), tm.sum(y_fac[y] for y in y_range)]
    ) * (w)
    z_cost = tm.sum(x_z[x] for x in x_range) * (n - w)
    tm.maximize(tm.sum([x_cost, f_cost, z_cost]))

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(
            tm.sum([y_fac[y], tm.sum(x_route[x, y] for x in x_range)]) == 1
        )

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(x_route[x, y] <= x_fac[x])
        tm.add_constraint(x_route[x, y] <= x_z[x])

    for x in x_range:
        tm.add_constraint(x_z[x] <= tm.sum(x_route[x, y] for y in y_range))
        tm.add_constraint(x_z[x] <= x_fac[x])

        tm.add_constraint(x_fac[x] == 1)

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve poly left")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_poly_right(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.info("Running solver for poly left")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve poly")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From y to x
    y_route = tm.binary_var_matrix(y_range, x_range, "y_route")

    x_fac = tm.binary_var_list(x_range, name="x_fac")
    y_fac = tm.binary_var_list(y_range, name="y_fac")

    # Selfless
    y_z = tm.binary_var_list(y_range, name="y_z")

    # Obj
    y_cost = tm.sum(
        y_route[y, x] * norm_sim[x, y] for x in x_range for y in y_range
    )
    f_cost = tm.sum(y_fac[y] for y in y_range) * w
    z_cost = tm.sum(y_z[y] for y in y_range) * (n - w)
    tm.maximize(tm.sum([y_cost, f_cost, z_cost]))

    # tm.print_information()

    # Constrains
    for x in x_range:
        tm.add_constraint(
            tm.sum([x_fac[x], tm.sum(y_route[y, x] for y in y_range)]) == 1
        )

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(y_route[y, x] <= y_fac[y])
        tm.add_constraint(y_route[y, x] <= y_z[y])

    for y in y_range:
        tm.add_constraint(y_z[y] <= tm.sum(y_route[y, x] for x in x_range))
        tm.add_constraint(y_z[y] <= y_fac[y])

        tm.add_constraint(y_fac[y] == 1)

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve poly left")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_bipoly(candidate_pair: Collection[Candidate], similarity, **kwargs):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.info("Running solver with w={}, n={}".format(w, n))
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve (w={}, n={})".format(w, n))
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # Define vars
    x_fac = tm.binary_var_list(x_range, name="x_fac")
    y_fac = tm.binary_var_list(y_range, name="y_fac")

    # Selfless
    x_z = tm.binary_var_list(x_range, name="x_z")
    y_z = tm.binary_var_list(y_range, name="y_z")

    # From x to y
    x_route = tm.binary_var_matrix(x_range, y_range, "x_route")
    # From y to x
    y_route = tm.binary_var_matrix(y_range, x_range, "y_route")

    # Obj
    x_cost = tm.sum(
        x_route[x, y] * norm_sim[x, y] for x in x_range for y in y_range
    )
    y_cost = tm.sum(
        y_route[y, x] * norm_sim[x, y] for x in x_range for y in y_range
    )
    f_cost = tm.sum(
        [tm.sum(x_fac[x] for x in x_range), tm.sum(y_fac[y] for y in y_range)]
    ) * (w)
    z_cost = tm.sum(
        [tm.sum(x_z[x] for x in x_range), tm.sum(y_z[y] for y in y_range)]
    ) * (n - w)
    tm.maximize(tm.sum([x_cost, y_cost, f_cost, z_cost]))

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(
            tm.sum([y_fac[y], tm.sum(x_route[x, y] for x in x_range)]) == 1
        )

        tm.add_constraint(y_z[y] <= tm.sum(y_route[y, x] for x in x_range))
        tm.add_constraint(y_z[y] <= y_fac[y])

    for x in x_range:
        tm.add_constraint(
            tm.sum([x_fac[x], tm.sum(y_route[y, x] for y in y_range)]) == 1
        )

        tm.add_constraint(x_z[x] <= tm.sum(x_route[x, y] for y in y_range))
        tm.add_constraint(x_z[x] <= x_fac[x])

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(x_route[x, y] <= x_fac[x])
        tm.add_constraint(y_route[y, x] <= y_fac[y])

        tm.add_constraint(x_route[x, y] <= x_z[x])
        tm.add_constraint(y_route[y, x] <= y_z[y])

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve selfless")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_bipoly_quota(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.info("Running solver with w={}, n={}".format(w, n))
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve (w={}, n={})".format(w, n))
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # Define vars
    x_fac = tm.binary_var_list(x_range, name="x_fac")
    y_fac = tm.binary_var_list(y_range, name="y_fac")

    # Selfless
    x_z = tm.binary_var_list(x_range, name="x_z")
    y_z = tm.binary_var_list(y_range, name="y_z")

    # From x to y
    x_route = tm.binary_var_matrix(x_range, y_range, "x_route")
    # From y to x
    y_route = tm.binary_var_matrix(y_range, x_range, "y_route")

    # Obj
    x_cost = tm.sum(
        x_route[x, y] * norm_sim[x, y] for x in x_range for y in y_range
    )
    y_cost = tm.sum(
        y_route[y, x] * norm_sim[x, y] for x in x_range for y in y_range
    )
    f_cost = tm.sum(
        [tm.sum(x_fac[x] for x in x_range), tm.sum(y_fac[y] for y in y_range)]
    ) * (w)
    z_cost = tm.sum(
        [tm.sum(x_z[x] for x in x_range), tm.sum(y_z[y] for y in y_range)]
    ) * (n - w)
    tm.maximize(tm.sum([x_cost, y_cost, f_cost, z_cost]))

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(
            tm.sum([y_fac[y], tm.sum(x_route[x, y] for x in x_range)]) == 1
        )

        tm.add_constraint(y_z[y] <= tm.sum(y_route[y, x] for x in x_range))
        tm.add_constraint(y_z[y] <= y_fac[y])

        tm.add_constraint(
            tm.sum(y_route[y, x] for x in x_range) <= quota * y_fac[y]
        )

    for x in x_range:
        tm.add_constraint(
            tm.sum([x_fac[x], tm.sum(y_route[y, x] for y in y_range)]) == 1
        )

        tm.add_constraint(x_z[x] <= tm.sum(x_route[x, y] for y in y_range))
        tm.add_constraint(x_z[x] <= x_fac[x])

        tm.add_constraint(
            tm.sum(x_route[x, y] for y in y_range) <= quota * x_fac[x]
        )

    for x, y in itertools.product(x_range, y_range):
        tm.add_constraint(x_route[x, y] <= x_z[x])
        tm.add_constraint(y_route[y, x] <= y_z[y])

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve selfless")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_bmatching_left(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.info("Running solver b-matching")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve b-matching")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From x to y
    x_route = tm.binary_var_matrix(x_range, y_range, "x_route")

    # Obj
    x_cost = tm.sum(
        x_route[x, y] * norm_sim[x, y] for x in x_range for y in y_range
    )
    tm.maximize(x_cost)

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(tm.sum(x_route[x, y] for x in x_range) <= 1)

    for x in x_range:
        tm.add_constraint(
            tm.sum(tm.sum(x_route[x, y] for y in y_range)) <= quota
        )

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve b-matching")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def solve_bmatching_right(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.info("Running solver b-matching")
    start = timer()

    norm_sim = _normalise(similarity)
    x_range = range(len(candidate_pair[0].G))
    y_range = range(len(candidate_pair[1].G))

    tm = Model(name="solve b-matching")
    # tm.context.cplex_parameters.threads = multiprocessing.cpu_count()

    # From y to x
    y_route = tm.binary_var_matrix(y_range, x_range, "y_route")

    # Obj
    y_cost = tm.sum(
        y_route[y, x] * norm_sim[x, y] for x in x_range for y in y_range
    )
    tm.maximize(y_cost)

    # tm.print_information()

    # Constrains
    for y in y_range:
        tm.add_constraint(
            tm.sum(tm.sum(y_route[y, x] for x in x_range)) <= quota
        )

    for x in x_range:
        tm.add_constraint(tm.sum(y_route[y, x] for y in y_range) == 1)

    # Solve
    tm.print_information()
    tms = tm.solve()
    # tms.display()

    logger.info("Solved using solve b-matching")
    end = timer()
    elasped = end - start

    return _tms_to_sol(tms, elasped)


def _get_smallest_in_clique(
    clique, G_fac, path_cost, distance, fac_cost, weight, reverse
):

    best_cost = sys.float_info.max
    best_clique = []
    best_fac = None

    total_path_cost = 0
    total_path_num = 0
    for a, b in itertools.combinations(clique, 2):
        total_path_cost += path_cost[a][b]
        total_path_num += 1

    for fac in G_fac:
        total_distance_cost = 0
        for node in clique:
            if reverse:
                total_distance_cost += distance[fac, node]
            else:
                total_distance_cost += distance[node, fac]

        cost = (total_distance_cost + fac_cost) / len(clique)
        if total_path_num > 0:
            cost += weight * (total_path_cost / total_path_num)

        if cost <= best_cost:
            best_cost = cost
            best_clique = clique
            best_fac = fac

    return best_cost, best_clique, best_fac


@dask.delayed
def _get_smallest_in_cliques(
    outputs, G_fac, path_cost, distance, fac_cost, weight, reverse
):
    if not outputs:
        logger.debug("No outputs")

    best_cost = sys.float_info.max
    best_clique = []
    best_fac = None

    for output in outputs:
        base, cnbrs = output
        cost, clique, fac = _get_smallest_in_clique(
            base, G_fac, path_cost, distance, fac_cost, weight, reverse
        )

        if cost <= best_cost:
            best_cost = cost
            best_clique = clique
            best_fac = fac

    logger.debug(
        "Clique: cost = {}, best_clique = {}, best_face = {}".format(
            best_cost, best_clique, best_fac
        )
    )
    return best_cost, best_clique, best_fac


def _get_largest_for_fac(G, fac, distance, reverse, **kwargs):
    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    if reverse:
        distance_vec = distance[fac, :]
    else:
        distance_vec = distance[:, fac]
    logger.debug(distance_vec)

    sorted_indexes = np.argsort(distance_vec)
    logger.debug(sorted_indexes)

    best_score = w
    customers = []
    prior_distances_cost = 0
    for index in sorted_indexes[::-1]:
        if index not in G:
            continue

        if distance_vec[index] == 0:
            break

        similarity = (n + prior_distances_cost + distance_vec[index]) / (
            len(customers) + 2
        )

        if similarity > best_score:
            best_score = similarity
            customers.append(index)
            prior_distances_cost += distance_vec[index]
        else:
            break

    return best_score, customers, fac


def _get_largest_within_G(
    G, G_fac, distance, reverse=False, for_fac=_get_largest_for_fac, **kwargs
):
    best_score = None
    best_customers = None
    best_fac = None
    for fac in G_fac:
        similarity, customers, fac = for_fac(
            G, fac, distance, reverse, **kwargs
        )

        if best_score is None or similarity > best_score:
            best_score = similarity
            best_customers = customers
            best_fac = fac

    logger.debug("Got result")

    return best_score, best_customers, best_fac


def solve_bipartite(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        w = 0

    logger.info("Running solve bipartite")
    start = timer()
    norm_sim = _normalise(similarity)

    solution = dict()
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []

    G_x: nx.Graph
    G_x = nx.convert_node_labels_to_integers(candidate_pair[0].G)

    x_length = len(candidate_pair[0].G)

    G_y: nx.Graph
    G_y = nx.convert_node_labels_to_integers(
        candidate_pair[1].G, first_label=x_length
    )

    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(G_x, bipartite=0)
    B.add_nodes_from(G_y, bipartite=1)
    # Add edges only between nodes of opposite node sets
    for x, y in itertools.product(G_x, G_y):
        weight = norm_sim[x, y - x_length]
        if weight > w:
            B.add_edge(x, y, weight=norm_sim[x, y - x_length])

    assert nx.is_bipartite(B)

    result = nx.max_weight_matching(B)
    logger.debug(result)

    for u, v in result:
        if u < x_length:
            assert v >= x_length
            x = u
            y = v - x_length
        else:
            assert v < x_length
            x = v
            y = u - x_length

        solution["x_fac"].append(x)
        solution["x_route"].append((x, y))

    solution["objective"] = _get_obj_use_sim(solution, norm_sim, 0)

    logger.info("Solved using bipartite")
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    return solution


def solve_stable_marriage(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):
    logger.info("Running solve stable marriage")
    start = timer()

    norm_sim = _normalise(similarity)
    if sparse.issparse(norm_sim):
        norm_sim = norm_sim.toarray()

    solution = dict()
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []

    flipped = False
    if len(candidate_pair[0].G) > len(candidate_pair[1].G):
        norm_sim = norm_sim.T
        flipped = True

    arg_sorted = np.argsort(norm_sim, axis=1)

    if flipped:
        not_married = deque(range(len(candidate_pair[1].G)))
    else:
        not_married = deque(range(len(candidate_pair[0].G)))
    married_to = {}
    while not_married:
        m = not_married.popleft()
        for w in arg_sorted[m, :][::-1]:
            if w not in married_to:
                married_to[w] = m
                break
            else:
                m_old = married_to[w]
                if norm_sim[m, w] > norm_sim[m_old, w]:
                    married_to[w] = m
                    not_married.append(m_old)
                    break

    if flipped:
        for x, y in married_to.items():
            solution["x_fac"].append(x)
            solution["x_route"].append((x, y))
    else:
        for y, x in married_to.items():
            solution["x_fac"].append(x)
            solution["x_route"].append((x, y))

    solution["objective"] = _get_obj_use_sim(solution, similarity, 0)

    logger.info("Solved using stable marriage")
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    return solution


def solve_bsuitor(
    candidate_pair: Collection[Candidate], similarity, reverse=False, **kwargs
):

    if "w" in kwargs:
        w = kwargs["w"]
    else:
        w = 0

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.info("Running b-suitor quota={}, reverse={}".format(quota, reverse))
    start = timer()

    solution = dict()
    solution["x_fac"] = set()
    solution["y_fac"] = set()
    solution["x_route"] = set()
    solution["y_route"] = set()

    x_length = len(candidate_pair[0].G)
    y_length = len(candidate_pair[1].G)

    norm_sim = _normalise(similarity)
    if sparse.issparse(norm_sim):
        norm_sim = norm_sim.toarray()

    threshold_indices = norm_sim <= w
    norm_sim[threshold_indices] = 0

    weight_x = np.argsort(norm_sim, axis=1)[:, ::-1] + x_length
    weight_y = np.argsort(norm_sim.T, axis=1)[:, ::-1]
    candidates = weight_x.tolist() + weight_y.tolist()
    logger.debug(candidates)

    s = defaultdict(list)
    t = defaultdict(set)

    def b(index):
        if reverse:
            if index < x_length:
                return 1
            else:
                return quota
        else:
            if index < x_length:
                return quota
            else:
                return 1

    def w(u, v):
        if u < v:
            return norm_sim[u, v - x_length]
        else:
            return norm_sim[v, u - x_length]

    def _make_suitor(u, x):
        if len(s[x]) < b(x):
            y = None
        else:
            y = heapq.heappop(s[x])[1]

        heapq.heappush(s[x], (w(u, x), u))
        t[u].add(x)

        if y is not None:
            z = None
            t[y].remove(x)
            for v in candidates[y]:
                if v in t[y]:
                    continue
                elif w(y, v) == 0:
                    break
                elif len(s[v]) < b(v) or w(y, v) > min(s[v])[0]:
                    z = v
                    break

            if z is not None:
                _make_suitor(y, z)

    for u in range(x_length + y_length):
        for _ in range(b(u)):
            x = None
            for v in candidates[u]:
                if v in t[u]:
                    continue
                elif w(u, v) == 0:
                    break
                elif len(s[v]) < b(v) or w(u, v) > min(s[v])[0]:
                    x = v
                    break

            if x is None:
                break
            else:
                _make_suitor(u, x)

    logger.debug(s)
    logger.debug(t)

    for i, j in t.items():
        if reverse:
            if i < x_length:
                solution["x_fac"].add(i)
                for y in j:
                    solution["x_route"].add((i, y - x_length))
            else:
                for x in j:
                    solution["x_fac"].add(x)
                    solution["x_route"].add((x, i - x_length))
        else:
            if i < x_length:
                for y in j:
                    solution["y_fac"].add(y - x_length)
                    solution["y_route"].add((y - x_length, i))
            else:
                solution["y_fac"].add(i - x_length)
                for x in j:
                    solution["y_route"].add((i - x_length, x))

    solution["x_fac"] = list(solution["x_fac"])
    solution["y_fac"] = list(solution["y_fac"])
    solution["x_route"] = list(solution["x_route"])
    solution["y_route"] = list(solution["y_route"])

    solution["objective"] = _get_obj(solution, norm_sim, 0)

    logger.info(
        "Solved using b-suitor quota={}, reverse={}".format(quota, reverse)
    )
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    logger.debug(solution)
    return solution


def _get_smallest_for_fac_mem(G, fac, similarity, reverse, **kwargs):
    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.debug("Getting the row/col vec...")
    if reverse:
        similarity_vec = similarity[fac, :]
    else:
        similarity_vec = similarity[:, fac]
    logger.debug(similarity_vec)

    best_score = w
    customers = {}
    prior_distances_cost = 0
    if sparse.issparse(similarity_vec):
        if similarity_vec.nnz == 0:
            return best_score, customers.keys(), fac, []

        if len(similarity_vec.indptr) != 2:
            raise Exception("indptr != 2")

        logger.debug("Getting score using sparse vec...")
        distance_vec = [1 - x for x in similarity_vec.data]
        h = list(zip(distance_vec, similarity_vec.indices))
        h.sort()
    else:
        distance_vec = 1 - similarity_vec
        logger.debug("Sorting...")
        sorted_indexes = np.argsort(distance_vec)
        for end_index, index in enumerate(sorted_indexes):
            if distance_vec[index] == 1:
                break

        logger.debug(sorted_indexes)
        if end_index == 0:
            sorted_indexes = []
        else:
            sorted_indexes = sorted_indexes[:end_index]
        logger.debug(sorted_indexes)

        h = [(distance_vec[index], index) for index in sorted_indexes]

    logger.debug(h)
    for distance, index in h:
        if index not in G:
            continue

        score = (n + prior_distances_cost + distance) / (len(customers) + 2)

        if score < best_score:
            best_score = score
            customers[index] = None
            prior_distances_cost += distance
        else:
            break

    return best_score, customers.keys(), fac, h


def _get_best_list(
    G,
    G_fac,
    similarity,
    reverse=False,
    for_fac=_get_smallest_for_fac_mem,
    **kwargs
):
    ret = [
        for_fac(G, fac, similarity, reverse, **kwargs) + (reverse,)
        for fac in G_fac
    ]
    logger.debug("Got list")

    return ret


def _recalculate_score(
    clique, fac, leads, available, leftovers, sim, reverse, w, n, quota=None
):
    logger.debug("Recalculating score...")
    best_score = w
    customers = {}
    prior_distances_cost = 0

    for node in clique:
        if node not in leftovers:
            customers[node] = None
            if reverse:
                distance = 1 - sim[fac, node]
            else:
                distance = 1 - sim[node, fac]
            prior_distances_cost += distance
            last_node = node
            continue
        break

    init_pointer = 0
    pointer = 0
    if customers:
        first_node = next(iter(customers))
        best_score = (n + prior_distances_cost) / (len(customers) + 1)
        for pointer, (_, index) in enumerate(leads):
            if index == first_node:
                init_pointer = pointer
            if index == last_node:
                break
        pointer += 1

    for distance, index in leads[pointer:]:
        if index not in available:
            continue
        elif quota is not None and len(customers) >= quota:
            break

        score = (n + prior_distances_cost + distance) / (len(customers) + 2)

        if score < best_score:
            best_score = score
            customers[index] = None
            prior_distances_cost += distance
        else:
            break

    logger.debug("Recalculated score.")

    if init_pointer > 0:
        return best_score, customers.keys(), leads[init_pointer:]
    return best_score, customers.keys(), leads


def solve_greedy_bipoly(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = (1 - kwargs["w"], 1 - kwargs["w"])
    elif "w1" in kwargs and "w2" in kwargs:
        w = (1 - kwargs["w1"], 1 - kwargs["w2"])
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = (1 - kwargs["n"], 1 - kwargs["n"])
    elif "n1" in kwargs and "n2" in kwargs:
        n = (1 - kwargs["n1"], 1 - kwargs["n2"])
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.info("Running greedy with w={}, n={}".format(w, n))
    start = timer()

    solution = dict()
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []

    norm_sim = _normalise(similarity)
    if sparse.issparse(norm_sim):
        logger.debug("Converting to the correct sparse index...")
        sim = norm_sim.tocsc()
        sim_rev = norm_sim.tocsr()
        logger.debug("Done converting to the correct sparse index.")
    else:
        sim = norm_sim
        sim_rev = norm_sim

    G_x: nx.Graph
    G_x = set(nx.convert_node_labels_to_integers(candidate_pair[0].G))

    G_y: nx.Graph
    G_y = set(nx.convert_node_labels_to_integers(candidate_pair[1].G))

    x_length = len(candidate_pair[0].G)
    y_length = len(candidate_pair[1].G)

    h = deque()
    if x_length > 0 and y_length > 0:
        logger.debug("Calculating lists...")
        h.extend(_get_best_list(G_x, G_y, sim, w=w[False], n=n[False]))
        h.extend(
            _get_best_list(
                G_y, G_x, sim_rev, reverse=True, w=w[True], n=n[True]
            )
        )

    insert_fac = ("y_fac", "x_fac")
    insert_route = ("y_route", "x_route")
    removal = (G_x, G_y)
    init = True
    while x_length > 0 and y_length > 0:
        logger.debug(x_length)
        logger.debug(y_length)

        if not init:
            new_h = deque()
            while h:
                score, clique, fac, leads, reverse = h.popleft()
                if fac not in removal[1 - reverse]:
                    continue

                leftovers = clique - removal[reverse]
                if leftovers:
                    score, customers, leads = _recalculate_score(
                        clique,
                        fac,
                        leads,
                        removal[reverse],
                        leftovers,
                        norm_sim,
                        reverse,
                        w[reverse],
                        n[reverse],
                    )
                    new_h.append((score, customers, fac, leads, reverse))
                else:
                    new_h.append((score, clique, fac, leads, reverse))
            h = new_h
        else:
            init = False

        if not h:
            break
        score, clique, fac, _, reverse = min(h)

        logger.debug(
            "Got best. score={}, clique={}, fac={}, rev={}".format(
                score, clique, fac, reverse
            )
        )

        solution[insert_fac[reverse]].append(fac)
        for node in clique:
            if node not in removal[reverse]:
                break
            solution[insert_route[reverse]].append((fac, node))
            removal[reverse].remove(node)
        removal[1 - reverse].remove(fac)

        x_length = len(G_x)
        y_length = len(G_y)

        logger.debug(solution)

    if x_length > 0:
        for node in G_x:
            solution["x_fac"].append(node)
    else:
        for node in G_y:
            solution["y_fac"].append(node)

    solution["objective"] = _get_obj_use_sim(solution, sim_rev, 1 - w[0])

    logger.info("Solved using greedy with w={}, n={}".format(w, n))
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    logger.debug(solution)
    return solution


def solve_greedy_poly(
    candidate_pair: Collection[Candidate], similarity, reverse=False, **kwargs
):

    if "w" in kwargs:
        w = 1 - kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = 1 - kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    logger.info("Running greedy with w={}, n={}".format(w, n))
    start = timer()

    solution = dict()
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []

    norm_sim = _normalise(similarity)
    if sparse.issparse(norm_sim):
        logger.debug("Converting to the correct sparse index...")
        if reverse:
            sim = norm_sim.tocsr()
        else:
            sim = norm_sim.tocsc()
        logger.debug("Done converting to the correct sparse index.")
    else:
        sim = norm_sim

    G_x: nx.Graph
    G_x = nx.convert_node_labels_to_integers(candidate_pair[0].G)

    G_y: nx.Graph
    G_y = nx.convert_node_labels_to_integers(candidate_pair[1].G)

    x_length = len(candidate_pair[0].G)
    y_length = len(candidate_pair[1].G)

    removal = (G_x, G_y)
    h = deque()
    if x_length > 0 and y_length > 0:
        logger.debug("Calculating lists...")
        h.extend(
            _get_best_list(
                removal[reverse],
                removal[1 - reverse],
                sim,
                reverse=reverse,
                w=w,
                n=n,
            )
        )

    insert_fac = ("y_fac", "x_fac")
    insert_route = ("y_route", "x_route")
    init = True
    while x_length > 0 and y_length > 0:
        logger.debug(x_length)
        logger.debug(y_length)

        if not init:
            new_h = deque()
            while h:
                score, clique, fac, leads, reverse = h.popleft()
                if fac not in removal[1 - reverse]:
                    continue

                leftovers = clique - removal[reverse]
                if leftovers:
                    score, customers, leads = _recalculate_score(
                        clique,
                        fac,
                        leads,
                        removal[reverse],
                        leftovers,
                        norm_sim,
                        reverse,
                        w,
                        n,
                    )
                    new_h.append((score, customers, fac, leads, reverse))
                else:
                    new_h.append((score, clique, fac, leads, reverse))
            h = new_h
        else:
            init = False

        if not h:
            break
        score, clique, fac, _, reverse = min(h)

        # Termination condition, when they should all just be alone.
        if score > w:
            logger.error("Breakout")
            break

        logger.debug("Got best. Score={}".format(score))

        solution[insert_fac[reverse]].append(fac)
        for node in clique:
            if node not in removal[reverse]:
                break
            solution[insert_route[reverse]].append((fac, node))
            removal[reverse].remove_node(node)
        removal[1 - reverse].remove_node(fac)

        x_length = len(G_x)
        y_length = len(G_y)

        logger.debug(solution)

    if x_length > 0:
        for node in G_x:
            solution["x_fac"].append(node)
    else:
        for node in G_y:
            solution["y_fac"].append(node)

    solution["objective"] = _get_obj_use_sim(solution, norm_sim, 1 - w)

    logger.info("Solved using greedy with w={}, n={}".format(w, n))
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    logger.debug(solution)
    return solution


def _get_smallest_for_fac_quota(G, fac, similarity, reverse, **kwargs):
    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.debug("Getting the row/col vec")
    if reverse:
        similarity_vec = similarity[fac, :]
    else:
        similarity_vec = similarity[:, fac]
    logger.debug(similarity_vec)

    best_score = w
    customers = []
    prior_distances_cost = 0
    if sparse.issparse(similarity_vec):
        if similarity_vec.nnz == 0:
            return best_score, customers, fac

        if len(similarity_vec.indptr) != 2:
            return

        logger.debug("Getting score using sparse vec...")
        distance_vec = [1 - x for x in similarity_vec.data]
        h = list(zip(distance_vec, similarity_vec.indices))
        heapq.heapify(h)
        logger.debug(h)

        while h:
            distance, index = heapq.heappop(h)
            if index not in G:
                continue

            score = (n + prior_distances_cost + distance) / (
                len(customers) + 2
            )

            if score < best_score:
                best_score = score
                customers.append(index)
                prior_distances_cost += distance
            else:
                break

            if len(customers) >= quota:
                break
    else:
        distance_vec = 1 - similarity_vec
        logger.debug("Sorting...")
        sorted_indexes = np.argsort(distance_vec)
        logger.debug(sorted_indexes)

        for index in sorted_indexes:
            if index not in G:
                continue

            distance = distance_vec[index]
            if distance == 1:
                break

            score = (n + prior_distances_cost + distance) / (
                len(customers) + 2
            )

            if score < best_score:
                best_score = score
                customers.append(index)
                prior_distances_cost += distance
            else:
                break

            if len(customers) >= quota:
                break

    return best_score, customers, fac


def _get_smallest_for_fac_quota_mem(G, fac, similarity, reverse, **kwargs):
    if "w" in kwargs:
        w = kwargs["w"]
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = kwargs["n"]
    else:
        raise Exception("No kwarg 'n' provided.")

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.debug("Getting the row/col vec...")
    if reverse:
        similarity_vec = similarity[fac, :]
    else:
        similarity_vec = similarity[:, fac]
    logger.debug(similarity_vec)

    best_score = w
    customers = {}
    prior_distances_cost = 0
    if sparse.issparse(similarity_vec):
        if similarity_vec.nnz == 0:
            return best_score, customers.keys(), fac, []

        if len(similarity_vec.indptr) != 2:
            raise Exception("indptr != 2")

        logger.debug("Getting score using sparse vec...")
        distance_vec = [1 - x for x in similarity_vec.data]
        h = list(zip(distance_vec, similarity_vec.indices))
        h.sort()
    else:
        distance_vec = 1 - similarity_vec
        logger.debug("Sorting...")
        sorted_indexes = np.argsort(distance_vec)
        for end_index, index in enumerate(sorted_indexes):
            if distance_vec[index] == 1:
                break

        logger.debug(sorted_indexes)
        if end_index == 0:
            sorted_indexes = []
        else:
            sorted_indexes = sorted_indexes[:end_index]
        logger.debug(sorted_indexes)

        h = [(distance_vec[index], index) for index in sorted_indexes]

    logger.debug(h)
    for distance, index in h:
        if index not in G:
            continue

        score = (n + prior_distances_cost + distance) / (len(customers) + 2)

        if score < best_score:
            best_score = score
            customers[index] = None
            prior_distances_cost += distance
        else:
            break

        if len(customers) >= quota:
            break

    return best_score, customers.keys(), fac, h


def solve_greedy_bipoly_quota(
    candidate_pair: Collection[Candidate], similarity, **kwargs
):

    if "w" in kwargs:
        w = (1 - kwargs["w"], 1 - kwargs["w"])
    elif "w1" in kwargs and "w2" in kwargs:
        w = (1 - kwargs["w1"], 1 - kwargs["w2"])
    else:
        raise Exception("No kwarg 'w' provided.")

    if "n" in kwargs:
        n = (1 - kwargs["n"], 1 - kwargs["n"])
    elif "n1" in kwargs and "n2" in kwargs:
        n = (1 - kwargs["n1"], 1 - kwargs["n2"])
    else:
        raise Exception("No kwarg 'n' provided.")

    if "quota" in kwargs:
        quota = kwargs["quota"]
    else:
        raise Exception("No kwarg 'quota' provided.")

    logger.info("Running greedy with w={}, n={}".format(w, n))
    start = timer()

    solution = dict()
    solution["x_fac"] = []
    solution["y_fac"] = []
    solution["x_route"] = []
    solution["y_route"] = []

    norm_sim = _normalise(similarity)
    if sparse.issparse(norm_sim):
        logger.debug("Converting to the correct sparse index...")
        sim = norm_sim.tocsc()
        sim_rev = norm_sim.tocsr()
        logger.debug("Done converting to the correct sparse index.")
    else:
        sim = norm_sim
        sim_rev = norm_sim

    G_x: nx.Graph
    G_x = set(nx.convert_node_labels_to_integers(candidate_pair[0].G))

    G_y: nx.Graph
    G_y = set(nx.convert_node_labels_to_integers(candidate_pair[1].G))

    x_length = len(candidate_pair[0].G)
    y_length = len(candidate_pair[1].G)

    h = deque()
    if x_length > 0 and y_length > 0:
        logger.debug("Calculating lists...")
        h.extend(
            _get_best_list(
                G_x,
                G_y,
                sim,
                for_fac=_get_smallest_for_fac_quota_mem,
                w=w[False],
                n=n[False],
                quota=quota,
            )
        )
        h.extend(
            _get_best_list(
                G_y,
                G_x,
                sim_rev,
                reverse=True,
                for_fac=_get_smallest_for_fac_quota_mem,
                w=w[True],
                n=n[True],
                quota=quota,
            )
        )
        # print(h)

    insert_fac = ("y_fac", "x_fac")
    insert_route = ("y_route", "x_route")
    removal = (G_x, G_y)
    init = True
    while x_length > 0 and y_length > 0:
        logger.debug(x_length)
        logger.debug(y_length)

        if not init:
            new_h = deque()
            while h:
                score, clique, fac, leads, reverse = h.popleft()
                if fac not in removal[1 - reverse]:
                    continue

                leftovers = clique - removal[reverse]
                if leftovers:
                    score, customers, leads = _recalculate_score(
                        clique,
                        fac,
                        leads,
                        removal[reverse],
                        leftovers,
                        norm_sim,
                        reverse,
                        w[reverse],
                        n[reverse],
                        quota=quota,
                    )
                    new_h.append((score, customers, fac, leads, reverse))
                else:
                    new_h.append((score, clique, fac, leads, reverse))
            h = new_h
        else:
            init = False

        if not h:
            break
        score, clique, fac, _, reverse = min(h)

        logger.debug(
            "Got best. score={}, clique={}, fac={}, rev={}".format(
                score, clique, fac, reverse
            )
        )

        solution[insert_fac[reverse]].append(fac)
        for node in clique:
            if node not in removal[reverse]:
                break
            solution[insert_route[reverse]].append((fac, node))
            removal[reverse].remove(node)
        removal[1 - reverse].remove(fac)

        x_length = len(G_x)
        y_length = len(G_y)

        logger.debug(solution)

    if x_length > 0:
        for node in G_x:
            solution["x_fac"].append(node)
    else:
        for node in G_y:
            solution["y_fac"].append(node)

    solution["objective"] = _get_obj_use_sim(solution, similarity, 1 - w[0])

    logger.info("Solved using greedy with w={}, n={}".format(w, n))
    end = timer()
    elasped = end - start
    solution["elasped"] = elasped

    logger.debug(solution)
    return solution

