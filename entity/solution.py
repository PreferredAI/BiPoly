from typing import Collection

from entity.candidate import Candidate


class Solution(object):
    def __init__(
        self, candidate_pair: Collection[Candidate], sim_func, solve_func
    ):
        self.candidate_pair = candidate_pair
        self.sim_func = sim_func
        self.solve_func = solve_func

        self.solution = None

        self.tp = None
        self.fp = None
        self.max_matches = None
        self.comparisons = None
        self.max_comparisons = None

    def __getattr__(self, attr):
        if attr == "recall":
            if (
                self.tp is None
                or self.max_matches is None
                or self.max_matches == 0
            ):
                return None
            return self.tp / self.max_matches
        elif attr == "precision":
            if self.tp is None or self.fp is None or self.tp + self.fp == 0:
                return None
            return self.tp / (self.tp + self.fp)
        elif attr == "f1":
            if (
                self.recall is None
                or self.precision is None
                or self.recall + self.precision == 0
            ):
                return None
            return (
                2
                * (self.recall * self.precision)
                / (self.recall + self.precision)
            )
        elif attr == "reduction":
            if (
                self.comparisons is None
                or self.max_comparisons is None
                or self.max_comparisons == 0
            ):
                return None
            return 1 - (self.comparisons / self.max_comparisons)
        elif attr == "tradeoff":
            if (
                self.recall is None
                or self.reduction is None
                or self.reduction + self.recall == 0
            ):
                return None
            return (
                2
                * (self.reduction * self.recall)
                / (self.reduction + self.recall)
            )
        elif attr == "cpbur":
            if (
                self.comparisons is None
                or self.recall is None
                or self.recall == 0
            ):
                return None
            return self.comparisons / self.recall / 1000
