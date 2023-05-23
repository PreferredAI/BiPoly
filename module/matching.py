import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Collection, Union

import networkx as nx
from dask.distributed import Client
from dask.distributed import as_completed as d_as_completed
from networkx.classes.graph import Graph
from tqdm import tqdm

import config
from entity.candidate import Candidate
from helper import logger
from module import textprocessing


class Matcher(object):
    def __init__(self):
        self._candidates = []
        self._prepared = False

    def add(self, candidate: Candidate):
        self._candidates.append(candidate)

    def _get_all_attributes(self, thread_pool):
        futures = []
        candidate: Candidate
        for i, candidate in enumerate(self._candidates):
            for node_name in tqdm(
                candidate.G,
                dynamic_ncols=True,
                desc="Submitting ({}/{})".format(i + 1, len(self._candidates)),
            ):
                future = thread_pool.submit(
                    candidate.get_attributes, node_name, should_cache=False
                )
                future.candidate = candidate
                future.node_name = node_name
                futures.append(future)

        return futures

    def _stem_all_attribute(self, client, io_futures, batch_size):
        stem_futures = []
        candidate: Candidate
        with tqdm(
            total=len(io_futures),
            desc="Retrieving",
            smoothing=0.1,
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(io_futures):
                attributes = dict(future.result())
                candidate = future.candidate
                node_name = future.node_name

                logger.debug("Stemming {}".format(node_name))

                def chunks(l, n):
                    """Yield successive n-sized chunks from l."""
                    for i in range(0, len(l), n):
                        yield l[i : i + n]

                # logger.info("Getting new attributes...")
                # chunks = chunks(
                #     [
                #         (k, attributes[k])
                #         for k in set(attributes) - set(candidate._attributes)
                #     ],
                #     batch_size,
                # )
                # logger.info("Got new attributes...")
                chunks = chunks(list(attributes.items()), batch_size)

                names_list = []
                idx_list = []
                for chunk in chunks:
                    items = dict(chunk)
                    names_list.append(list(items.values()))
                    idx_list.append(list(items.keys()))

                if names_list:
                    logger.debug("Sending attributes...")
                    names_list = client.scatter(names_list)
                    while names_list:
                        logger.debug(
                            "{} items sent to stem.".format(len(chunk))
                        )
                        stem_future = client.submit(
                            textprocessing.stem, names_list.pop()
                        )
                        stem_future.candidate = candidate
                        stem_future.chunk = idx_list.pop()
                        stem_futures.append(stem_future)
                    logger.debug("Attributes sent.")

                candidate._attributes.update(attributes)
                candidate._node_attributes_map[node_name] = attributes.keys()

                logger.debug("Stemming {} sent".format(node_name))
                pbar.update(1)

        return stem_futures

    def _merge_all_attributes(self, stem_futures):
        with tqdm(
            total=len(stem_futures), desc="Stemming", dynamic_ncols=True
        ) as pbar:
            for future in d_as_completed(stem_futures):
                stemmed_names = future.result()
                candidate = future.candidate
                chunk = future.chunk

                stemmed_attributes = dict(zip(chunk, stemmed_names))
                candidate._stemmed_attributes.update(stemmed_attributes)
                pbar.update()

    def _recursive_combine_titles(self):
        G: Graph
        candidate: Candidate
        for i, candidate in enumerate(self._candidates):
            G = candidate.G
            order = nx.dfs_postorder_nodes(G)

            with tqdm(
                total=len(G),
                desc="Combining ({}/{})".format(i, len(self._candidates)),
                dynamic_ncols=True,
            ) as pbar:
                for node_name in order:
                    attributes = candidate._node_attributes_map[node_name]
                    joint_attributes = [attributes]
                    successors = G.successors(node_name)
                    for successor in successors:
                        # only works for tree!
                        joint_attributes.extend(
                            candidate._node_recursive_attributes_map[successor]
                        )

                        # length = sum(len(keys) for keys in
                        #              candidate._node_recursive_attributes_map[
                        #                  successor])
                        # logger.info("{} length.".format(length))

                    candidate._node_recursive_attributes_map[
                        node_name
                    ] = joint_attributes

                    # length = sum(len(keys) for keys in joint_attributes)
                    # logger.info("{} has {} recursively.".format(
                    #     node_name, length))
                    pbar.update()

    def prepare(self, client: Client, batch_size: int = 10000):
        logger.info("Preparing candidates...")

        if self._prepared:
            logger.info("Candidates already prepared.")
            return

        candidate: Candidate
        for candidate in self._candidates:
            if not candidate.is_clean():
                logger.info("Error! Candidates are not clean.")
                exit()

            candidate._stemmed_attributes = {}

        if config.CONNECTIONS <= 0:
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = config.CONNECTIONS
        with ThreadPoolExecutor(max_workers=num_cores) as thread_pool:
            io_futures = self._get_all_attributes(thread_pool)
            stem_futures = self._stem_all_attribute(
                client, io_futures, batch_size
            )
            self._merge_all_attributes(stem_futures)
        logger.debug("Thread pool has been closed.")

        # Convert to list to pickle
        candidate: Candidate
        for candidate in self._candidates:
            candidate._attr_loader = None
            for name, keys in candidate._node_attributes_map.items():
                candidate._node_attributes_map[name] = set(keys)

        self._recursive_combine_titles()
        self._prepared = True
        logger.info("Candidates prepared.")

    def run(
        self,
        client: Client,
        candidate_similarity_func: Union[Callable, Collection[Callable]],
    ):
        if isinstance(candidate_similarity_func, Callable):
            candidate_similarity_func = [candidate_similarity_func]

        if not self._prepared:
            self.prepare(client)

        candidate_pairs = itertools.combinations(self._candidates, 2)

        futures = []
        datas = []
        funcs = []
        for candidate_pair in candidate_pairs:
            for sim_func in candidate_similarity_func:
                datas.append(candidate_pair)
                funcs.append(sim_func)

        data_scatters = client.scatter(datas)

        for i in range(len(funcs)):
            sim_func = funcs[i]
            candidate_pair = datas[i]
            logger.info(
                "Running {} on {} <-> {}.".format(
                    sim_func.__name__,
                    candidate_pair[0].name,
                    candidate_pair[1].name,
                )
            )
            future = client.submit(sim_func, data_scatters[i])
            future.candidate_pair = candidate_pair
            future.sim_func = sim_func
            futures.append(future)

        return futures
