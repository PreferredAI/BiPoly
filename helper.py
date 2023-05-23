import datetime
import logging
import os
import pickle
from time import sleep
from typing import Mapping

from dask.distributed import Client, LocalCluster
from pymongo import MongoClient
from tqdm import tqdm

import config

logger = logging.getLogger("categoryMatching")


def save_obj(obj, name):
    with open(os.path.join(config.OUTPUT_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(os.path.join(config.OUTPUT_DIR, name + ".pkl"), "rb") as f:
        return pickle.load(f)


def get_cluster():
    if config.DISTRIBUTED_LOCAL:
        return LocalCluster(
            n_workers=config.DISTRIBUTED_CORES,
            memory_limit=config.DISTRIBUTED_MEMORY,
            threads_per_worker=config.DISTRIBUTED_THREADS,
        )

    nworkers = max(
        1, config.DISTRIBUTED_WORKERS, config.DISTRIBUTED_WORKERS_MIN
    )
    nworkers_min = min(config.DISTRIBUTED_WORKERS_MIN, nworkers)
    if min(config.DISTRIBUTED_WORKERS_MIN, nworkers) <= 0:
        nworkers_min = nworkers

    preexec_commands = (
        "export PYTHONPATH=$PYTHONPATH:" + os.getcwd(),
        "export PYTHONPATH=$PYTHONPATH:/home/name/ibm/"
        + "ILOG/CPLEX_Studio128/cplex/python/3.6/x86-64_linux",
        ". /home/name/anaconda3/etc/profile.d/conda.sh",
        "conda activate cm",
    )

    from dask_jobqueue import SLURMCluster

    cluster = SLURMCluster(
        job_cpu=config.DISTRIBUTED_JOB_CPU,
        job_mem=config.DISTRIBUTED_JOB_MEM,
        walltime="2-0",
        cores=config.DISTRIBUTED_CORES * config.DISTRIBUTED_THREADS,
        processes=config.DISTRIBUTED_CORES,
        memory=config.DISTRIBUTED_MEMORY,
        local_directory=config.DISTRIBUTED_DIR,
        queue=config.DISTRIBUTED_JOB_QUEUE,
        env_extra=preexec_commands,
    )

    if nworkers_min == nworkers:
        cluster.scale(jobs=nworkers)
    else:
        cluster.adapt(
            minimum_jobs=nworkers_min,
            maximum_jobs=nworkers,
            startup_cost="500ms",
        )

    return cluster


def ensure_worker(process_pool: Client, nworkers_min: int = None):
    if not nworkers_min:
        nworkers = max(
            1,
            config.DISTRIBUTED_WORKERS * config.DISTRIBUTED_CORES,
            config.DISTRIBUTED_WORKERS_MIN * config.DISTRIBUTED_CORES,
        )
        nworkers_min = min(
            config.DISTRIBUTED_WORKERS_MIN * config.DISTRIBUTED_CORES, nworkers
        )
        if min(nworkers_min, nworkers) <= 0:
            nworkers_min = nworkers

    nworkers_s = len(process_pool.scheduler_info()["workers"])
    initial = min(nworkers_s, nworkers_min)
    with tqdm(
        total=nworkers_min,
        desc="Wating for workers",
        dynamic_ncols=True,
        initial=initial,
    ) as pbar:
        while process_pool.status == "running" and nworkers_s < nworkers_min:
            sleep(1.0)
            nworkers_c = len(process_pool.scheduler_info()["workers"])
            if nworkers_c > nworkers_min:
                pbar.n = nworkers_min
            elif nworkers_c > nworkers_s:
                pbar.update(nworkers_c - nworkers_s)
            nworkers_s = nworkers_c

    # logger.debug(process_pool.scheduler_info())


def broadcast_add_and_get(client: Client, mapping: Mapping, obj):
    if obj not in mapping:
        scatter = client.scatter((obj,), broadcast=True)[0]
        mapping[obj] = scatter
        return scatter
    else:
        return mapping[obj]


def _testing_attr_loader(node: dict, node_name: str):
    import random
    import string

    text = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    text2 = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    return {text: "test", text2: "separating"}


def isInt(s):
    try:
        int(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False


class Loader(object):
    def __init__(
        self,
        client: MongoClient,
        database: str,
        listing_collection: str,
        category_id_key: str = "CategoryIds",
        title_key: str = "Title",
        id_key: str = "_id",
        time=datetime.datetime(2020, 1, 1, 0, 0, 0, 0),
    ):
        self.client = client
        self.database = database
        self.listing_collection = listing_collection
        self.category_id_key = category_id_key
        self.title_key = title_key
        self.id_key = id_key
        self.time = time

    def query(self, node_id):
        return self.client[self.database][self.listing_collection].find(
            {
                "$and": [
                    {self.title_key: {"$exists": True}},
                    {"LastUpdated": {"$gt": self.time}},
                    {self.category_id_key: node_id},
                ]
            }
        )

    def get_immediate_listings(self, node: dict, node_name: str):
        logger.debug(
            "Getting immediate listings for '{}'...".format(node_name)
        )
        if "_id" not in node:
            return {}

        listings = self.query(node["_id"])
        immediate_listings = {}
        for listing in listings:
            listing_id = listing[self.id_key]
            title = listing[self.title_key]
            immediate_listings[listing_id] = title

        logger.debug(
            "Immediate listings for '{}' loaded, found {}.".format(
                node_name, len(immediate_listings)
            )
        )
        return immediate_listings

    def get_immediate_listings_sample(
        self,
        node: dict,
        node_name: str,
        even: bool = True,
        denominator: int = 3,
    ):
        logger.debug(
            "Getting immediate listings for '{}'...".format(node_name)
        )
        if "_id" not in node:
            return {}

        listings = self.query(node["_id"])

        immediate_listings = {}
        if even:
            for listing in listings:
                listing_id = listing[self.id_key]
                if isInt(listing["_id"]):
                    id_int = int(listing["_id"])
                else:
                    id_int = int(str(listing["_id"]), 16)
                if not id_int % denominator == 2:
                    continue
                title = listing[self.title_key]
                immediate_listings[listing_id] = title
        else:
            for listing in listings:
                listing_id = listing[self.id_key]
                if isInt(listing["_id"]):
                    id_int = int(listing["_id"])
                else:
                    id_int = int(str(listing["_id"]), 16)
                if id_int % denominator == 1:
                    continue
                title = listing[self.title_key]
                immediate_listings[listing_id] = title

        logger.debug(
            "Immediate listings for '{}' loaded, found {}.".format(
                node_name, len(immediate_listings)
            )
        )
        return immediate_listings
