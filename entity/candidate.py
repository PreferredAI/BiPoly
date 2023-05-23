import itertools
import os
from typing import Callable, Collection, Mapping

import networkx as nx
from networkx.classes.graph import Graph

from helper import logger


class Candidate(object):
    def __init__(self, name: str, G: Graph, attr_loader: Callable = None):

        # Check if DAG
        logger.info("Verifying '{}' graph...".format(name))
        if not nx.is_directed_acyclic_graph(G):
            logger.info("Error! Graph not DAG.")
            exit()

        self.name = name
        self.G = G
        self._attr_loader = attr_loader
        self._node_attributes_map = {}
        self._node_recursive_attributes_map = {}
        self._attributes = {}

    def is_clean(self):
        return not self._node_attributes_map and not self._attributes

    def _load_attributes(self, node_name, should_cache: bool = True):
        if not self._attr_loader:
            logger.info("Warning! No attribute loader present.")
            attributes = None
        elif isinstance(self._attr_loader, Callable):
            node_attributes = self.G.nodes[node_name]
            attributes = self._attr_loader(node_attributes, node_name)
        else:
            logger.info("Error! Unable to extract attributes.")
            exit()

        attributes_map: Mapping
        if attributes:
            if isinstance(attributes, Mapping):
                attributes_map = attributes
            elif isinstance(attributes, Collection):
                attributes_keys = [hash(attribute) for attribute in attributes]
                attributes_map = dict(zip(attributes_keys, attributes))
            else:
                logger.info(
                    "Error! Unable to extract attributes. \
                Return type must be Collection or Mapping."
                )
                exit()
        else:
            attributes_map = {}

        if should_cache:
            self._attributes.update(attributes_map)
            self._node_attributes_map[node_name] = attributes_map.keys()

        return attributes_map

    def get_attributes(self, node_name: str, should_cache: bool = True):
        if node_name not in self.G:
            logger.info("No such node found.")
            return None

        if node_name not in self._node_attributes_map:
            return self._load_attributes(node_name, should_cache).items()

        return {
            k: self._attributes[k]
            for k in self._node_attributes_map[node_name]
        }.items()

    def get_attributes_keys(self, node_name: str, should_cache: bool = True):
        if node_name not in self.G:
            logger.info("No such node found.")
            return None

        if node_name not in self._node_attributes_map:
            return self._load_attributes(node_name, should_cache).keys()

        return self._node_attributes_map[node_name]

    def _load_recursive_attributes_keys(
        self, node_name: str, should_cache: bool = True
    ):
        G: Graph
        G = self.G

        successors = nx.dfs_preorder_nodes(G, node_name)

        attribute_keys = []
        attribute_keys.append(
            self.get_attributes_keys(node_name, should_cache)
        )
        for successor in successors:
            attribute_keys.append(
                self.get_attributes_keys(successor, should_cache)
            )

        if should_cache:
            self._node_recursive_attributes_map[node_name] = attribute_keys

        return attribute_keys

    def get_recursive_attributes(
        self, node_name: str, should_cache: bool = True
    ):
        if node_name not in self.G:
            logger.info("No such node found.")
            return None

        if node_name not in self._node_recursive_attributes_map:
            keys = self._load_recursive_attributes_keys(
                node_name, should_cache
            )
        else:
            keys = self._node_recursive_attributes_map[node_name]

        key_iter = itertools.chain.from_iterable(keys)
        return iter([(k, self._attributes[k]) for k in key_iter])

    def get_recursive_attributes_keys(
        self, node_name: str, should_cache: bool = True
    ):
        if node_name not in self.G:
            logger.info("No such node found.")
            return None

        if node_name not in self._node_recursive_attributes_map:
            keys = self._load_recursive_attributes_keys(
                node_name, should_cache
            )
        else:
            keys = self._node_recursive_attributes_map[node_name]

        return itertools.chain.from_iterable(keys)

    def write_dot(self, path: str = None):
        if not path:
            path = os.path.join(os.getcwd(), "output/")

        filepath = os.path.join(path, self.name + "-raw.dot")

        logger.info("Writing raw dot...")
        nx.nx_pydot.write_dot(self.G, filepath)
        logger.info("Writing raw dot complete.")
