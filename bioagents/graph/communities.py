"""
Community Building

This module provides community detection and building functionality for graph structures,
following the Single Responsibility Principle and Strategy Pattern for different algorithms.

Author: Theodore Mui
Date: 2025-08-24
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx

from .interfaces import ICommunityBuilder


class HierarchicalLeidenBuilder(ICommunityBuilder):
    """Community builder using hierarchical Leiden algorithm.

    Uses the graspologic library for advanced community detection with
    hierarchical clustering capabilities.
    """

    def __init__(self, max_cluster_size: int = 5):
        """Initialize with clustering parameters.

        Args:
            max_cluster_size: Maximum size for clusters
        """
        self.max_cluster_size = max_cluster_size
        self._algorithm_metadata = {}

    def build_communities(
        self, graph
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
        """Build communities using hierarchical Leiden algorithm.

        Args:
            graph: NetworkX graph

        Returns:
            Tuple of (entity_info, community_info)
        """
        try:
            from graspologic.partition import hierarchical_leiden

            community_clusters = hierarchical_leiden(
                graph, max_cluster_size=self.max_cluster_size
            )

            self._algorithm_metadata = {
                "algorithm": "hierarchical_leiden",
                "library": "graspologic.partition",
                "parameters": {"max_cluster_size": self.max_cluster_size},
                "nx_graph_nodes": graph.number_of_nodes(),
                "nx_graph_edges": graph.number_of_edges(),
            }

            return self._collect_community_info(graph, community_clusters)

        except ImportError:
            # Fallback to connected components if graspologic not available
            fallback_builder = ConnectedComponentsBuilder(self.max_cluster_size)
            return fallback_builder.build_communities(graph)

    def get_algorithm_metadata(self) -> Dict[str, Any]:
        """Get metadata about the algorithm used."""
        return self._algorithm_metadata.copy()

    def _collect_community_info(
        self, graph, clusters
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
        """Collect community information from clustering results.

        Args:
            graph: NetworkX graph
            clusters: Clustering results

        Returns:
            Tuple of (entity_info, community_info)
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = getattr(item, "node", item)
            cluster_id = getattr(item, "cluster", 0)

            # Update entity_info
            entity_info[node].add(cluster_id)

            # Collect relationship details for this community
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(
                        {
                            "detail": detail,
                            "triplet_key": edge_data.get("triplet_key"),
                        }
                    )

        # Convert sets to lists for serialization
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)


class ConnectedComponentsBuilder(ICommunityBuilder):
    """Community builder using connected components.

    Fallback algorithm that uses NetworkX connected components
    for community detection when advanced algorithms are unavailable.
    """

    def __init__(self, max_cluster_size: int = 5):
        """Initialize with clustering parameters.

        Args:
            max_cluster_size: Maximum size for clusters (informational only)
        """
        self.max_cluster_size = max_cluster_size
        self._algorithm_metadata = {}

    def build_communities(
        self, graph
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
        """Build communities using connected components.

        Args:
            graph: NetworkX graph

        Returns:
            Tuple of (entity_info, community_info)
        """
        components = list(nx.connected_components(graph))

        # Create cluster items compatible with hierarchical format
        class ClusterItem:
            def __init__(self, node, cluster):
                self.node = node
                self.cluster = cluster

        cluster_items = []
        for idx, component in enumerate(components):
            for node in component:
                cluster_items.append(ClusterItem(node, idx))

        self._algorithm_metadata = {
            "algorithm": "connected_components",
            "library": "networkx",
            "parameters": {"max_cluster_size": self.max_cluster_size},
            "nx_graph_nodes": graph.number_of_nodes(),
            "nx_graph_edges": graph.number_of_edges(),
            "num_components": len(components),
        }

        return self._collect_community_info(graph, cluster_items)

    def get_algorithm_metadata(self) -> Dict[str, Any]:
        """Get metadata about the algorithm used."""
        return self._algorithm_metadata.copy()

    def _collect_community_info(
        self, graph, clusters
    ) -> Tuple[Dict[str, List[int]], Dict[int, List[Dict[str, Any]]]]:
        """Collect community information from clustering results."""
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            # Collect relationship details for this community
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(
                        {
                            "detail": detail,
                            "triplet_key": edge_data.get("triplet_key"),
                        }
                    )

        # Convert sets to lists for serialization
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)


class CommunityBuilderFactory:
    """Factory for creating community builders.

    Implements the Factory Pattern to provide different community building
    strategies based on availability and requirements.
    """

    @staticmethod
    def create_builder(
        algorithm: str = "auto", max_cluster_size: int = 5
    ) -> ICommunityBuilder:
        """Create a community builder instance.

        Args:
            algorithm: Algorithm to use ("hierarchical_leiden", "connected_components", "auto")
            max_cluster_size: Maximum cluster size

        Returns:
            Community builder instance

        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm == "auto":
            # Try hierarchical Leiden first, fallback to connected components
            try:
                import graspologic.partition

                return HierarchicalLeidenBuilder(max_cluster_size)
            except ImportError:
                return ConnectedComponentsBuilder(max_cluster_size)

        elif algorithm == "hierarchical_leiden":
            return HierarchicalLeidenBuilder(max_cluster_size)

        elif algorithm == "connected_components":
            return ConnectedComponentsBuilder(max_cluster_size)

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    @staticmethod
    def get_available_algorithms() -> List[str]:
        """Get list of available algorithms.

        Returns:
            List of available algorithm names
        """
        algorithms = ["connected_components"]

        try:
            import graspologic.partition

            algorithms.append("hierarchical_leiden")
        except ImportError:
            pass

        return algorithms
