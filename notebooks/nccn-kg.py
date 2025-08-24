from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import asyncio
import json
import re
import hashlib
import warnings
import os

import networkx as nx

# Global constants
NCCN_COMMUNITIES_CACHE_FILE = "data/nccn_communities.json"

# Suppress deprecation warning from Setuptools about pkg_resources used in graspologic
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)
from collections import defaultdict

from typing import Any, List, Callable, Optional, Union, Dict, Tuple

from llama_index.core import PropertyGraphIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.llms import ChatMessage, LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import TransformComponent, BaseNode

from bioagents.utils.text_utils import make_contextual_snippet, make_title_and_snippet

try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
except Exception:
    # Lightweight fallback for testing environments without llama-index neo4j package
    class Neo4jPropertyGraphStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        # Minimal API surface used by GraphRAGStore/DummyGraphStore in tests
        def get_triplets(self):
            return []


llm = OpenAI(model="gpt-4.1-mini")

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

-An Output Example-
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    },
    {
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    },
    {
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    },
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }
  ]
}

-Real Data-
######################
text: {text}
######################
output:"""


class GraphRAGStore(Neo4jPropertyGraphStore):
    """Property-graph store with community building, caching, and provenance support.

    Extends the Neo4j store to:
    - Build NetworkX graphs and detect communities
    - Generate LLM-based community summaries
    - Persist and load a cache file with community data and triplet provenance
    - Provide helpers to backfill provenance from graph relations
    """

    community_summary = {}
    entity_info = None
    max_cluster_size = 5
    # Added fields to capture full, reproducible graph community state
    cluster_assignments: Dict[str, int] | None = None  # node -> cluster id
    community_info: Dict[int, List[str]] | None = (
        None  # cluster id -> relationship detail strings
    )
    algorithm_metadata: Dict[str, Any] | None = None
    # Provenance map: triplet_key -> list of provenance dicts
    triplet_provenance: Dict[str, List[Dict[str, Any]]] | None = None
    
    @property
    def supports_vector_queries(self) -> bool:
        """Return whether this store supports vector queries."""
        return False

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = OpenAI(model="gpt-4.1-mini").chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        # Try using graspologic hierarchical_leiden; fallback to connected components
        try:
            from graspologic.partition import hierarchical_leiden  # type: ignore

            community_hierarchical_clusters = hierarchical_leiden(
                nx_graph, max_cluster_size=self.max_cluster_size
            )
            algorithm_used = "hierarchical_leiden"
            library_used = "graspologic.partition"
        except Exception:
            # Fallback: simple connected components as clusters
            components = list(nx.connected_components(nx_graph))

            class _ClusterItem:
                def __init__(self, node, cluster):
                    self.node = node
                    self.cluster = cluster

            items = []
            for idx, comp in enumerate(components):
                for node in comp:
                    items.append(_ClusterItem(node, idx))
            community_hierarchical_clusters = items
            algorithm_used = "connected_components"
            library_used = "networkx"
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        # Persist additional state for reproducibility
        # Flatten cluster assignments
        try:
            assignments: Dict[str, int] = {}
            for item in community_hierarchical_clusters:
                node = getattr(item, "node", None)
                cluster_id = getattr(item, "cluster", None)
                if node is not None and cluster_id is not None:
                    assignments[str(node)] = int(cluster_id)
            self.cluster_assignments = assignments
        except Exception:
            self.cluster_assignments = None

        self.community_info = community_info
        self.algorithm_metadata = {
            "algorithm": algorithm_used,
            "library": library_used,
            "parameters": {"max_cluster_size": self.max_cluster_size},
            "nx_graph_nodes": nx_graph.number_of_nodes(),
            "nx_graph_edges": nx_graph.number_of_edges(),
        }

        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph.

        Returns:
            nx.Graph: Nodes are entity names. Edges carry 'relationship', 'description', and 'triplet_key'.
        """
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            triplet_key = f"{relation.source_id}|{relation.label}|{relation.target_id}"
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
                triplet_key=triplet_key,
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Aggregate per-community relationship details.

        Collect information for each node based on its community assignment,
        allowing entities to belong to multiple clusters.

        Returns:
            tuple(dict, dict): (entity_info, community_info)
                - entity_info maps node -> list[cluster_id]
                - community_info maps cluster_id -> list[{detail, triplet_key}]
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(
                        {
                            "detail": detail,
                            "triplet_key": edge_data.get("triplet_key"),
                        }
                    )

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        """Generate and store LLM summaries for each community."""
        for community_id, details in community_info.items():
            # Support both legacy list[str] and new list[dict]
            if details and isinstance(details[0], dict):
                text_lines = [d.get("detail", "") for d in details]
            else:
                text_lines = [str(d) for d in details]
            details_text = "\n".join(text_lines) + "."
            self.community_summary[community_id] = self.generate_community_summary(
                details_text
            )

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

    def _compute_graph_signature(self) -> str:
        """Compute a stable signature of the current graph triplets for cache validation."""
        triplets = self.get_triplets()
        items = []
        for entity1, relation, entity2 in triplets:
            relation_desc = ""
            try:
                relation_desc = relation.properties.get("relationship_description", "")
            except Exception:
                relation_desc = ""
            items.append((entity1.name, relation.label, entity2.name, relation_desc))
        # Ensure deterministic ordering
        items.sort()
        payload = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def save_communities(self, filepath: str) -> None:
        """Persist full community/graph-derived state to a JSON file.

        Notes:
        - Nodes and relationships (and their properties) live in Neo4j via the base store.
        - This file captures the rest needed to reproduce community-based answering without recomputation.
        """
        # Ensure destination directory exists
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "graph_signature": self._compute_graph_signature(),
            "max_cluster_size": self.max_cluster_size,
            "entity_info": self.entity_info or {},
            "community_summary": self.community_summary or {},
            # Extended state for reproducibility
            "cluster_assignments": self.cluster_assignments or {},
            "community_info": self.community_info or {},
            "triplet_provenance": self.triplet_provenance or {},
            "algorithm_metadata": self.algorithm_metadata
            or {
                "algorithm": "hierarchical_leiden",
                "parameters": {"max_cluster_size": self.max_cluster_size},
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_communities(self, filepath: str, validate_signature: bool = True) -> bool:
        """Load community data from a JSON file. Returns True if loaded successfully.

        If validate_signature is True, the on-disk signature must match the current
        graph signature (derived from current triplets); otherwise, the cache is ignored.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False

        if validate_signature:
            try:
                current_sig = self._compute_graph_signature()
            except Exception:
                current_sig = None
            if not current_sig or data.get("graph_signature") != current_sig:
                return False

        self.entity_info = data.get("entity_info") or {}
        # Convert community_summary string keys to integers for consistency
        raw_cs = data.get("community_summary") or {}
        norm_cs = {}
        for cid, summary in raw_cs.items():
            try:
                int_cid = int(cid)
            except (ValueError, TypeError):
                int_cid = cid  # Keep original if conversion fails
            norm_cs[int_cid] = summary
        self.community_summary = norm_cs
        # Populate extended state if present
        self.cluster_assignments = data.get("cluster_assignments") or {}
        # Normalize community_info: ensure list of dicts with 'detail' and optional 'triplet_key'
        # Also convert string keys to integers for consistency
        raw_ci = data.get("community_info") or {}
        norm_ci: Dict[int, List[Dict[str, Any]]] = {}
        for cid, items in raw_ci.items():
            # Convert string keys to integers
            try:
                int_cid = int(cid)
            except (ValueError, TypeError):
                int_cid = cid  # Keep original if conversion fails
            
            new_items: List[Dict[str, Any]] = []
            for it in items:
                if isinstance(it, dict):
                    # already new format
                    new_items.append(
                        {
                            "detail": it.get("detail", ""),
                            "triplet_key": it.get("triplet_key"),
                        }
                    )
                else:
                    new_items.append({"detail": str(it), "triplet_key": None})
            norm_ci[int_cid] = new_items
        self.community_info = norm_ci
        self.triplet_provenance = data.get("triplet_provenance") or {}
        self.algorithm_metadata = data.get("algorithm_metadata") or None
        return True

    def has_graph_data(self) -> bool:
        """Return True if the backing graph store has any triplets."""
        try:
            triplets = self.get_triplets()
        except Exception:
            return False
        return bool(triplets)

    def ensure_communities(
        self,
        persist_path: str | None = None,
        validate_signature: bool = True,
        prefer_cache_when_graph_empty: bool = True,
    ) -> None:
        """Ensure communities are available, optionally using a persisted cache.

        If persist_path is provided, attempt to load; if that fails, build and then save.
        """
        if persist_path:
            loaded = False
            # If graph is empty and we prefer cache, load without strict validation
            if prefer_cache_when_graph_empty and not self.has_graph_data():
                loaded = self.load_communities(persist_path, validate_signature=False)
            else:
                loaded = self.load_communities(
                    persist_path, validate_signature=validate_signature
                )
            if loaded:
                # Auto-backfill provenance if missing but graph has data
                if (
                    (not getattr(self, "triplet_provenance", None) or not self.triplet_provenance)
                    and self.has_graph_data()
                ):
                    self.build_triplet_provenance_from_graph()
                    try:
                        self.save_communities(persist_path)
                    except Exception:
                        pass
                return
        self.build_communities()
        if persist_path:
            self.save_communities(persist_path)

    def build_triplet_provenance_from_graph(self) -> None:
        """Backfill triplet_provenance by reading relation properties from the graph store.

        Iterates current triplets and collects provenance-relevant fields from each relation's properties.
        Intended for use when a cache file lacks triplet_provenance.
        """
        provenance: Dict[str, List[Dict[str, Any]]] = {}
        try:
            triplets = self.get_triplets()
        except Exception:
            triplets = []
        for e1, relation, e2 in triplets:
            props_r = getattr(relation, "properties", {}) or {}
            props_e1 = getattr(e1, "properties", {}) or {}
            props_e2 = getattr(e2, "properties", {}) or {}
            triplet_key = (
                props_r.get("triplet_key")
                or f"{relation.source_id}|{relation.label}|{relation.target_id}"
            )
            # Prefer relation properties; fallback to entity properties
            def pick(key: str):
                return (
                    props_r.get(key)
                    or props_e1.get(key)
                    or props_e2.get(key)
                )

            prov = {
                "source_doc_id": pick("doc_id") or pick("source_doc_id"),
                "source_doc_title": pick("doc_title")
                or pick("source_doc_title"),
                "source_file_path": pick("file_path")
                or pick("source_file_path"),
                "source_page": pick("page_number") or pick("source_page"),
                "source_paragraph_index": pick("paragraph_index")
                or pick("source_paragraph_index"),
                "char_start": pick("char_start"),
                "char_end": pick("char_end"),
                "source_snippet": make_contextual_snippet(
                    props_r.get("source_snippet")
                    or props_r.get("relationship_description")
                    or props_e1.get("entity_description")
                    or props_e2.get("entity_description")
                    or "",
                    "",
                    max_length=500
                ),
                "extraction_node_id": props_r.get("extraction_node_id"),
                "provenance_id": props_r.get("provenance_id"),
            }
            if not prov["provenance_id"]:
                payload = json.dumps(
                    {**prov, "triplet_key": triplet_key},
                    default=str,
                    ensure_ascii=False,
                )
                prov["provenance_id"] = hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()
            provenance.setdefault(triplet_key, [])
            # Deduplicate by provenance_id
            if all(
                p.get("provenance_id") != prov["provenance_id"]
                for p in provenance[triplet_key]
            ):
                provenance[triplet_key].append(prov)
        self.triplet_provenance = provenance


class GraphRAGExtractor(TransformComponent):
    """Extract triples from text nodes, producing entities/relations with provenance.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node, attaching provenance to relations.

        Returns:
            BaseNode: The same node with KG nodes/relations added into metadata.
        """
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata_base = node.metadata.copy()
        # Baseline provenance from node metadata
        base_prov = {
            "source_doc_id": relation_metadata_base.get("doc_id"),
            "source_doc_title": relation_metadata_base.get("doc_title"),
            "source_file_path": relation_metadata_base.get("file_path"),
            "source_page": relation_metadata_base.get("page_number"),
            "source_paragraph_index": relation_metadata_base.get("paragraph_index"),
            "char_start": relation_metadata_base.get("char_start"),
            "char_end": relation_metadata_base.get("char_end"),
            "extraction_node_id": getattr(node, "node_id", None),
        }
        # Create a short snippet window
        try:
            full_text = node.get_content(metadata_mode="llm") or ""
        except Exception:
            full_text = ""
        snippet = make_contextual_snippet(full_text, "", max_length=500)
        # if isinstance(base_prov.get("char_start"), int) and isinstance(
        #     base_prov.get("char_end"), int
        # ):
        #     s, e = base_prov["char_start"], base_prov["char_end"]
        #     if 0 <= s < e <= len(full_text):
        #         snippet = full_text[s:e]
        # snippet = (snippet or full_text)[:500]

        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata = relation_metadata_base.copy()
            relation_metadata["relationship_description"] = description
            triplet_key = f"{subj}|{rel}|{obj}"
            # Compute provenance id
            prov_id_payload = json.dumps(
                {**base_prov, "triplet_key": triplet_key},
                default=str,
                ensure_ascii=False,
            )
            prov_id = hashlib.sha256(prov_id_payload.encode("utf-8")).hexdigest()
            relation_metadata.update(
                {
                    "triplet_key": triplet_key,
                    "source_doc_id": base_prov["source_doc_id"],
                    "source_doc_title": base_prov["source_doc_title"],
                    "source_file_path": base_prov["source_file_path"],
                    "source_page": base_prov["source_page"],
                    "source_paragraph_index": base_prov["source_paragraph_index"],
                    "char_start": base_prov["char_start"],
                    "char_end": base_prov["char_end"],
                    "extraction_node_id": base_prov["extraction_node_id"],
                    "source_snippet": snippet,
                    "provenance_id": prov_id,
                }
            )
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )


def parse_fn(response_str: str) -> Any:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities = []
    relationships = []
    if not match:
        return entities, relationships
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
        return entities, relationships
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)


class GraphRAGQueryEngine(CustomQueryEngine):
    """Query engine over community summaries and provenance-bearing triplets.

    Strategy:
    - Resolve entities via retriever or fallback vocabulary scan
    - Rank communities by keyword overlap, then rank triplets in those communities
    - Build cited evidence bullets and instruct LLM to answer with [n] markers
    - Append a citations section to the response
    """

    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20
    max_summaries_to_use: int = 6
    max_triplets_to_use: int = 20
    # capture last computed citations for programmatic consumers
    last_citations: List[Dict[str, Any]] = []

    def custom_query(self, query_str: str) -> str:
        """Answer a query using community summaries and triplet provenance, with citations."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_summaries = self.graph_store.get_community_summaries()

        # Fallbacks when no entities/communities were resolved via retrieval
        if not community_ids:
            # If we have any communities, consider all of them
            community_ids = list(community_summaries.keys())

        # Ensure community_ids are integers to match the keys in community_summaries
        # After the load_communities fix, all keys are integers
        community_ids = [int(cid) if isinstance(cid, str) and cid.isdigit() else cid for cid in community_ids]

        # Rank summaries by simple keyword overlap with the query and select top-k
        ranked = self._rank_communities_by_query_overlap(
            community_summaries, query_str, community_ids
        )
        chosen_ids = [cid for cid, _ in ranked[: self.max_summaries_to_use]]

        # If ranking produced nothing (e.g., type mismatch), default to first K summaries
        if not chosen_ids and community_summaries:
            chosen_ids = list(community_summaries.keys())[: self.max_summaries_to_use]

        # Build a candidate triplet set and collect detail-only fallbacks
        candidate_triplets: List[Tuple[str, str]] = []  # (triplet_key, detail)
        detail_only_blocks: List[str] = []
        if isinstance(self.graph_store.community_info, dict):
            for cid in chosen_ids:
                # Ensure cid is an integer for consistent dictionary access
                int_cid = int(cid) if isinstance(cid, str) and cid.isdigit() else cid
                items = self.graph_store.community_info.get(int_cid, [])
                for it in items:
                    if isinstance(it, dict):
                        detail = it.get("detail", "")
                        tk = it.get("triplet_key")
                        if tk:
                            candidate_triplets.append((tk, detail))
                        elif detail:
                            detail_only_blocks.append(f"- {detail}")
                    else:
                        detail_only_blocks.append(f"- {str(it)}")

        # Rank triplets by overlap and select top M
        ranked_triplets = self._rank_triplets_by_query_overlap(
            candidate_triplets, query_str
        )
        chosen_triplets = ranked_triplets[: self.max_triplets_to_use]

        # Prepare context blocks with citations
        context_blocks, citations = self._prepare_cited_triplets(chosen_triplets)

        # Generate answers with fallbacks
        community_answers: List[str] = []
        if context_blocks:
            community_answers = [
                self.generate_answer_from_cited_context(
                    "\n".join(context_blocks), query_str
                )
            ]
        elif detail_only_blocks:
            community_answers = [
                self.generate_answer_from_cited_context(
                    "\n".join(detail_only_blocks), query_str
                )
            ]
            citations = []
        else:
            top_summary_ids = chosen_ids[: self.max_summaries_to_use]
            community_answers = [
                self.generate_answer_from_summary(community_summaries[cid], query_str)
                for cid in top_summary_ids
                if cid in community_summaries
            ]
            citations = []

        final_answer = self.aggregate_answers(community_answers)
        # Append citations section for human readability
        if "citations" in locals() and citations:
            lines = ["\nCitations:"]
            for c in citations:
                title = c.get("title") or c.get("doc_id") or "Source"
                page = c.get("page")
                para = c.get("paragraph")
                # Clean any existing uncleaned snippets from cached data
                raw_snippet = c.get("snippet") or ""
                snippet = make_contextual_snippet(raw_snippet, "", max_length=500) if raw_snippet else ""
                loc = []
                if page is not None:
                    loc.append(f"p. {page}")
                if para is not None:
                    loc.append(f"¶ {para}")
                loc_str = f" ({', '.join(loc)})" if loc else ""
                lines.append(f"[{c['id']}] {title}{loc_str}: {snippet}")
            final_answer = final_answer.rstrip() + "\n" + "\n".join(lines)
        # Save for programmatic consumers
        self.last_citations = citations if "citations" in locals() else []
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        """Resolve entities relevant to the query.

        Strategy:
        1) Try retriever-based extraction from indexed path triples.
        2) If nothing found, fallback to vocabulary scanning over graph entities.
        """
        try:
            nodes_retrieved = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            ).retrieve(query_str)
        except Exception:
            nodes_retrieved = []

        entities = set()
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                subject = match[0]
                obj = match[2]
                entities.add(subject)
                entities.add(obj)

        if entities:
            return list(entities)

        # Fallback: scan graph vocabulary (entity names) and select those mentioned in query
        try:
            triplets = self.graph_store.get_triplets()
        except Exception:
            triplets = []

        vocab = set()
        for e1, _, e2 in triplets:
            try:
                vocab.add(str(e1.name))
            except Exception:
                pass
            try:
                vocab.add(str(e2.name))
            except Exception:
                pass

        q_lower = query_str.lower()
        fallback_entities = [name for name in vocab if name.lower() in q_lower]

        return fallback_entities

    def _rank_communities_by_query_overlap(
        self,
        community_summaries: Dict[int, str],
        query_str: str,
        candidate_ids: List[int],
    ) -> List[tuple[int, int]]:
        """Rank community summaries by simple keyword overlap with the query."""
        query_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9+-]{3,}", query_str)}
        scored: List[tuple[int, int]] = []
        for cid in candidate_ids:
            summ = community_summaries.get(cid) or ""
            text_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9+-]{3,}", summ)}
            score = len(query_terms & text_terms)
            scored.append((cid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_triplets_by_query_overlap(
        self, triplets: List[Tuple[str, str]], query_str: str
    ) -> List[Tuple[str, str]]:
        """Rank triplets by overlap between query terms and edge 'detail' text."""
        query_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9+-]{3,}", query_str)}
        scored: List[Tuple[int, Tuple[str, str]]] = []
        for tk, detail in triplets:
            terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9+-]{3,}", detail)}
            scored.append((len(query_terms & terms), (tk, detail)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def _prepare_cited_triplets(
        self, triplets: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Build bullet points with [n] and collect citations metadata.

        Returns:
            tuple(list[str], list[dict]): (context bullets, citations list)
        """
        citations: List[Dict[str, Any]] = []
        lines: List[str] = []
        seen_prov = set()
        for idx, (triplet_key, detail) in enumerate(triplets, start=1):
            prov_list = []
            if getattr(self.graph_store, "triplet_provenance", None):
                prov_list = self.graph_store.triplet_provenance.get(triplet_key, [])
            # choose one best provenance (first available)
            prov = prov_list[0] if prov_list else {}
            citation_id = idx
            lines.append(f"- {detail} [{citation_id}]")
            sig = (
                prov.get("provenance_id")
                or f"{triplet_key}|{prov.get('source_doc_id')}|{prov.get('source_page')}|{prov.get('source_paragraph_index')}"
            )
            if sig in seen_prov:
                continue
            seen_prov.add(sig)
            citations.append(
                {
                    "id": citation_id,
                    "triplet_key": triplet_key,
                    "doc_id": prov.get("source_doc_id"),
                    "title": prov.get("source_doc_title"),
                    "page": prov.get("source_page"),
                    "paragraph": prov.get("source_paragraph_index"),
                    "file_path": prov.get("source_file_path"),
                    "snippet": prov.get("source_snippet"),
                    "provenance_id": prov.get("provenance_id"),
                }
            )
        return lines, citations

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        """(Legacy) Answer from a single summary."""
        system_msg = (
            "You are an oncology research assistant. Answer the user's question using only the provided context. "
            "Be specific and actionable. If the context is insufficient, say so briefly."
        )
        user_msg = f"Context:\n{community_summary}\n\nQuestion: {query}\n\nAnswer strictly from the context above."
        messages = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def generate_answer_from_cited_context(self, cited_context: str, query: str) -> str:
        """Answer using cited triplets context; require inline [n] markers in the response."""
        system_msg = "You are an oncology research assistant. Use only the provided bullet points; include inline citation markers [n] that refer to those bullets."
        user_msg = f"Evidence bullets (with citations):\n{cited_context}\n\nQuestion: {query}\n\nWrite a concise answer that includes inline [n] markers referencing the relevant bullets."
        messages = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers: List[str]) -> str:
        """Aggregate individual community answers into a final, coherent response."""
        if not community_answers:
            return "No relevant knowledge found in cached community summaries."
        system_msg = (
            "You combine multiple short medical answers into one concise response. "
            "Deduplicate, resolve conflicts conservatively, and keep it specific."
        )
        answers_bulleted = "\n- " + "\n- ".join(
            a.strip() for a in community_answers if a.strip()
        )
        user_msg = f"Combine these into one coherent answer:\n{answers_bulleted}"
        messages = [
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response


if __name__ == "__main__":
    graph_store = GraphRAGStore(
        username="neo4j",
        password="Salesforce1",
        url="bolt://localhost:7687",  # database="nccn"
    )
    # Persist under project data/ directory using absolute path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    persist_path = os.path.join(project_root, NCCN_COMMUNITIES_CACHE_FILE)
    print(f"loading graph store from {persist_path}", flush=True)
    graph_store.ensure_communities(persist_path=persist_path)

    index = PropertyGraphIndex(
        nodes=[],
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=20,
    )
    query = "How best to treat breast cancer for patients with HER2?"
    while query.strip() != "":
        response = query_engine.query(query)
        print(response.response)
        query = input("Enter your query: ")
    print(response.response)
