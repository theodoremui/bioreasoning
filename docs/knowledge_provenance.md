# Knowledge Provenance & Citations in GraphRAG

### Overview
This document describes how the system captures, persists, and uses provenance (document, page, paragraph, snippet) for knowledge triplets so that answers can include citations.

### Key Concepts
- Triplet: (subject, relation, object) extracted from source text.
- Triplet Key: `subject|relation|object` stable identifier.
- Provenance: source metadata for a triplet: document id/title, file path, page, paragraph index, optional character span, and a short snippet; plus a deterministic `provenance_id`.

### Where Provenance Lives
1) On graph relationships (Neo4j):
   - `triplet_key`, `relationship_description`, `source_doc_id`, `source_doc_title`, `source_file_path`, `source_page`, `source_paragraph_index`, `char_start`, `char_end`, `source_snippet`, `extraction_node_id`, `provenance_id`.
2) In the cache file `data/nccn_communities.json`:
   - `community_info`: per community, list of edges with `{ detail, triplet_key }`.
   - `triplet_provenance`: map `triplet_key -> [provenance...]`.

### Ingestion & Extraction
During ingestion, `BaseNode.metadata` should include:
- `doc_id`, `doc_title`, `file_path`, `page_number`, `paragraph_index`, optional `char_start`, `char_end`.

`GraphRAGExtractor` builds `Relation` objects and propagates provenance from the node into relation properties, computes `triplet_key` and a `provenance_id`, and stores a snippet (first 500 chars or selected span).

### Community Build & Persistence
`GraphRAGStore` collects `community_info` with both `detail` and `triplet_key`. The `save_communities()`/`load_communities()` methods persist/restore `triplet_provenance` and normalize `community_info` to the new format.

### Query & Citations
`GraphRAGQueryEngine`:
- Ranks communities by query overlap, then ranks triplets.
- Builds a cited context with bullet points ending in `[n]` markers.
- Prompts the LLM to answer using only those bullets, preserving `[n]`.
- A separate citations list (title, page, paragraph, snippet, file path) maps `[n]` to provenance.

### Customization
- Swap ranking strategies or retrievers by injecting different implementations when constructing the query engine.
- Adjust limits: `max_summaries_to_use`, `max_triplets_to_use`.

### Backfill
For existing caches without `triplet_provenance`, query Neo4j relationships to populate it, then re-save the cache.

### Testing
- Unit tests validate extraction provenance attachment, cache round-trip, and citation-bearing answers.


