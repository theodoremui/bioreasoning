"""
Graph Processing Constants

This module contains all constants, templates, and static data used throughout
the graph processing system. Centralizing constants follows DRY principles
and makes maintenance easier.

Author: Theodore Mui
Date: 2025-08-24
"""

from typing import Dict, List

# Knowledge Graph Extraction Template
KG_TRIPLET_EXTRACT_TEMPLATE = """
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

# Community Summary Generation Template
COMMUNITY_SUMMARY_SYSTEM_PROMPT = (
    "You are provided with a set of relationships from a knowledge graph, each represented as "
    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
    "relationships. The summary should include the names of the entities involved and a concise synthesis "
    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
    "integrates the information in a way that emphasizes the key aspects of the relationships."
)

# Query Engine System Prompts
ONCOLOGY_ASSISTANT_SYSTEM_PROMPT = (
    "You are an oncology research assistant. Answer the user's question using only the provided context. "
    "Be specific and actionable. If the context is insufficient, say so briefly."
)

CITED_CONTEXT_SYSTEM_PROMPT = (
    "You are an oncology research assistant. Use only the provided bullet points; include inline citation "
    "markers [n] that refer to those bullets."
)

ANSWER_AGGREGATION_SYSTEM_PROMPT = (
    "You combine multiple short medical answers into one concise response. "
    "Deduplicate, resolve conflicts conservatively, and keep it specific."
)

# Document Cleaning Patterns
DOCUMENT_CLEANUP_PATTERNS: List[str] = [
    "Printed by Stina Singel on 6/17/2025 3:05:05 AM. For personal use only. Not approved for distribution. Copyright © 2025 National Comprehensive Cancer Network, Inc., All Rights Reserved.\n\n"
]

# Medical Term Mappings for Enhanced Query Processing
MEDICAL_TERM_MAPPINGS: Dict[str, List[str]] = {
    "her2": ["HER2", "HER-2", "ERBB2"],
    "breast cancer": ["Breast Cancer", "breast cancer", "Breast Neoplasms"],
    "treatment": ["Treatment", "Therapy", "Therapeutic"],
    "patient": ["Patient", "Patients"],
}

# Medical Term Expansions for Community Ranking
MEDICAL_EXPANSIONS: Dict[str, List[str]] = {
    "her2": ["her2", "her-2", "erbb2", "human epidermal growth factor receptor 2"],
    "breast": ["breast", "mammary", "mammographic"],
    "cancer": ["cancer", "carcinoma", "neoplasm", "tumor", "malignancy"],
    "treatment": ["treatment", "therapy", "therapeutic", "treat", "intervention"],
    "patient": ["patient", "patients", "individual", "case"],
}

# Regular Expression Patterns
PATTERNS = {
    "json_extraction": r"\{.*\}",
    "triplet_pattern": r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$",
    "query_terms_basic": r"[a-zA-Z0-9+-]{2,}",
    "query_terms_detailed": r"[a-zA-Z0-9+-]{3,}",
    "word_tokens": r"\b\w+\b",
}

# Default Values
DEFAULTS = {
    "max_knowledge_triplets": 10,
    "num_workers": 4,
    "similarity_top_k": 20,
    "max_summaries_to_use": 6,
    "max_triplets_to_use": 20,
    "snippet_max_length": 500,
    "update_interval": 0.1,
}

# Scoring Weights for Community Ranking
RANKING_WEIGHTS = {
    "basic_score_weight": 3,
    "token_score_weight": 2,
    "expanded_score_weight": 1,
}

# File Extensions and Formats
SUPPORTED_FORMATS = {
    "documents": [".pdf", ".txt", ".md"],
    "cache": [".json"],
    "config": [".yaml", ".yml", ".json", ".env"],
}

# Error Messages
ERROR_MESSAGES = {
    "empty_message": "Message cannot be empty",
    "invalid_workers": "Number of workers must be positive",
    "invalid_paths": "Max paths per chunk must be positive",
    "invalid_cluster_size": "Max cluster size must be positive",
    "missing_api_key": "API key is required",
    "invalid_file_path": "File path is invalid or does not exist",
    "cache_validation_failed": "Cache validation failed",
    "no_graph_data": "No graph data available",
}

# Success Messages
SUCCESS_MESSAGES = {
    "communities_built": "✓ Communities built successfully",
    "cache_loaded": "✓ Cache loaded successfully",
    "document_processed": "✓ Document processed successfully",
    "extraction_complete": "✓ Knowledge extraction complete",
    "query_processed": "✓ Query processed successfully",
}
