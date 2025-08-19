from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingConfig:
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    region: Optional[str] = None
    key_id: Optional[str] = None
    embedding_config: Optional[object] = None
