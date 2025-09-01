from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    log_level: str = "info"
    # Optional path to a YAML file; can be set via env SERVER_CONFIG_PATH
    config_path: Optional[str] = None
    # Mapping of agent name -> dotted path to class (module:Class)
    agent_registry: Dict[str, str] | None = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_config(merged[k], v)  # type: ignore[index]
        else:
            merged[k] = v
    return merged


def load_app_config(explicit_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from env or YAML.

    Precedence:
    1) Explicit path argument (tests may pass this)
    2) Env var SERVER_CONFIG_PATH
    3) Default ./config/server.yaml if exists
    Env overrides (if set): SERVER_HOST, SERVER_PORT, SERVER_LOG_LEVEL
    """
    # Discover YAML config path
    env_path = os.getenv("SERVER_CONFIG_PATH")
    candidate_path = explicit_path or env_path
    if not candidate_path:
        default_path = os.path.abspath(os.path.join(os.getcwd(), "config", "server.yaml"))
        candidate_path = default_path if os.path.exists(default_path) else None

    yaml_data = _load_yaml(candidate_path) if candidate_path else {}

    # Base config from YAML
    server_cfg = (yaml_data.get("server") or {}) if isinstance(yaml_data, dict) else {}
    host = str(server_cfg.get("host", "0.0.0.0"))
    port = int(server_cfg.get("port", 9000))
    log_level = str(server_cfg.get("log_level", "info"))
    agent_registry = yaml_data.get("agents") if isinstance(yaml_data, dict) else None
    if agent_registry is not None and not isinstance(agent_registry, dict):
        agent_registry = None

    # Env overrides
    host = os.getenv("SERVER_HOST", host)
    log_level = os.getenv("SERVER_LOG_LEVEL", log_level)
    if os.getenv("SERVER_PORT") is not None:
        try:
            port = int(os.getenv("SERVER_PORT") or port)
        except ValueError:
            pass

    return AppConfig(
        host=host,
        port=port,
        log_level=log_level,
        config_path=candidate_path,
        agent_registry=agent_registry,
    )


