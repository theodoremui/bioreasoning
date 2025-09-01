import os
import tempfile

import pytest

from server.config import AppConfig, load_app_config


def test_load_default_env_only(monkeypatch):
    monkeypatch.delenv("SERVER_CONFIG_PATH", raising=False)
    monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("SERVER_PORT", "8338")
    cfg = load_app_config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8338


def test_load_from_yaml(tmp_path):
    yaml_content = """
server:
  host: 0.0.0.0
  port: 9000
  log_level: debug
agents:
  custom: bioagents.agents.graph_agent:GraphAgent
"""
    p = tmp_path / "server.yaml"
    p.write_text(yaml_content)
    cfg = load_app_config(str(p))
    assert cfg.log_level == "debug"
    assert isinstance(cfg.agent_registry, dict)
    assert cfg.agent_registry["custom"].endswith("GraphAgent")


