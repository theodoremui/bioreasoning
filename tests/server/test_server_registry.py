import pytest

from server.registry import AgentRegistry


def test_default_agents_available():
    reg = AgentRegistry()
    got = reg.available()
    assert "router" in got and "graph" in got and "halo" in got


def test_get_known_agent():
    reg = AgentRegistry()
    agent = reg.get("router")
    assert hasattr(agent, "achat")


def test_unknown_agent_raises():
    reg = AgentRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


