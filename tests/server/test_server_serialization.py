from bioagents.agents.common import AgentResponse, AgentRouteType
from server.api import _to_response


def test_to_response_basic():
    ar = AgentResponse(response_str="hello", citations=[], judgement="", route=AgentRouteType.REASONING)
    pr = _to_response(ar)
    assert pr.response == "hello"
    assert pr.route == "reasoning"


