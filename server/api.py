from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bioagents.agents.common import AgentResponse
from server.config import AppConfig, load_app_config
from server.registry import AgentRegistry
from server.schemas import ChatRequest, ChatResponse
from server.service import ChatService


def _to_response(agent_resp: AgentResponse) -> ChatResponse:
    # Convert AgentResponse dataclass to pydantic model
    citations = []
    for c in getattr(agent_resp, "citations", []) or []:
        citations.append({
            "url": getattr(c, "url", ""),
            "title": getattr(c, "title", ""),
            "snippet": getattr(c, "snippet", ""),
            "source": getattr(c, "source", ""),
            "file_name": getattr(c, "file_name", ""),
            "start_page_label": getattr(c, "start_page_label", ""),
            "end_page_label": getattr(c, "end_page_label", ""),
            "score": getattr(c, "score", 0.0),
            "text": getattr(c, "text", ""),
        })
    route = getattr(agent_resp, "route", None)
    route_value = getattr(route, "value", str(route)) if route is not None else "reasoning"
    return ChatResponse(
        response=getattr(agent_resp, "response_str", ""),
        citations=citations,
        judgement=getattr(agent_resp, "judgement", ""),
        route=route_value,
    )


def create_app(
    config: AppConfig | None = None,
    registry: AgentRegistry | None = None,
    service: ChatService | None = None,
) -> FastAPI:
    cfg = config or load_app_config()
    registry = registry or AgentRegistry(overrides=cfg.agent_registry)
    service = service or ChatService(registry)

    app = FastAPI(title="BioReasoning API", version="0.1.0")

    # CORS - permissive by default for dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/{agent_name}/chat", response_model=ChatResponse)
    async def chat(agent_name: str, req: ChatRequest) -> ChatResponse:
        try:
            agent_resp = await service.achat(agent_name, req.query)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return _to_response(agent_resp)

    @app.post("/{agent_name}/simplechat")
    async def simplechat(agent_name: str, req: ChatRequest) -> str:
        try:
            response_str = await service.simple_achat(agent_name, req.query)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return response_str

    return app


