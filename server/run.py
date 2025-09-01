from __future__ import annotations

import os

import uvicorn

from server.api import create_app
from server.config import load_app_config


def main() -> None:
    cfg = load_app_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)


if __name__ == "__main__":
    main()


