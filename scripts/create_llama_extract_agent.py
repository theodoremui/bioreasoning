import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_cloud_services import LlamaExtract
from models.notebook import Notebook
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    conn = LlamaExtract(api_key=os.getenv("LLAMACLOUD_API_KEY"))
    try:
        agent = conn.create_agent(name="nccn_extract_agent", data_schema=Notebook)
        _id = agent.id
        with open(".env", "a") as f:
            f.write(f'\nEXTRACT_AGENT_ID="{_id}"')
        return 0
    except Exception as e:
        print(f"Your extraction agent already exists: {e}")
        return 1


if __name__ == "__main__":
    main()
