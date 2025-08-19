from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import os
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

LLAMACLOUD_INDEX_NAME = os.getenv("LLAMACLOUD_INDEX_NAME")
LLAMACLOUD_PROJECT_NAME = os.getenv("LLAMACLOUD_PROJECT_NAME")
LLAMACLOUD_ORG_ID = os.getenv("LLAMACLOUD_ORG_ID")
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")

index = LlamaCloudIndex(
  name=LLAMACLOUD_INDEX_NAME,
  project_name=LLAMACLOUD_PROJECT_NAME,
  organization_id=LLAMACLOUD_ORG_ID,
  api_key=LLAMACLOUD_API_KEY,
)

if __name__ == "__main__":
    # query = "How best to treat a patient with a HER2+ breast cancer?"
    query = input("You > ")
    while query.strip() != "":
        try:
            nodes = index.as_retriever().retrieve(query)
            for node in nodes:
                print(f"==> {node.text}")
            response = index.as_query_engine().query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        query = input("You > ")
