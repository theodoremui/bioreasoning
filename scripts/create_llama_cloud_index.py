import os
from dotenv import load_dotenv, find_dotenv
from cli.embedding_app import EmbeddingSetupApp

from llama_cloud import (
    PipelineTransformConfig_Advanced,
    AdvancedModeTransformConfigChunkingConfig_Sentence,
    AdvancedModeTransformConfigSegmentationConfig_Page,
    PipelineCreate,
)
from llama_cloud.client import LlamaCloud


def main():
    """
    Create a new Llama Cloud index with the given embedding configuration.
    """
    load_dotenv(find_dotenv())

    LLAMACLOUD_INDEX_NAME = os.getenv("LLAMACLOUD_INDEX_NAME")
    
    client = LlamaCloud(token=os.getenv("LLAMACLOUD_API_KEY"))

    # Run the embedding setup app to get the embedding configuration
    # This prompts the user to select an embedding provider and configure the embedding model
    app = EmbeddingSetupApp()
    embedding_config = app.run()

    if embedding_config:
        segm_config = AdvancedModeTransformConfigSegmentationConfig_Page(mode="page")
        chunk_config = AdvancedModeTransformConfigChunkingConfig_Sentence(
            chunk_size=1024,
            chunk_overlap=200,
            separator="<whitespace>",
            paragraph_separator="\n\n\n",
            mode="sentence",
        )

        transform_config = PipelineTransformConfig_Advanced(
            segmentation_config=segm_config,
            chunking_config=chunk_config,
            mode="advanced",
        )

        pipeline_request = PipelineCreate(
            name=LLAMACLOUD_INDEX_NAME,
            embedding_config=embedding_config,
            transform_config=transform_config,
        )

        pipeline = client.pipelines.upsert_pipeline(request=pipeline_request)

        with open(".env", "a") as f:
            f.write(f'\nLLAMACLOUD_PIPELINE_ID="{pipeline.id}"')

        return 0
    else:
        print("No embedding configuration provided")
        return 1


if __name__ == "__main__":
    main()
