# System Architecture: BioReasoning

## High-Level Overview

The BioReasoning system consists of a Streamlit client and multiple MCP servers:
- **Knowledge Client (Streamlit Web App)**: User interface for chat, uploads, results, and podcast generation.
- **BioMCP Server**: Biomedical tools and research endpoints (variants, PubMed, biomedical datasets).
- **DocMCP Server**: Document-centric tools backed by a LlamaCloud index (RAG, mindmaps, podcast generation).

## Component Diagram

```mermaid
flowchart LR
    User([User])
    subgraph Client
        UI([Streamlit Knowledge Client])
        Orchestrator([Orchestrator Selector: halo/router/...])
    end
    subgraph Servers
        subgraph BioMCP
            BioMCPServer([BioMCP Server])
        end
        subgraph DocMCP
            DocMCPServer([DocMCP Server])
        end
    end

    User --> UI
    UI --> Orchestrator
    Orchestrator --> BioMCPServer
    Orchestrator --> DocMCPServer
```

## Data Flow: Document Processing

1. **User uploads a document** via the Streamlit client.
2. **Client sends the file** to the MCP server.
3. **MCP server processes the document**:
   - Extracts summary, bullet points, FAQ
   - Generates a mindmap (HTML)
   - Prepares content for podcast generation
4. **Results are sent back** to the client and displayed in the UI.

## Data Flow: Mindmap Generation (DocMCP)

```mermaid
sequenceDiagram
    participant User
    participant Client as Streamlit Client
    participant Server as DocMCP Server
    participant MindMap as Mindmap Generator

    User->>Client: Upload document
    Client->>Server: Send document
    Server->>MindMap: Generate mindmap
    MindMap-->>Server: HTML mindmap
    Server-->>Client: Mindmap HTML
    Client->>User: Display mindmap
```

## Data Flow: Podcast Generation (DocMCP)

```mermaid
sequenceDiagram
    participant User
    participant Client as Streamlit Client
    participant Server as DocMCP Server
```

## Chat Orchestration

The chat page (`frontend/pages/1_Chat.py`) provides an Orchestrator selector. The `SessionManager` maps selections to agents (`BioHALOAgent`, `BioRouterAgent`, `GraphAgent`, `LlamaRAGAgent`, `LlamaMCPAgent`, `BioMCPAgent`, `WebReasoningAgent`, `ChitChatAgent`).

```mermaid
flowchart TD
    UI([Chat UI]) --> Sel[Select Orchestrator]
    Sel -->|halo| HALO[BioHALOAgent]
    Sel -->|router| Router[BioRouterAgent]
    Sel -->|graph| Graph[GraphAgent]
    Sel -->|llamarag| RAG[LlamaRAGAgent]
    Sel -->|llamamcp| LMCP[LlamaMCPAgent]
    Sel -->|biomcp| Bio[BioMCPAgent]
    Sel -->|web| Web[WebReasoningAgent]
    Sel -->|chitchat| Chat[ChitChatAgent]

    HALO --> DocMCP
    HALO --> BioMCP
    RAG --> DocMCP
    Graph --> DocMCP
    LMCP --> DocMCP
    Bio --> BioMCP
    Web --> Internet[[Web]]
```
    participant Podcast as Podcast Generator

    User->>Client: Configure podcast options
    Client->>Server: Request podcast generation
    Server->>Podcast: Generate podcast audio
    Podcast-->>Server: Audio file
    Server-->>Client: Audio file
    Client->>User: Play podcast
```

## Key Design Principles
- **Separation of Concerns**: UI and processing are decoupled for scalability and maintainability.
- **Extensibility**: New document processors, mindmap types, or podcast styles can be added independently.
- **Interactive Feedback**: Users receive immediate feedback and can customize podcast generation.

---

For more details, see [processing.md](processing.md) and [troubleshooting.md](troubleshooting.md). 