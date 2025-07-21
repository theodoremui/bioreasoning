# System Architecture: BioReasoning

## High-Level Overview

The BioReasoning system is split into two main components:
- **Knowledge Client (Streamlit Web App)**: User interface for uploading documents, viewing results, and generating podcasts.
- **Knowledge Server (MCP)**: Backend server responsible for document processing, mindmap generation, and podcast creation.

## Component Diagram

```mermaid
flowchart TD
    User([User])
    subgraph Client
        StreamlitApp([Streamlit Knowledge Client])
    end
    subgraph Server
        MCPServer([MCP Knowledge Server])
        DocProc([Document Processor])
        MindMap([Mindmap Generator])
        Podcast([Podcast Generator])
    end

    User --> StreamlitApp
    StreamlitApp --"Upload/Query"--> MCPServer
    MCPServer --"Process Document"--> DocProc
    DocProc --"Results"--> MCPServer
    MCPServer --"Generate Mindmap"--> MindMap
    MindMap --"HTML Mindmap"--> MCPServer
    MCPServer --"Generate Podcast"--> Podcast
    Podcast --"Audio File"--> MCPServer
    MCPServer --"Results, Mindmap, Podcast"--> StreamlitApp
```

## Data Flow: Document Processing

1. **User uploads a document** via the Streamlit client.
2. **Client sends the file** to the MCP server.
3. **MCP server processes the document**:
   - Extracts summary, bullet points, FAQ
   - Generates a mindmap (HTML)
   - Prepares content for podcast generation
4. **Results are sent back** to the client and displayed in the UI.

## Data Flow: Mindmap Generation

```mermaid
sequenceDiagram
    participant User
    participant Client as Streamlit Client
    participant Server as MCP Server
    participant MindMap as Mindmap Generator

    User->>Client: Upload document
    Client->>Server: Send document
    Server->>MindMap: Generate mindmap
    MindMap-->>Server: HTML mindmap
    Server-->>Client: Mindmap HTML
    Client->>User: Display mindmap
```

## Data Flow: Podcast Generation

```mermaid
sequenceDiagram
    participant User
    participant Client as Streamlit Client
    participant Server as MCP Server
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