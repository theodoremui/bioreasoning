# Bio Reasoning Project

> Theodore Mui <theodoremui@gmail.com>
>
> July 12, 2025 4:13:28 PM

A comprehensive biomedical reasoning agent system that intelligently routes queries to specialized sub-agents for optimal response quality. The system combines conversational AI, real-time web search, biomedical research, document processing, mindmap creation, and podcast generation through a unified interface.

---

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Feature Walkthrough](#feature-walkthrough)
- [Troubleshooting](#troubleshooting)
- [Development & Advanced Usage](#development--advanced-usage)
- [Documentation](#documentation)

---

## Overview

The Bio Reasoning Project implements an intelligent agent orchestration system designed for biomedical research, document analysis, and general knowledge queries. The core **BioConciergeAgent** acts as a smart router that analyzes incoming queries and delegates them to the most appropriate specialized sub-agent or processing pipeline.

### Key Features
- **Intelligent Query Routing**: Automatically determines the best sub-agent for each query
- **Document Upload & Processing**: Upload PDFs, DOCX, TXT, and more for instant analysis
- **Mindmap Generation**: Visualize document structure and key concepts
- **Podcast Generation**: Create custom audio conversations from document content
- **Multi-Modal Expertise**: Handles biomedical research, web search, and conversational queries
- **Citation Support**: Provides source citations for research and web-based responses
- **Streamlit Interface**: User-friendly web interface for interaction
- **Model Selection**: Support for multiple OpenAI models (GPT-4o, GPT-4.1 variants)

---

## System Architecture

The system is composed of two main components:
- **Knowledge Server (MCP)**: Handles document processing, mindmap, and podcast generation
- **Knowledge Client (Streamlit)**: User-facing web interface for uploads, queries, and results

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

---

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Unix/macOS or WSL (for bash scripts)

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd bioreasoning
```
2. **Install dependencies**
```bash
uv sync
```
3. **Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
4. **Set up environment variables**
   - Create a `.env` file in the project root:
     ```bash
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - (Optional) Set database and server variables as needed.

---

## Running the System

### 1. Start the Knowledge Server (MCP)
This server handles all document, mindmap, and podcast processing.

```bash
./run-knowledge-server.sh
```
- Activates the virtual environment if needed
- Runs the MCP server (`bioagents/mcp/knowledge_server.py`)

### 2. Start the Knowledge Client (Streamlit)
This is the user-facing web app.

```bash
./run-knowledge-client.sh
```
- Activates the virtual environment if needed
- Installs Streamlit if missing
- Runs the Streamlit app (`frontend/app.py`)

> **Note:** Both the server and client must be running for full functionality.

---

## Feature Walkthrough

### 1. Document Upload & Processing
- Upload PDF, DOCX, TXT, or MD files via the web interface
- The system processes the document and displays:
  - **Summary**
  - **Bullet Points**
  - **FAQ** (expandable)
  - **Mindmap** (interactive HTML)

**Example:**
1. Go to the "Documents" page
2. Upload `example.pdf`
3. View the summary, bullet points, FAQ, and mindmap

### 2. Mindmap Generation
- Automatically generated for each processed document
- Visualizes document structure and key concepts

**Example:**
- After uploading a document, scroll to the "Mind Map" section to explore the visualization

### 3. Podcast Generation
- Customize style, tone, audience, and speaker roles
- Click "Generate In-Depth Conversation" to create a podcast from the document
- Listen to the generated audio directly in the browser

**Example:**
- After processing a document, expand the "Podcast Configuration" panel
- Choose "interview" style, "friendly" tone, and set speakers
- Click the button and listen to the result

---

## Troubleshooting

- **Server/Client not running**: Ensure both `run-knowledge-server.sh` and `run-knowledge-client.sh` are running in separate terminals
- **Port conflicts**: Default ports are 8501 (client) and 8131 (server); change if needed
- **Missing dependencies**: Run `uv sync` and ensure your virtual environment is activated
- **API key issues**: Check your `.env` file and ensure `OPENAI_API_KEY` is set
- **File upload issues**: Supported formats are PDF, DOCX, TXT, and MD

---

## Development & Advanced Usage

### Project Structure
```text
bioreasoning/
├── bioagents/              # Core agent library
│   ├── agents/            # Agent implementations
│   ├── mcp/               # MCP server implementation
│   └── models/            # Data models and LLM interface
├── frontend/              # Streamlit frontend
│   ├── app.py             # Main entry point
│   └── pages/             # Multi-page app modules
├── run-knowledge-server.sh
├── run-knowledge-client.sh
├── pyproject.toml         # Dependencies and project config
└── README.md
```

### Adding New Document Processors or Features
- Add new processors to `bioagents/mcp/` and update the server logic
- Add new UI components to `frontend/pages/`
- Update the architecture diagrams in `docs/` as needed

---

## Documentation

- See the `docs/` folder for:
  - **Architecture**: Detailed diagrams and flowcharts
  - **Processing**: Step-by-step document, mindmap, and podcast flows
  - **Troubleshooting**: Common issues and solutions

---

**Contact**: Theodore Mui <theodoremui@gmail.com>

For questions, issues, or contributions, please reach out via email.
