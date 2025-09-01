# Troubleshooting Guide: BioReasoning

This guide covers common issues and solutions for running and using the BioReasoning system.

---

## 1. Server/Client Startup Issues

**Q: The web app says it can't connect to the server.**
- **A:** Make sure you have started BOTH the Docs Server (`./scripts/run-server-docs.sh`) and the Knowledge Client (`./scripts/run-client.sh`) in separate terminals.

**Q: Port is already in use.**
- **A:**
  - Default ports: 8501 (client), 8131 (server)
  - Change the port in the scripts or configuration if needed.

**Q: Virtual environment not found or not activated.**
- **A:**
  - Run `python -m venv .venv` to create a virtual environment.
  - Activate with `source .venv/bin/activate` (Unix/macOS) or `.venv\Scripts\activate` (Windows).

---

## 2. File Upload & Processing

**Q: My file won't upload or process.**
- **A:**
  - Supported formats: PDF, DOCX, TXT, MD
  - Check file size (limit: 200MB per file)
  - Ensure the server is running and reachable

**Q: Mindmap or summary is missing.**
- **A:**
  - Check server logs for errors
  - Try re-uploading the file

---

## 3. Podcast Generation

**Q: Podcast generation fails or is slow.**
- **A:**
  - Ensure the MCP server is running
  - Check your OpenAI API key and quota
  - Try with a smaller or simpler document

**Q: Audio file does not play.**
- **A:**
  - Check browser compatibility
  - Ensure the audio file was generated (see server logs)

---

## 4. API Keys & Environment

**Q: I get an error about missing API keys.**
- **A:**
  - Create a `.env` file in the project root with `OPENAI_API_KEY=...`
  - Restart both server and client after setting the key

---

## 5. Getting Help

- Check the [README.md](../README.md) for setup and usage instructions
- Review the `docs/` folder for architecture and processing details
- For further help, contact Theodore Mui <theodoremui@gmail.com> 