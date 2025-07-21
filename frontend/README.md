# BioReasoning Multi-Page Streamlit Application

A well-structured, multi-page Streamlit application for the BioReasoning Agent, built following SOLID principles and best practices.

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

```
frontend/
├── app.py                     # Main entry point
├── components/                # Reusable UI components
│   ├── __init__.py
│   ├── model_selector.py      # Model selection component
│   └── session_manager.py     # Session state management
├── pages/                     # Application pages
│   ├── 1_Chat.py             # Chat interface page
│   └── 2_Documents.py        # Document upload page
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── file_utils.py         # File handling utilities
└── README.md                 # This file
```

## Features

### 🏠 Main Page (app.py)
- Application navigation hub
- Common layout and branding
- Navigation instructions

### 💬 Chat Page
- Interactive chat interface with the BioReasoning Agent
- Model selection (GPT-4.1 Mini, GPT-4.1 Nano, GPT-4.1, GPT-4o)
- Chat history management
- Citation display
- Clear chat functionality
- Chat statistics

### 📁 Documents Page
- File upload functionality
- Support for multiple file formats (PDF, TXT, DOCX, CSV, XLSX, MD)
- File management (view, analyze, delete)
- Progress tracking during uploads
- File metadata display

## Design Principles

### SOLID Principles
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Components are extensible without modification
- **Liskov Substitution**: Components can be replaced with implementations
- **Interface Segregation**: Focused interfaces for specific needs
- **Dependency Inversion**: High-level modules don't depend on low-level details

### DRY (Don't Repeat Yourself)
- Shared components for common UI elements
- Utility functions for file operations
- Centralized session management

### Clear Documentation
- Comprehensive docstrings for all classes and methods
- Type hints for better code maintainability
- Inline comments for complex logic

## Running the Application

### Prerequisites
1. Install required dependencies (see main project requirements)
2. Ensure `OPENAI_API_KEY` is set in your environment or `.env` file

### Starting the Application
```bash
# From the project root directory
cd frontend
streamlit run app.py
```

### Navigation
Once the application is running:
1. The main page provides navigation instructions
2. Use the sidebar navigation to switch between pages:
   - **Chat**: Interact with the BioReasoning Agent
   - **Documents**: Upload and manage files

## File Structure Details

### Components
- `ModelSelector`: Reusable component for LLM model selection
- `SessionManager`: Centralized session state management

### Pages
- `1_Chat.py`: Chat interface with the BioReasoning Agent
- `2_Documents.py`: Document upload and management interface

### Utilities
- `FileUtils`: Common file operations and utilities

## Data Storage

### Uploaded Files
- Files are stored in `data/uploads/` directory
- Automatic filename conflict resolution
- File metadata tracking in session state

### Session State
- Chat messages
- Uploaded file information
- LLM client and agent instances

## Future Enhancements

### Planned Features
- File content preview
- Integration between documents and chat
- File analysis with the BioReasoning Agent
- Export/import functionality
- Advanced file search and filtering

### Extensibility
The modular architecture allows for easy addition of:
- New pages
- Additional components
- Enhanced file processing
- Integration with external services

## Development Guidelines

### Adding New Pages
1. Create a new file in `pages/` with appropriate numbering (e.g., `3_NewPage.py`)
2. Follow the existing class-based structure
3. Use shared components where appropriate
4. Update navigation instructions in main app

### Adding New Components
1. Create component classes in `components/`
2. Follow the single responsibility principle
3. Add comprehensive documentation
4. Update `__init__.py` imports

### Code Quality
- Use type hints for all function parameters and return values
- Write comprehensive docstrings
- Follow PEP 8 style guidelines
- Add error handling where appropriate

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from the `frontend/` directory
2. **Session State Issues**: Clear browser cache/cookies if state becomes inconsistent
3. **File Upload Failures**: Check directory permissions for `data/uploads/`

### Debug Mode
Run with debug mode for development:
```bash
streamlit run app.py --logger.level debug
``` 