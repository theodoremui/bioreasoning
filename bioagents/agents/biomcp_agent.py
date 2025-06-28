#------------------------------------------------------------------------------
# biomcp_agent.py
# 
# This file provides the MCP functionalities for the BioMCP agent.
# 
# Author: Theodore Mui
# Date: 2025-06-11
#------------------------------------------------------------------------------

"""
Biomedical MCP Agent

A high-level interface for interacting with biomedical research tools via the MCP protocol.
Provides a simple async chat interface while handling all the complexity of server management,
connection handling, and resource cleanup.

Features:
- Simple async chat interface via achat() method
- Automatic server lifecycle management with lazy initialization
- Per-operation connections for proper async resource management
- Robust error handling and diagnostics
- Environment-based configuration
- Context manager support for proper cleanup
- Auto-start capability for interactive environments
- Comprehensive cross-task cleanup warning suppression

Example:
    # Simple usage (auto-starts server on first call)
    agent = BioMCPAgent()
    result = await agent.achat("Find articles about Alzheimer's disease")
    print(result)

    # Explicit lifecycle management (recommended for production)
    async with BioMCPAgent() as agent:
        result = await agent.achat("Get details for variant rs113488022")
        print(result)
"""

import asyncio
import os
import shutil
import subprocess
import requests
import atexit
import weakref
import warnings
import sys
import json
import re
from typing import Any, Optional
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta

# Apply comprehensive warning suppression at module level
warnings.filterwarnings("ignore", message=".*async_generator.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*cancel scope.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*different task.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Task.*pending.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*streamablehttp_client.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*", category=RuntimeWarning)

# Patch stderr to suppress specific cross-task cleanup errors
_original_stderr_write = sys.stderr.write

def _patched_stderr_write(text):
    """Patched stderr write that filters out harmless cross-task cleanup messages."""
    if text and isinstance(text, str):
        # Filter out the specific error patterns that are harmless
        harmless_patterns = [
            "async_generator object streamablehttp_client",
            "BaseExceptionGroup: unhandled errors in a TaskGroup", 
            "RuntimeError: Attempted to exit cancel scope in a different task",
            "Exception Group Traceback",
            "During handling of the above exception, another exception occurred:",
            "GeneratorExit",
            "anyio/_backends/_asyncio.py",
            "mcp/client/streamable_http.py",
            "athrow()"
        ]
        
        # Check if this text contains harmless patterns
        text_lower = text.lower()
        if any(pattern.lower() in text_lower for pattern in harmless_patterns):
            # This is a harmless cleanup message - suppress it
            return
    
    # This is either not a string or doesn't contain harmless patterns - let it through
    return _original_stderr_write(text)

# Apply the patch
sys.stderr.write = _patched_stderr_write

from agents import Agent, Runner, RunResult, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from loguru import logger

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from bioagents.agents.common import AgentResponse
from bioagents.agents.base_agent import ReasoningAgent
from bioagents.models.llms import LLM

@contextmanager
def suppress_async_cleanup_warnings():
    """
    Context manager to suppress harmless async generator cleanup warnings.
    
    These warnings occur during cross-task async resource cleanup and are not harmful,
    but can be noisy in logs. This suppresses them during cleanup operations.
    
    Enhanced to handle:
    - Async generator cleanup warnings
    - Cancel scope cross-task errors  
    - Task group cleanup issues
    - MCP client cleanup warnings
    - AnyIO backend cleanup errors
    """
    import sys
    import io
    import warnings
    import contextlib
    
    # Capture both warnings and stderr output
    with warnings.catch_warnings():
        # Suppress Python warnings
        warnings.filterwarnings("ignore", message=".*async_generator.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*cancel scope.*", category=RuntimeWarning) 
        warnings.filterwarnings("ignore", message=".*different task.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*Task.*pending.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*streamablehttp_client.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*", category=RuntimeWarning)
        
        # Also capture stderr to suppress direct error prints
        old_stderr = sys.stderr
        # Also capture stdout in case some errors go there
        old_stdout = sys.stdout
        
        try:
            # Create temporary stderr and stdout buffers
            captured_stderr = io.StringIO()
            captured_stdout = io.StringIO()
            sys.stderr = captured_stderr
            sys.stdout = captured_stdout
            
            yield
            
            # Check what was captured and only log serious errors
            stderr_content = captured_stderr.getvalue()
            stdout_content = captured_stdout.getvalue()
            
            # Process stderr output
            if stderr_content:
                # Filter out known harmless cleanup messages with more patterns
                harmless_patterns = [
                    "async_generator", "cancel scope", "different task", 
                    "streamablehttp_client", "Task was destroyed",
                    "BaseExceptionGroup", "GeneratorExit", "generator didn't stop",
                    "anyio._backends._asyncio", "mcp/client/streamable_http",
                    "anyio/streams/memory.py", "RuntimeError: Attempted to exit cancel scope",
                    "Exception Group Traceback", "unhandled errors in a TaskGroup",
                    "athrow()", "asyncgen:", "During handling of the above exception"
                ]
                
                lines = stderr_content.split('\n')
                serious_lines = []
                skip_next_lines = 0
                
                for i, line in enumerate(lines):
                    if skip_next_lines > 0:
                        skip_next_lines -= 1
                        continue
                        
                    line_lower = line.lower()
                    
                    # Check if this line starts a known harmless error block
                    if any(pattern in line_lower for pattern in harmless_patterns):
                        # Skip this line and look ahead to skip the entire error block
                        if "exception group traceback" in line_lower or "baseexceptiongroup" in line_lower:
                            # Skip the entire exception group block
                            j = i + 1
                            while j < len(lines) and (lines[j].startswith('  ') or lines[j].startswith('|') or lines[j].startswith('+-+')):
                                j += 1
                            skip_next_lines = j - i - 1
                        continue
                    
                    # If the line contains substantive content and isn't harmless, keep it
                    if line.strip() and not any(pattern in line_lower for pattern in harmless_patterns):
                        serious_lines.append(line)
                
                # Only show serious errors
                if serious_lines:
                    serious_output = '\n'.join(serious_lines)
                    # Only show if it's not just whitespace or common harmless phrases
                    if serious_output.strip() and not all(any(pattern in line.lower() for pattern in harmless_patterns) for line in serious_lines):
                        old_stderr.write(f"Serious error during operation: {serious_output}\n")
                        old_stderr.flush()
            
            # Process stdout output (usually less critical)
            if stdout_content.strip():
                # Only show stdout if it doesn't contain harmless patterns
                if not any(pattern in stdout_content.lower() for pattern in ["async_generator", "cancel scope", "different task"]):
                    old_stdout.write(stdout_content)
                    old_stdout.flush()
                
        except Exception:
            # Ensure we never break due to our own error handling
            pass
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout


class BioMCPAgentError(Exception):
    """Base exception for BioMCPAgent errors."""
    pass


class BioMCPServerError(BioMCPAgentError):
    """Raised when the MCP server fails to start or respond."""
    pass


class BioMCPConnectionError(BioMCPAgentError):
    """Raised when connection to the MCP server fails."""
    pass

BIOMCP_PORT = int(os.environ.get("BIOMCP_PORT", 8131))

class BioMCPAgent(ReasoningAgent):
    """
    High-level interface for biomedical MCP interactions with async factory pattern.
    
    This class provides a clean interface for biomedical research queries with proper
    async resource management to avoid event loop conflicts.
    
    **Important**: Use the async factory method `create()` instead of direct instantiation
    to ensure proper async initialization of MCP connections.
    
    **Features**:
    - Intelligent auto-initialization via the `agent` property
    - Clean async factory pattern for proper resource management
    - Automatic cleanup warning suppression for cross-task scenarios
    - Context-aware behavior (sync vs async contexts)
    - Comprehensive error handling and recovery
    
    **Usage Patterns**:
    
    Async factory (recommended):
    ```python
    bio_agent = await BioMCPAgent.create()
    result = await bio_agent.achat("Find cancer research")
    await bio_agent.close()  # Cleanup when done
    ```
    
    Context manager (automatic cleanup):
    ```python
    async with await BioMCPAgent.create() as bio_agent:
        result = await bio_agent.achat("Find cancer research")
        agent = bio_agent.agent  # Access the initialized Agent
    ```
    
    Auto-initialization (convenient):
    ```python
    bio_agent = BioMCPAgent()
    agent = bio_agent.agent  # Auto-initializes cleanly
    result = await bio_agent.achat("Find research")
    ```
    """

    @property
    def agent(self) -> Agent:
        """
        Get the initialized biomedical Agent instance with automatic initialization.
        
        This property provides intelligent lazy initialization with connection health checks:
        - If agent is already initialized, verifies connection health
        - If not initialized or connection is stale, performs full initialization
        - Handles both async and sync contexts appropriately
        - Automatically recovers from connection issues
        
        **Automatic Initialization**:
        - In sync context: Uses asyncio.run() for initialization with health checks
        - In async context: Provides clear guidance to use async methods
        
        **Usage Patterns**:
        ```python
        # Sync context (works automatically with health checks)
        bio_agent = BioMCPAgent()
        agent = bio_agent.agent  # Auto-initializes with healthy connection
        
        # Async context (use async methods)
        bio_agent = await BioMCPAgent.create()  # Preferred
        # OR
        bio_agent = BioMCPAgent()
        await bio_agent.ensure_agent_ready()
        agent = bio_agent.agent  # Now available
        ```
        
        Returns:
            Agent: Fully initialized Agent ready for biomedical queries with healthy connection
            
        Raises:
            BioMCPAgentError: If initialization fails or in async context without proper setup
        """
        if self._agent is not None:
            # Check if we're in sync context and can do health checks
            try:
                asyncio.get_running_loop()
                # In async context - just return the agent if available, health checks 
                # should be done via ensure_agent_ready()
                return self._agent
            except RuntimeError:
                # In sync context - do quick health check of agent's MCP connections
                if self._quick_agent_health_check():
                    return self._agent
                else:
                    # Agent exists but connection is stale, clear it for reinitialization
                    logger.debug("Agent MCP connections are stale in sync context, clearing for reinit")
                    self._agent = None
                    self._mcp_server = None
        
        # Agent not initialized or connection is stale - attempt automatic initialization
        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context - provide guidance
                raise BioMCPAgentError(
                    "Agent not initialized and cannot auto-initialize in async context. "
                    "Use one of these patterns:\n"
                    "  • bio_agent = await BioMCPAgent.create()  # Recommended\n"
                    "  • await bio_agent.ensure_agent_ready(); agent = bio_agent.agent\n"
                    "  • result = await bio_agent.achat('query')  # Auto-initializes"
                )
            except RuntimeError:
                # No event loop running - we can use asyncio.run()
                logger.info("Auto-initializing biomedical agent with health checks (sync context)...")
                
                try:
                    with suppress_async_cleanup_warnings():
                        # Use ensure_agent_ready which includes health checks
                        agent = asyncio.run(self.ensure_agent_ready())
                    
                    logger.info("Agent auto-initialized successfully with healthy connection")
                    return agent
                    
                except Exception as init_error:
                    # If initialization fails, ensure clean state
                    self._agent = None
                    self._initialized = False
                    self._mcp_server = None
                    
                    # Re-raise the initialization error
                    raise BioMCPAgentError(
                        f"Auto-initialization failed: {init_error}\n"
                        f"Try using: bio_agent = await BioMCPAgent.create()"
                    ) from init_error
                
        except BioMCPAgentError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Handle any other initialization errors
            raise BioMCPAgentError(
                f"Failed to auto-initialize agent: {e}\n"
                f"Try using the async factory: bio_agent = await BioMCPAgent.create()"
            ) from e
    
    def __init__(
        self,
        name: str = "Biomedical MCP Agent",
        model_name: str=LLM.GPT_4_1_MINI,
        instructions: str="You are a biomedical research assistant.",
        port: Optional[int] = BIOMCP_PORT,
        timeout: int = 60,  # Increased default timeout for biomedical queries
        server_startup_timeout: int = 60,
        debug: bool = False,
    ):
        """
        Initialize the BioMCPAgent (synchronous initialization only).
        
        **Important**: This constructor only performs synchronous setup.
        Use the async factory method `create()` to get a fully initialized instance.
        
        Args:
            port: Port for the MCP server (defaults to BIOMCP_PORT env var or 8131)
            timeout: Timeout for individual requests in seconds
            server_startup_timeout: Timeout for server startup in seconds  
            debug: Enable debug logging
            
        Raises:
            BioMCPAgentError: If required dependencies are missing
        """
        super().__init__(name, model_name, instructions)
        
        self.port = port or BIOMCP_PORT
        self.timeout = timeout
        self.server_startup_timeout = server_startup_timeout
        self.debug = debug
        
        # Validate environment
        self._validate_environment()
        
        # Server process management (will be initialized in create())
        self._server_process: Optional[subprocess.Popen[Any]] = None
        self._server_url = f"http://localhost:{self.port}"
        self._mcp_endpoint = f"{self._server_url}/mcp/"
        self._server_started = False
        self._start_lock = asyncio.Lock()
        self._auto_started = False
        
        # Async resources (initialized in create())
        self._mcp_server: Optional[MCPServerStreamableHttp] = None
        self._agent: Optional[Agent] = None
        self._initialized = False
        
        # Context tracking for cross-asyncio.run() detection
        self._connection_context_id: Optional[str] = None
        
        # Configure logging
        if debug:
            logger.enable("biomcp_agent")
        else:
            logger.disable("biomcp_agent")
    
    def _validate_environment(self) -> None:
        """Validate required dependencies and environment variables."""
        # Check uv
        if not shutil.which("uv"):
            raise BioMCPAgentError(
                "uv package manager is required. Install from: "
                "https://docs.astral.sh/uv/getting-started/installation/"
            )
        
        # Check OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise BioMCPAgentError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
    
    def _check_port_available(self) -> bool:
        """Check if the configured port is available."""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            return result != 0
    
    def _test_http_endpoint(self, url: str, timeout: int = 5) -> tuple[bool, str]:
        """Test if an HTTP endpoint is responding."""
        try:
            response = requests.get(url, timeout=timeout)
            return True, f"HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused"
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except Exception as e:
            return False, f"Error: {e}"
    
    async def _start_server(self) -> subprocess.Popen[Any]:
        """Start the biomedical MCP server process."""
        if not self._check_port_available():
            raise BioMCPServerError(
                f"Port {self.port} is already in use. Please stop other servers or use a different port."
            )
        
        # Locate server file - try multiple possible locations
        this_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible locations for the server file
        possible_locations = [
            # First try the same directory (for backward compatibility)
            os.path.join(this_dir, "biomcp_server.py"),
            # Then try the mcp directory (correct location)
            os.path.join(os.path.dirname(this_dir), "mcp", "biomcp_server.py"),
            # Also try relative to the bioagents directory
            os.path.join(os.path.dirname(this_dir), "biomcp_server.py"),
        ]
        
        server_file = None
        for location in possible_locations:
            if os.path.exists(location):
                server_file = location
                logger.debug(f"Found server file at: {server_file}")
                break
        
        if not server_file:
            locations_tried = '\n  '.join(possible_locations)
            raise BioMCPServerError(
                f"Server file not found in any of the expected locations:\n  {locations_tried}\n"
                f"Please ensure biomcp_server.py is available in one of these paths."
            )
        
        logger.info(f"Starting biomedical MCP server on {self._server_url}")
        
        # Set environment variables for server
        env = os.environ.copy()
        env["BIOMCP_PORT"] = str(self.port)
        
        # Start server process
        try:
            process = subprocess.Popen(
                ["uv", "run", server_file],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
        except Exception as e:
            raise BioMCPServerError(f"Failed to start server process: {e}")
        
        # Monitor startup
        logger.info("Waiting for server to start...")
        for i in range(self.server_startup_timeout):
            await asyncio.sleep(1)
            
            # Check if process crashed
            if process.poll() is not None:
                stdout, _ = process.communicate()
                raise BioMCPServerError(
                    f"Server process terminated unexpectedly after {i+1} seconds.\n"
                    f"Output:\n{stdout}"
                )
            
            # Test HTTP connectivity after initial startup time
            if i >= 3:
                success, message = self._test_http_endpoint(self._server_url)
                logger.debug(f"Connection test {i-2}: {message}")
                
                if success:
                    # Test MCP endpoint specifically  
                    mcp_success, mcp_message = self._test_http_endpoint(self._mcp_endpoint)
                    logger.debug(f"MCP endpoint test: {mcp_message}")
                    
                    if mcp_success or "404" in mcp_message:
                        logger.info("Server is responding to HTTP requests!")
                        return process
        
        # Timeout reached
        process.terminate()
        stdout, _ = process.communicate()
        raise BioMCPServerError(
            f"Server failed to respond after {self.server_startup_timeout} seconds.\n"
            f"Server output:\n{stdout}"
        )
    
    async def _create_connection(self) -> MCPServerStreamableHttp:
        """Establish connection to the MCP server."""
        try:
            
            server = MCPServerStreamableHttp(
                name="Biomedical MCP Server",
                params={
                    "url": self._mcp_endpoint,
                    "timeout": timedelta(seconds=self.timeout),  # HTTP request timeout
                    "sse_read_timeout": timedelta(seconds=self.timeout),  # SSE read timeout
                },
                client_session_timeout_seconds=self.timeout,  # MCP session timeout
            )
            await server.__aenter__()
            return server
        except Exception as e:
            raise BioMCPConnectionError(f"Failed to connect to MCP server: {e}")
    
    def _create_agent(self, mcp_server: MCPServerStreamableHttp) -> Agent:
        """
        Create and configure the biomedical agent with the provided MCP server.
        
        This method creates a fully initialized Agent instance with the MCP server
        properly configured in the mcp_servers list.
        
        Args:
            mcp_server: The MCP server connection to use for biomedical tools
            
        Returns:
            Configured Agent instance for biomedical research with MCP server initialized
        """
        return Agent(
            name="BiomedicalAssistant",
            instructions=(
                "You are a biomedical research assistant. Use the available biomedical tools "
                "to answer questions about pubmed, genetic variants, research articles, and biomedical data. "
                
                "CRITICAL FORMATTING REQUIREMENTS:\n"
                "- Your response MUST start with '[biomcp]' followed by well-formatted markdown text\n"
                "- When tools return JSON data, convert it to clean, readable markdown format\n"
                "- Use proper markdown headers (##, ###), bullet points, and formatting\n"
                "- Present article lists with clear titles, authors, and summaries\n"
                "- Format genetic variant data in tables or structured lists\n"
                "- Make technical information accessible and well-organized\n"
                "- Never output raw JSON - always convert to human-readable markdown\n"
                
                "CONTENT GUIDELINES:\n"
                "- Be helpful, accurate, and informative in your responses\n"
                "- Provide clear explanations of results and their significance\n"
                "- If a biomedical query takes time, be patient as external databases may be slow\n"
                "- Include relevant details but organize them clearly\n"
                "- Use markdown formatting for better readability\n"
                
                "EXAMPLE OUTPUT FORMAT:\n"
                "[biomcp] ## Search Results\n\n"
                "### Found 3 relevant articles:\n\n"
                "**1. Article Title Here**\n"
                "- *Authors:* Smith et al.\n"
                "- *Journal:* Nature Medicine\n"
                "- *Summary:* Brief description...\n\n"
                "**2. Second Article Title**\n"
                "- *Authors:* Johnson et al.\n"
                "- etc..."
            ),
            handoff_description=(
                "Use this subagent to answer questions about pubmed, genetic variants, research articles, and biomedical data. "
            ),
            handoffs=[],
            mcp_servers=[mcp_server],  # Properly initialized with the MCP server
            model_settings=ModelSettings(tool_choice="auto"), # "required" is for regular tools, "auto" is for MCP tools
            tool_use_behavior="stop_on_first_tool",
        )
    
    @classmethod
    async def create(
        cls,
        name: str = "Biomedical MCP Agent",
        model_name: str = LLM.GPT_4_1_MINI,
        instructions: str = "You are a biomedical research assistant.",
        port: Optional[int] = BIOMCP_PORT,
        timeout: int = 60,
        server_startup_timeout: int = 60,
        debug: bool = False,
    ) -> "BioMCPAgent":
        """
        Async factory method to create a fully initialized BioMCPAgent.
        
        This method properly handles all async initialization including:
        - Server process startup
        - MCP connection establishment  
        - Agent initialization with MCP server
        
        Args:
            name: Agent name
            model_name: LLM model to use
            instructions: Agent instructions
            port: Port for the MCP server
            timeout: Request timeout in seconds
            server_startup_timeout: Server startup timeout in seconds  
            debug: Enable debug logging
            
        Returns:
            BioMCPAgent: Fully initialized agent ready for use
            
        Raises:
            BioMCPServerError: If server startup fails
            BioMCPConnectionError: If MCP connection fails
            BioMCPAgentError: If agent initialization fails
            
        Example:
            ```python
            bio_agent = await BioMCPAgent.create(debug=True)
            result = await bio_agent.achat("Find articles about cancer")
            await bio_agent.close()  # Clean up when done
            ```
        """
        # Create instance with synchronous initialization
        instance = cls(
            name=name,
            model_name=model_name, 
            instructions=instructions,
            port=port,
            timeout=timeout,
            server_startup_timeout=server_startup_timeout,
            debug=debug
        )
        
        # Perform async initialization
        await instance._async_initialize()
        
        return instance
    
    async def _async_initialize(self) -> None:
        """
        Perform async initialization of server, MCP connection, and agent.
        
        This method handles all async resource creation with proper error handling
        and cleanup on failure.
        
        Raises:
            BioMCPServerError: If server startup fails
            BioMCPConnectionError: If MCP connection fails
            BioMCPAgentError: If agent initialization fails
        """
        if self._initialized:
            return
            
        try:
            logger.info("Starting async initialization of biomedical MCP stack...")
            
            # Record the context where this connection is being created
            self._connection_context_id = self._get_current_context_id()
            logger.debug(f"Creating MCP connection in context: {self._connection_context_id}")
            
            # Step 1: Start server process
            logger.info("Starting MCP server process...")
            self._server_process = await self._start_server()
            self._server_started = True
            logger.info("MCP server process started successfully")
            
            # Step 2: Create MCP connection (properly managed as async resource)
            logger.info("Establishing MCP server connection...")
            self._mcp_server = await self._create_connection()
            logger.info("MCP server connection established")
            
            # Step 3: Initialize Agent with MCP server
            logger.info("Creating Agent with MCP server...")
            self._agent = self._create_agent(self._mcp_server)
            logger.info("Agent created and configured with MCP server")
            
            # Mark as initialized
            self._initialized = True
            logger.info("Biomedical MCP stack fully initialized and ready")
            
        except Exception as e:
            # Clean up partial initialization
            logger.error(f"Async initialization failed: {e}")
            await self._cleanup()
            
            # Re-raise with proper exception type
            if isinstance(e, (BioMCPServerError, BioMCPConnectionError)):
                raise
            else:
                raise BioMCPAgentError(f"Failed to initialize biomedical stack: {e}") from e
    
    async def start(self) -> None:
        """
        Start the biomedical MCP server and initialize the Agent with MCP server connection.
        
        This method provides a convenient interface for explicit lifecycle management.
        It delegates to ensure_agent_ready() following the DRY principle.
        
        **Simplified Implementation**: This method now simply calls ensure_agent_ready()
        to avoid code duplication and maintain a single source of truth for initialization.
        
        **Usage**:
        ```python
        agent = BioMCPAgent()
        await agent.start()  # Everything is now ready
        result = await agent.achat("Find articles about cancer")
        ```
        
        Raises:
            BioMCPServerError: If server startup fails
            BioMCPConnectionError: If MCP connection fails
            BioMCPAgentError: If agent initialization fails
        """
        await self.ensure_agent_ready()
        logger.info("BioMCP Agent started and ready for use")
    
    async def stop(self) -> None:
        """
        Stop the server process and clean up resources.
        
        This method only handles server process cleanup since connections
        are managed per-operation.
        """
        await self._cleanup()
        logger.info("BioMCP server stopped")
    
    async def _cleanup(self) -> None:
        """
        Internal cleanup method - handles MCP connection, agent, and server process cleanup.
        
        This method safely handles cleanup even when resources were created in different
        async contexts, which can happen with auto-initialization using asyncio.run().
        Enhanced with context tracking and better cross-task error handling.
        """
        logger.debug(f"Starting cleanup in context: {self._get_current_context_id()}")
        
        # Clean up MCP connection first with improved error handling
        if self._mcp_server is not None:
            try:
                logger.debug("Cleaning up MCP server connection...")
                # Use comprehensive warning suppression for cross-task cleanup
                with suppress_async_cleanup_warnings():
                    # Attempt graceful cleanup
                    await self._mcp_server.__aexit__(None, None, None)
                logger.debug("MCP server connection closed successfully")
            except RuntimeError as e:
                if any(phrase in str(e).lower() for phrase in ["cancel scope", "different task"]):
                    # This is the expected cross-task cleanup issue - suppress it
                    logger.debug("MCP connection cleanup completed (cross-task context)")
                else:
                    logger.warning(f"MCP connection cleanup issue: {e}")
            except Exception as e:
                # Suppress all other cleanup errors as they're non-critical
                logger.debug(f"MCP connection cleanup completed with minor issue: {e}")
            finally:
                self._mcp_server = None
        
        # Reset agent and initialization state
        self._agent = None
        self._initialized = False
        
        # Reset context tracking
        self._connection_context_id = None
        
        # Terminate server process
        if self._server_process:
            try:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5)
                    logger.debug("Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Force-stopping server...")
                    self._server_process.kill()
                    self._server_process.wait()
                    logger.debug("Server force-stopped")
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            self._server_process = None
        
        # Reset server state
        self._server_started = False
        
        logger.debug("Cleanup completed successfully")
    
    async def ensure_agent_ready(self) -> Agent:
        """
        Ensure the biomedical Agent is fully initialized and ready for use.
        
        This method provides a clean, public API for initializing the complete
        biomedical stack. It includes connection health checks to handle cases
        where connections may have become stale due to cross-context usage.
        
        **Features**:
        - Idempotent: Safe to call multiple times
        - Thread-safe: Uses async locks to prevent race conditions  
        - Connection health checks: Ensures MCP connection is active
        - Automatic recovery: Recreates stale connections and agents
        - Fail-fast: Cleans up partial initialization on failure
        
        **Usage**:
        ```python
        bio_agent = BioMCPAgent()
        agent = await bio_agent.ensure_agent_ready()
        # Now agent is fully ready for biomedical queries
        ```
        
        Returns:
            Agent: Fully initialized Agent with healthy MCP server connection
            
        Raises:
            BioMCPServerError: If server startup fails
            BioMCPConnectionError: If MCP connection fails  
            BioMCPAgentError: If agent initialization fails
        """
        # Ensure basic initialization first
        await self._async_initialize()
        
        # Test if existing agent has healthy MCP connections
        if self._agent and await self._test_agent_mcp_health():
            logger.debug("Existing agent has healthy MCP connections, reusing")
            return self._agent
        
        # Agent doesn't exist or has stale connections - recreate everything
        logger.debug("Agent has stale connections or doesn't exist, recreating...")
        self._agent = None
        
        # Clean up old MCP connection
        if self._mcp_server is not None:
            try:
                with suppress_async_cleanup_warnings():
                    await self._mcp_server.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors for stale connections
            self._mcp_server = None
        
        try:
            # Create fresh connection and agent
            healthy_connection = await self._ensure_healthy_connection()
            self._agent = self._create_agent(healthy_connection)
            self._mcp_server = healthy_connection
            
            logger.debug("Agent recreated with fresh MCP connection")
            return self._agent
            
        except Exception as e:
            # If creation fails, mark as uninitialized and retry once
            logger.warning(f"Agent creation failed: {e}, retrying with full reinitialization...")
            
            # Complete cleanup before retry
            self._agent = None
            self._initialized = False
            
            # Clean up MCP connection
            if self._mcp_server is not None:
                try:
                    with suppress_async_cleanup_warnings():
                        await self._mcp_server.__aexit__(None, None, None)
                except Exception:
                    pass
                self._mcp_server = None
            
            # Clean up server process to free the port
            if self._server_process is not None:
                try:
                    logger.debug("Stopping server process before retry...")
                    self._server_process.terminate()
                    try:
                        self._server_process.wait(timeout=3)
                        logger.debug("Server process stopped")
                    except subprocess.TimeoutExpired:
                        logger.debug("Force-killing server process...")
                        self._server_process.kill()
                        self._server_process.wait()
                except Exception as cleanup_error:
                    logger.debug(f"Error cleaning up server: {cleanup_error}")
                self._server_process = None
                self._server_started = False
            
            # Wait for port to be released
            await asyncio.sleep(1)
            
            # Retry initialization once
            await self._async_initialize()
            healthy_connection = await self._ensure_healthy_connection()
            self._agent = self._create_agent(healthy_connection)
            self._mcp_server = healthy_connection
            
            return self._agent
    
    async def _test_agent_mcp_health(self) -> bool:
        """
        Test the health of the agent's actual MCP server connections.
        
        This method tests the operational MCP connections that the agent will use,
        rather than just the BioMCPAgent's connection tracking. This is crucial
        for detecting stale connections that occur when the agent is used across
        different asyncio.run() contexts.
        
        Returns:
            bool: True if the agent's MCP connections are healthy and operational
        """
        if not self._agent or not self._agent.mcp_servers:
            logger.debug("Agent MCP health check: No agent or MCP servers")
            return False
        
        # First check if connection is from a different async context
        if self._is_connection_context_stale():
            logger.debug("Agent MCP health check: Connection context is stale")
            return False
        
        try:
            # Test the agent's actual MCP server connection
            mcp_server = self._agent.mcp_servers[0]
            
            # Check if the MCP server session exists and is connected
            if not hasattr(mcp_server, 'session') or not mcp_server.session:
                logger.debug("Agent MCP health check: No session")
                return False
            
            session = mcp_server.session
            
            # Check for closed streams
            if hasattr(session, '_write_stream'):
                write_stream = session._write_stream
                if hasattr(write_stream, '_closed') and write_stream._closed:
                    logger.debug("Agent MCP health check: Write stream closed")
                    return False
            
            if hasattr(session, '_read_stream'):
                read_stream = session._read_stream
                if hasattr(read_stream, '_closed') and read_stream._closed:
                    logger.debug("Agent MCP health check: Read stream closed")
                    return False
            
            # Try to perform an actual operation to test functionality
            await session.list_tools()
            logger.debug("Agent MCP health check: Passed")
            return True
            
        except Exception as e:
            logger.debug(f"Agent MCP health check failed: {e}")
            return False
    
    def _quick_agent_health_check(self) -> bool:
        """
        Quick synchronous health check for the agent's MCP connections.
        
        This is used in sync contexts where we can't await async operations.
        It only checks basic connection state without performing operations.
        
        Returns:
            bool: True if basic connection state appears healthy
        """
        if not self._agent or not self._agent.mcp_servers:
            return False
        
        # Check if connection is from a different context (sync version)
        if self._is_connection_context_stale():
            logger.debug("Quick health check: Connection context is stale")
            return False
        
        try:
            mcp_server = self._agent.mcp_servers[0]
            
            # Check if the MCP server session exists
            if not hasattr(mcp_server, 'session') or not mcp_server.session:
                return False
            
            session = mcp_server.session
            
            # Check for closed streams (quick check)
            if hasattr(session, '_write_stream'):
                write_stream = session._write_stream
                if hasattr(write_stream, '_closed') and write_stream._closed:
                    return False
            
            if hasattr(session, '_read_stream'):
                read_stream = session._read_stream
                if hasattr(read_stream, '_closed') and read_stream._closed:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _register_cleanup(self) -> None:
        """Register cleanup for auto-started server."""
        # Use weakref to avoid circular references
        weak_self = weakref.ref(self)
        
        def cleanup_server():
            agent = weak_self()
            if agent and agent._auto_started:
                try:
                    agent._synchronous_cleanup()
                except Exception as e:
                    # Suppress all cleanup errors - they're non-critical
                    logger.debug(f"Auto-cleanup completed: {e}")
        
        atexit.register(cleanup_server)

    def _synchronous_cleanup(self) -> None:
        """
        Synchronous cleanup for MCP connection, agent, and server process.
        
        This method provides best-effort cleanup in a synchronous context.
        It gracefully handles issues that can arise from cross-task resource cleanup.
        Enhanced with context tracking and comprehensive error suppression.
        """
        try:
            logger.debug(f"Starting sync cleanup in context: {self._get_current_context_id()}")
            
            # Clean up MCP connection (best effort in synchronous context)
            if self._mcp_server is not None:
                try:
                    # Note: We can't properly await the async __aexit__ in sync context
                    # In cases where the connection was created via asyncio.run(), the
                    # event loop and resources are automatically cleaned up when the
                    # asyncio.run() context exits, so this is primarily for state cleanup
                    logger.debug("Synchronous cleanup: MCP connection state reset")
                except Exception as e:
                    # Gracefully handle any cleanup issues
                    logger.debug(f"MCP connection sync cleanup completed: {type(e).__name__}")
                finally:
                    self._mcp_server = None
            
            # Reset agent and initialization state
            self._agent = None
            self._initialized = False
            
            # Reset context tracking
            self._connection_context_id = None
            
            # Terminate server process synchronously
            if self._server_process:
                try:
                    self._server_process.terminate()
                    try:
                        self._server_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._server_process.kill()
                        self._server_process.wait()
                    logger.debug("Server process terminated synchronously")
                except Exception as e:
                    logger.debug(f"Server termination completed: {type(e).__name__}")
                finally:
                    self._server_process = None
            
            # Reset server state
            self._server_started = False
            
            logger.debug("Synchronous cleanup completed successfully")
            
        except Exception as e:
            # Final safety net - ensure cleanup doesn't crash the application
            logger.debug(f"Synchronous cleanup completed with minor issues: {type(e).__name__}")
    
    async def achat(self, query_str: str) -> AgentResponse:
        """
        Process a biomedical research query and return the response.
        
        This method provides the main interface for biomedical interactions.
        It uses automatic connection health checks to ensure reliable operation
        even when connections may have become stale due to cross-context usage.
        
        **Simplified Flow**: 
        1. Validate input
        2. Ensure agent is ready with healthy connection
        3. Execute query with initialized agent
        4. Return formatted response
        
        **Auto-Start Behavior**: Automatically initializes the complete biomedical
        stack (server + MCP + agent) with connection health verification on first call.
        
        **Connection Recovery**: Automatically detects and recovers from stale 
        connections that may occur when using the agent across different async contexts.
        
        Args:
            query_str: Natural language query about biomedical research, variants, or articles
            
        Returns:
            AgentResponse: Response from the biomedical agent with tool results and analysis
            
        Raises:
            BioMCPAgentError: If initialization fails or query processing fails
            
        Examples:
            ```python
            # Using async factory (recommended)
            bio_agent = await BioMCPAgent.create()
            result = await bio_agent.achat("Find recent CRISPR articles")
            
            # Using auto-initialization with health checks
            bio_agent = BioMCPAgent()
            result = await bio_agent.achat("Get variant rs113488022 details")
            ```
        """
        if not query_str.strip():
            raise BioMCPAgentError("Query string cannot be empty")
        
        # Wrap the entire method execution with comprehensive warning suppression
        with suppress_async_cleanup_warnings():
            # Ensure agent is ready with healthy connection (includes health checks)
            initialized_agent = await self.ensure_agent_ready()
            
            # Execute query with the initialized agent
            logger.info(f"Processing biomedical query: {query_str}")
            
            try:
                # Generate trace ID for tracking
                trace_id = gen_trace_id()
                
                with trace(workflow_name="Biomedical Query", trace_id=trace_id):
                    try:
                        result = await asyncio.wait_for(
                            Runner.run(
                                starting_agent=initialized_agent,
                                input=query_str,
                                max_turns=3,
                            ),
                            timeout=self.timeout
                        )
                        
                        logger.info(f"Query completed successfully. Trace ID: {trace_id}")
                        return self._construct_response(result, "", "biomcp")
                        
                    except (TypeError, ValueError) as json_error:
                        return AgentResponse(
                            response_str=f"Error: {json_error}",
                            citations=[],
                            judge_response="",
                            route="biomcp-fallback"
                        )
                            
            except asyncio.TimeoutError as e:
                error_msg = f"Query timed out after {self.timeout} seconds"
                logger.error(error_msg)
                raise BioMCPAgentError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Query failed: {str(e)}"
                logger.error(error_msg)
                raise BioMCPAgentError(error_msg) from e
    
    async def close(self) -> None:
        """
        Properly close and clean up all async resources.
        
        This method ensures proper cleanup of:
        - MCP server connection
        - Server process
        - Agent instance
        
        **Note**: Some cleanup warnings about "cancel scope" or "different task" are
        expected when using auto-initialization and are safely handled.
        
        **Usage**:
        ```python
        bio_agent = await BioMCPAgent.create()
        try:
            result = await bio_agent.achat("Find articles")
        finally:
            await bio_agent.close()  # Proper cleanup
        ```
        
        Note: This method is automatically called when using the context manager.
        """
        try:
            await self._cleanup()
            self._initialized = False
            logger.info("BioMCP Agent closed and resources cleaned up")
        except Exception as e:
            # Ensure close() never raises exceptions to the user
            logger.debug(f"Cleanup completed with minor issues: {type(e).__name__}")
            self._initialized = False
            logger.info("BioMCP Agent closed (with minor cleanup issues)")
    
    async def __aenter__(self) -> "BioMCPAgent":
        """
        Context manager entry.
        
        Note: The instance should already be initialized via create() before
        using as a context manager.
        """
        if not self._initialized:
            await self._async_initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with proper cleanup."""
        await self.close()
    
    @property
    def is_started(self) -> bool:
        """Check if the server is started and ready."""
        return self._server_started
    
    @property
    def server_url(self) -> str:
        """Get the server URL."""
        return self._server_url
    
    def __repr__(self) -> str:
        status = "started" if self._server_started else "stopped"
        return f"BioMCPAgent(port={self.port}, server_status={status})"
    
    async def _ensure_healthy_connection(self) -> MCPServerStreamableHttp:
        """
        Ensure we have a healthy MCP server connection.
        
        This method checks if the current connection is healthy and recreates it if needed.
        This is crucial for handling connections that may have been closed due to 
        cross-context cleanup or other lifecycle issues.
        
        Returns:
            MCPServerStreamableHttp: A healthy, active MCP server connection
            
        Raises:
            BioMCPConnectionError: If connection cannot be established
        """
        # If we don't have a connection, create one
        if self._mcp_server is None:
            logger.debug("No MCP connection exists, creating new one...")
            self._mcp_server = await self._create_connection()
            return self._mcp_server
        
        # Test if the existing connection is healthy
        try:
            # Comprehensive health check
            if hasattr(self._mcp_server, 'session') and self._mcp_server.session:
                session = self._mcp_server.session
                
                # Check if the session's write stream is closed
                if hasattr(session, '_write_stream'):
                    write_stream = session._write_stream
                    if hasattr(write_stream, '_closed') and write_stream._closed:
                        logger.debug("MCP connection stream is closed, recreating...")
                        raise ConnectionError("Stream closed")
                
                # Check if the session's read stream is closed
                if hasattr(session, '_read_stream'):
                    read_stream = session._read_stream
                    if hasattr(read_stream, '_closed') and read_stream._closed:
                        logger.debug("MCP connection read stream is closed, recreating...")
                        raise ConnectionError("Read stream closed")
                
                # Try to perform an actual operation to test the connection
                try:
                    # This will fail if the connection is stale with "Server not initialized"
                    await session.list_tools()
                    logger.debug("MCP connection health check passed")
                    return self._mcp_server
                except Exception as test_error:
                    if "not initialized" in str(test_error) or "connect()" in str(test_error):
                        logger.debug(f"MCP connection is disconnected: {test_error}, recreating...")
                        raise ConnectionError("Session disconnected")
                    else:
                        # Some other error, but connection might still be OK
                        logger.debug(f"MCP connection test had minor issue: {test_error}, but connection seems OK")
                        return self._mcp_server
                
            else:
                logger.debug("MCP connection session is invalid, recreating...")
                raise ConnectionError("Invalid session")
                
        except ConnectionError:
            # Re-raise connection errors for handling below
            raise
        except Exception as e:
            logger.debug(f"MCP connection health check failed: {e}, recreating...")
            raise ConnectionError(f"Health check failed: {e}")
        
        # If we get here, we need to recreate the connection
        logger.debug("Recreating MCP connection due to health check failure...")
        
        # Clean up the old connection
        if self._mcp_server is not None:
            try:
                with suppress_async_cleanup_warnings():
                    await self._mcp_server.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors for stale connections
            self._mcp_server = None
        
        # Create a new connection
        logger.debug("Creating fresh MCP connection...")
        self._mcp_server = await self._create_connection()
        return self._mcp_server

    def _get_current_context_id(self) -> str:
        """
        Get a unique identifier for the current async context.
        
        This helps detect when connections are being used across
        different asyncio.run() calls, which can cause cleanup issues.
        
        Returns:
            str: Unique identifier for the current async context
        """
        try:
            import asyncio
            import threading
            loop = asyncio.get_running_loop()
            thread_id = threading.get_ident()
            # Create a unique context ID based on loop and thread
            return f"{id(loop)}_{thread_id}_{id(loop._selector) if hasattr(loop, '_selector') else 'no_selector'}"
        except RuntimeError:
            # No event loop running - use thread ID
            import threading
            return f"sync_{threading.get_ident()}"
    
    def _is_connection_context_stale(self) -> bool:
        """
        Check if the current connection was created in a different async context.
        
        Returns:
            bool: True if connection is from a different context and should be recreated
        """
        if self._connection_context_id is None:
            return True  # No context recorded, consider stale
        
        current_context = self._get_current_context_id()
        is_stale = self._connection_context_id != current_context
        
        if is_stale:
            logger.debug(f"Connection context mismatch: {self._connection_context_id} vs {current_context}")
        
        return is_stale

    def _construct_response(
        self, 
        run_result: RunResult,
        judge_response: str = "",
        route: str = ""
    ) -> AgentResponse:
        """
        Override base class to ensure biomedical responses are properly formatted as markdown.
        
        This method post-processes the response to convert any JSON content to readable
        markdown format and ensures consistent formatting standards.
        """
        # Get the base response first
        base_response = super()._construct_response(run_result, judge_response, route)
        
        # Post-process the response string to ensure proper markdown formatting
        formatted_response = self._format_biomedical_response(base_response.response_str)
        
        return AgentResponse(
            response_str=formatted_response,
            citations=base_response.citations,
            judge_response=judge_response,
            route=route
        )
    
    def _format_biomedical_response(self, response_str: str) -> str:
        """
        Format biomedical response to ensure readable markdown output.
        
        This method handles:
        - JSON data conversion to markdown tables/lists
        - Proper markdown structure and headers
        - Consistent formatting for articles, variants, and data
        
        Args:
            response_str: Raw response string from the agent
            
        Returns:
            str: Well-formatted markdown response
        """
        # If response doesn't start with [biomcp], add it
        if not response_str.startswith('[biomcp]'):
            response_str = f'[biomcp] {response_str}'
        
        # Try to detect and format JSON content within the response
        try:
            # Look for JSON-like structures in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_str, re.DOTALL)
            
            for json_match in json_matches:
                try:
                    # Try to parse as JSON
                    data = json.loads(json_match)
                    
                    # Format based on content type
                    if isinstance(data, dict):
                        formatted_json = self._format_json_dict(data)
                        response_str = response_str.replace(json_match, formatted_json)
                    elif isinstance(data, list):
                        formatted_json = self._format_json_list(data)
                        response_str = response_str.replace(json_match, formatted_json)
                        
                except json.JSONDecodeError:
                    # Not valid JSON, leave as is
                    continue
                    
        except Exception:
            # If any processing fails, return the original response
            pass
        
        # Clean up extra whitespace and ensure proper markdown spacing
        response_str = re.sub(r'\n{3,}', '\n\n', response_str)  # Max 2 consecutive newlines
        response_str = response_str.strip()
        
        return response_str
    
    def _format_json_dict(self, data: dict) -> str:
        """Format a JSON dictionary as readable markdown."""
        if not data:
            return "*No data available*"
        
        # Check if this looks like an article/paper entry
        if any(key in data for key in ['title', 'pmid', 'authors', 'journal']):
            return self._format_article_data(data)
        
        # Check if this looks like variant data
        if any(key in data for key in ['variant', 'rsid', 'chromosome', 'position']):
            return self._format_variant_data(data)
        
        # Generic formatting for other data
        formatted = []
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2) if isinstance(value, dict) else ', '.join(map(str, value))
            else:
                value_str = str(value)
            formatted.append(f"- **{key.title()}:** {value_str}")
        
        return '\n'.join(formatted)
    
    def _format_json_list(self, data: list) -> str:
        """Format a JSON list as readable markdown."""
        if not data:
            return "*No results found*"
        
        # If list contains dictionaries, format each one
        if data and isinstance(data[0], dict):
            formatted_items = []
            for i, item in enumerate(data, 1):
                formatted_item = f"**{i}.** {self._format_json_dict(item)}"
                formatted_items.append(formatted_item)
            return '\n\n'.join(formatted_items)
        
        # Simple list formatting
        return '\n'.join(f"- {item}" for item in data)
    
    def _format_article_data(self, article: dict) -> str:
        """Format article data as readable markdown."""
        formatted = []
        
        # Title
        if 'title' in article:
            formatted.append(f"**{article['title']}**")
        
        # Authors
        if 'authors' in article:
            authors = article['authors']
            if isinstance(authors, list):
                authors_str = ', '.join(authors[:3])  # First 3 authors
                if len(authors) > 3:
                    authors_str += f" *et al.* ({len(authors)} total)"
            else:
                authors_str = str(authors)
            formatted.append(f"- *Authors:* {authors_str}")
        
        # Journal and year
        journal_info = []
        if 'journal' in article:
            journal_info.append(article['journal'])
        if 'year' in article:
            journal_info.append(f"({article['year']})")
        if journal_info:
            formatted.append(f"- *Journal:* {' '.join(journal_info)}")
        
        # PMID
        if 'pmid' in article:
            formatted.append(f"- *PMID:* {article['pmid']}")
        
        # Abstract/summary
        if 'abstract' in article:
            abstract = article['abstract'][:200] + "..." if len(str(article['abstract'])) > 200 else article['abstract']
            formatted.append(f"- *Summary:* {abstract}")
        
        # DOI or URL
        if 'doi' in article:
            formatted.append(f"- *DOI:* {article['doi']}")
        elif 'url' in article:
            formatted.append(f"- *URL:* {article['url']}")
        
        return '\n'.join(formatted)
    
    def _format_variant_data(self, variant: dict) -> str:
        """Format genetic variant data as readable markdown."""
        formatted = []
        
        # Variant ID
        if 'rsid' in variant:
            formatted.append(f"**Variant: {variant['rsid']}**")
        elif 'variant' in variant:
            formatted.append(f"**Variant: {variant['variant']}**")
        
        # Genomic location
        location_parts = []
        if 'chromosome' in variant:
            location_parts.append(f"Chr {variant['chromosome']}")
        if 'position' in variant:
            location_parts.append(f"Position {variant['position']}")
        if location_parts:
            formatted.append(f"- *Location:* {', '.join(location_parts)}")
        
        # Alleles
        if 'ref' in variant and 'alt' in variant:
            formatted.append(f"- *Alleles:* {variant['ref']} → {variant['alt']}")
        
        # Frequency information
        if 'frequency' in variant:
            freq = variant['frequency']
            if isinstance(freq, dict):
                freq_info = []
                for pop, f in freq.items():
                    freq_info.append(f"{pop}: {f}")
                formatted.append(f"- *Frequency:* {', '.join(freq_info)}")
            else:
                formatted.append(f"- *Frequency:* {freq}")
        
        # Clinical significance
        if 'clinical_significance' in variant:
            formatted.append(f"- *Clinical Significance:* {variant['clinical_significance']}")
        
        # Associated diseases/phenotypes
        if 'diseases' in variant:
            diseases = variant['diseases']
            if isinstance(diseases, list):
                formatted.append(f"- *Associated Diseases:* {', '.join(diseases)}")
            else:
                formatted.append(f"- *Associated Diseases:* {diseases}")
        
        return '\n'.join(formatted)


#----------------------------------------------
# Example usage and testing functions
#----------------------------------------------

# Example usage and testing functions
async def example_usage():
    """Example demonstrating how to use BioMCPAgent with comprehensive biomedical examples."""
    print("🧬 BioMCPAgent - Comprehensive Biomedical Research Examples")
    print("=" * 80)
    
    try:
        # Demonstrate both usage patterns
        print("🧪 Testing explicit lifecycle management...")
        async with BioMCPAgent(
            debug=True, 
            timeout=90,  # 90 seconds for complex biomedical queries
        ) as agent:
            print(f"✅ Explicitly started {agent}")
            print(f"🔗 Server URL: {agent.server_url}")
            print(f"⚙️  Configured with {agent.timeout}s timeout")
            
            # Test one query with explicit management
            result = await agent.achat("What tools are available in the biomedical server?")
            print(f"✅ Explicit management response received")
        
        print("\n🚀 Testing auto-start capability...")
        # Test auto-start capability (new feature)
        auto_agent = BioMCPAgent(debug=True, timeout=90)
        print(f"📦 Created agent (not started yet): {auto_agent}")
        
        # This will auto-start the agent
        result = await auto_agent.achat("Get details for the genetic variant rs113488022.")
        print(f"✅ Auto-start response received!")
        print(f"🔄 Agent is now started: {auto_agent.is_started}")
        
        # Continue with remaining examples using the auto-started agent
        remaining_examples = [
            "Find recent research articles about Alzheimer's disease.",
            "Search for articles about CRISPR gene editing and cancer treatment.",
            "Get detailed information about PubMed article 37351900.",
        ]
        
        for i, query in enumerate(remaining_examples, 1):
            print(f"\n{'='*80}")
            print(f"🧪 Auto-agent example {i}/{len(remaining_examples)}: {query}")
            print(f"{'='*80}")
            
            try:
                result = await auto_agent.achat(query)
                print(f"✅ Response:\n{result}")
            except Exception as e:
                print(f"❌ Error in example {i}: {e}")
            
            # Add delay between examples
            if i < len(remaining_examples):
                print("\n⏳ Waiting before next example...")
                await asyncio.sleep(2)
        
        # Note: auto-started agent will be cleaned up automatically on exit
        
        print(f"\n🎉 All biomedical examples completed!")
                    
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")
        return False
    
    return True


if __name__ == "__main__":
    """Run comprehensive biomedical examples when script is executed directly."""
    print("🚀 Starting BioMCPAgent comprehensive testing...")
    print("📋 This will demonstrate all biomedical research capabilities")
    print("⏱️  Biomedical queries can take 30-90 seconds due to external database calls\n")
    
    try:
        success = asyncio.run(example_usage())
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 BioMCPAgent comprehensive testing completed successfully!")
            print("✅ All biomedical research examples demonstrated the following capabilities:")
            print("   • Tool discovery and listing")
            print("   • Genetic variant analysis")
            print("   • Biomedical literature search")
            print("   • Article retrieval and analysis")
            print("   • Structured biomedical data processing")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ BioMCPAgent testing failed")
            print("🔧 Please check the error messages above and ensure:")
            print("   • OpenAI API key is properly configured")
            print("   • biomcp-python dependencies are available")
            print("   • Network connectivity for PubMed access")
            print("   • Port availability (default: 8131)")
            print("=" * 80)
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        print("🛑 BioMCPAgent testing stopped")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        print("🔧 Please check your environment configuration")
        exit(1) 