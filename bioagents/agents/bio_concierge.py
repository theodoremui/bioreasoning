#------------------------------------------------------------------------------
# bio_concierge.py
# 
# This is a "Bio Reasoning Concierge" that triage across multiple agents to answer
# a user's question.  This agent orchestrates across the following subagents:
# 
# 1. Chit Chat Agent
# 2. Web Reasoning Agent
# 3. Bio MCP Agent
# 
# Author: Theodore Mui
# Date: 2025-04-26
#------------------------------------------------------------------------------

import warnings
import sys

# Apply comprehensive warning suppression at module level for BioConcierge
warnings.filterwarnings("ignore", message=".*async_generator.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*cancel scope.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*different task.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Task.*pending.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*streamablehttp_client.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*", category=RuntimeWarning)

from agents import (
    Agent,
    RunContextWrapper,
    handoff
)
from datetime import datetime
from loguru import logger
import json

from bioagents.agents.base_agent import ReasoningAgent
from bioagents.agents.biomcp_agent import BioMCPAgent
from bioagents.agents.chitchat_agent import ChitChatAgent
from bioagents.agents.common import AgentResponse
from bioagents.models.llms import LLM
from bioagents.agents.web_agent import WebReasoningAgent

def on_handoff(ctx: RunContextWrapper[None]):
    print(f"\tHandoff: {ctx.agent.name} -> {ctx.next_agent.name}")

class BioConciergeAgent(ReasoningAgent):
    def __init__(
        self, name: str, 
        model_name: str=LLM.GPT_4_1_NANO,
        port: int = None,
    ):
        instructions = (
            f"You are a bio-reasoning agent {name} that routes queries to appropriate specialists. "
            "You analyze the user's query and determine the best way to respond."
            f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
        )

        super().__init__(name, model_name, instructions)

        # Store references to agent wrappers instead of accessing their agents
        self._chit_chat_agent = ChitChatAgent(name="Chit Chat Agent")
        self._web_agent = WebReasoningAgent(name="Web Reasoning Agent")
        self._biomcp_agent = BioMCPAgent(name="Bio MCP Agent", port=port)
        
        # Agent will be created dynamically when needed
        self._agent = None
        self._agent_initialized = False

    def _create_agent(self):
        """
        Create the main agent with fresh handoff agents.
        
        This method dynamically accesses the agent properties to ensure
        MCP connections are fresh when the agent is used.
        
        Note: This method assumes that BioMCP agent has already been initialized
        via ensure_agent_ready() before calling this method.
        """
        return Agent(
            name=self.name,
            model=self.model_name,
            instructions=self.instructions,
            handoffs=[
                self._chit_chat_agent.agent, 
                self._web_agent.agent, 
                self._biomcp_agent.agent,
            ]
        )
    
    async def _ensure_agent_ready(self):
        """
        Ensure the agent is ready with fresh MCP connections.
        
        This method handles the lazy initialization and refresh of the agent
        when MCP connections may have become stale.
        """
        if self._agent is None:
            logger.debug("Creating BioConcierge agent for first time")
            # Ensure BioMCP agent is ready before accessing its agent property
            await self._biomcp_agent.ensure_agent_ready()
            self._agent = self._create_agent()
            self._agent_initialized = True
            return
        
        # Check if BioMCP agent connections are healthy
        if hasattr(self._biomcp_agent, '_test_agent_mcp_health'):
            try:
                # Test if the BioMCP agent's connections are healthy
                if not await self._biomcp_agent._test_agent_mcp_health():
                    logger.debug("BioMCP agent connections are stale, recreating BioConcierge agent")
                    # Ensure BioMCP agent is ready before recreating
                    await self._biomcp_agent.ensure_agent_ready()
                    self._agent = self._create_agent()
                    self._agent_initialized = True
                    return
            except Exception as e:
                logger.debug(f"Error checking BioMCP health, recreating agent: {e}")
                # Ensure BioMCP agent is ready before recreating
                await self._biomcp_agent.ensure_agent_ready()
                self._agent = self._create_agent()
                self._agent_initialized = True
                return
        
        # Agent exists and connections appear healthy
        logger.debug("BioConcierge agent already ready")
    
    @property
    def agent(self):
        """
        Get the BioConcierge agent with automatic health checking.
        
        This property ensures that MCP connections are fresh by checking
        the health of the BioMCP agent and recreating the BioConcierge agent
        if connections are stale.
        
        Returns:
            Agent: BioConcierge agent with fresh handoff agents and MCP connections
        """
        # Check if we're in async context
        try:
            import asyncio
            asyncio.get_running_loop()
            # In async context - return agent if available, health checks should be done via ensure_agent_ready()
            if self._agent is None:
                raise ValueError(
                    "Agent not initialized. In async context, use:\n"
                    "  await bio_concierge._ensure_agent_ready()\n"
                    "  agent = bio_concierge.agent"
                )
            return self._agent
        except RuntimeError:
            # In sync context - we can do automatic initialization with health checks
            if self._agent is None:
                logger.debug("Auto-initializing BioConcierge agent (sync context)")
                
                # First ensure BioMCP agent is initialized
                try:
                    # This will trigger BioMCPAgent auto-initialization in sync context
                    biomcp_agent = self._biomcp_agent.agent
                    self._agent = self._create_agent()
                    self._agent_initialized = True
                except Exception as e:
                    raise ValueError(f"Failed to initialize BioConcierge agent: {e}")
                
                return self._agent
            
            # Agent exists, do a quick health check of BioMCP connections
            if hasattr(self._biomcp_agent, '_quick_agent_health_check'):
                if not self._biomcp_agent._quick_agent_health_check():
                    logger.debug("BioMCP connections stale, recreating BioConcierge agent (sync context)")
                    try:
                        # Re-initialize BioMCP agent and recreate BioConcierge agent
                        biomcp_agent = self._biomcp_agent.agent  # This will auto-fix in sync context
                        self._agent = self._create_agent()
                        self._agent_initialized = True
                    except Exception as e:
                        raise ValueError(f"Failed to refresh BioConcierge agent: {e}")
            
            return self._agent
    
    async def achat(self, query_str: str) -> AgentResponse:
        """
        Override the base class achat method to handle JSON serialization errors
        and ensure MCP connections are fresh, with comprehensive warning suppression.
        """
        # Import the warning suppression from BioMCPAgent
        from bioagents.agents.biomcp_agent import suppress_async_cleanup_warnings
        
        # Wrap the entire method execution with comprehensive warning suppression
        with suppress_async_cleanup_warnings():
            # Ensure agent is ready with fresh connections
            await self._ensure_agent_ready()
            
            if self._agent is None:
                raise ValueError("Agent not initialized")
            
            try:
                # Try the normal flow first
                return await super().achat(query_str)
                
            except Exception as e:
                return AgentResponse(
                    response_str=f"⚠️ Error: {e} BioMCP not available",
                    citations=[],
                    judge_response="",
                    route="biomcp-error"
                )
        
#------------------------------------------------
# Example usage
#------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import time
    
    start_time = time.time()
    agent = BioConciergeAgent(name="Bio")
    response = asyncio.run(agent.achat("How are you?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")
    
    start_time = time.time()
    response = asyncio.run(agent.achat("What is the latest news in the field of genetics?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")

    start_time = time.time()
    response = asyncio.run(agent.achat("How is the weather in Tokyo?"))
    print(f"{str(response)} ({time.time() - start_time:.1f}s)")