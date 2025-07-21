"""
Tests for bioagents.models.llms module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bioagents.models.llms import LLM


class TestLLM:
    """Test LLM dataclass and functionality."""
    
    def test_llm_model_constants(self):
        """Test that LLM has expected model constants."""
        assert LLM.GPT_4_1 == "gpt-4.1"
        assert LLM.GPT_4_1_MINI == "gpt-4.1-mini" 
        assert LLM.GPT_4_1_NANO == "gpt-4.1-nano"
        assert LLM.GPT_4O == "gpt-4o"
    
    def test_llm_creation_default(self):
        """Test creating LLM with default parameters."""
        llm = LLM()
        
        assert llm._model_name == LLM.GPT_4_1_MINI
        assert llm._timeout == 60
        assert llm._client is not None
        assert llm._async_client is not None
    
    def test_llm_creation_custom_model(self):
        """Test creating LLM with custom model."""
        llm = LLM(model_name=LLM.GPT_4O)
        
        assert llm._model_name == LLM.GPT_4O
        assert llm._timeout == 60
    
    def test_llm_creation_custom_timeout(self):
        """Test creating LLM with custom timeout."""
        llm = LLM(timeout=120)
        
        assert llm._model_name == LLM.GPT_4_1_MINI
        assert llm._timeout == 120
    
    def test_llm_creation_custom_both(self):
        """Test creating LLM with custom model and timeout."""
        llm = LLM(model_name=LLM.GPT_4_1_NANO, timeout=90)
        
        assert llm._model_name == LLM.GPT_4_1_NANO
        assert llm._timeout == 90
    
    @patch('bioagents.models.llms.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_achat_completion_success(self, mock_async_openai):
        """Test successful achat_completion call."""
        # Setup mock
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response from LLM"
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLM and test
        llm = LLM()
        llm._async_client = mock_client
        
        result = await llm.achat_completion("Test query")
        
        assert result == "Test response from LLM"
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Test query"}],
            model=LLM.GPT_4_1_MINI
        )
    
    @patch('bioagents.models.llms.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_achat_completion_with_kwargs(self, mock_async_openai):
        """Test achat_completion with additional kwargs."""
        # Setup mock
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLM and test
        llm = LLM(model_name=LLM.GPT_4O)
        llm._async_client = mock_client
        
        result = await llm.achat_completion(
            "Test query", 
            temperature=0.5,
            max_tokens=100
        )
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "Test query"}],
            model=LLM.GPT_4O,
            temperature=0.5,
            max_tokens=100
        )
    
    @patch('bioagents.models.llms.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_achat_completion_empty_response(self, mock_async_openai):
        """Test achat_completion with empty response."""
        # Setup mock
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLM and test
        llm = LLM()
        llm._async_client = mock_client
        
        result = await llm.achat_completion("Test query")
        
        assert result == ""
    
    @patch('bioagents.models.llms.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_achat_completion_none_content(self, mock_async_openai):
        """Test achat_completion with None content."""
        # Setup mock
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLM and test
        llm = LLM()
        llm._async_client = mock_client
        
        result = await llm.achat_completion("Test query")
        
        assert result == ""
    
    @patch('bioagents.models.llms.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_achat_completion_api_error(self, mock_async_openai):
        """Test achat_completion with API error."""
        # Setup mock
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Create LLM and test
        llm = LLM()
        llm._async_client = mock_client
        
        with pytest.raises(Exception, match="API Error"):
            await llm.achat_completion("Test query")
    
    def test_llm_multiple_instances(self):
        """Test that multiple LLM instances are independent."""
        llm1 = LLM(model_name=LLM.GPT_4O, timeout=60)
        llm2 = LLM(model_name=LLM.GPT_4_1_NANO, timeout=120)
        
        assert llm1._model_name == LLM.GPT_4O
        assert llm2._model_name == LLM.GPT_4_1_NANO
        assert llm1._timeout == 60
        assert llm2._timeout == 120
        
        # Should have different client instances
        assert llm1._client is not llm2._client
        assert llm1._async_client is not llm2._async_client
    
    def test_llm_model_name_validation(self):
        """Test that any string can be used as model name."""
        custom_model = "custom-model-name"
        llm = LLM(model_name=custom_model)
        
        assert llm._model_name == custom_model
    
    def test_llm_timeout_validation(self):
        """Test timeout validation."""
        # Should accept positive integers
        llm = LLM(timeout=1)
        assert llm._timeout == 1
        
        llm = LLM(timeout=3600)
        assert llm._timeout == 3600
        
        # Should accept zero (though not practical)
        llm = LLM(timeout=0)
        assert llm._timeout == 0 