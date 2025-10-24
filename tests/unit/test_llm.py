"""
Unit tests for LLM integration functionality.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from lib.llm import get_completion


@pytest.mark.unit
@pytest.mark.llm
class TestLLMIntegration:
    """Test LLM integration operations."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion_success(self, mock_model_class, mock_configure, mock_environment):
        """Test successful LLM completion."""
        # Mock the model instance and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test response from Gemini."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        result = get_completion("Test prompt")
        
        assert result == "This is a test response from Gemini."
        mock_configure.assert_called_once_with(api_key=mock_environment["GOOGLE_API_KEY"])
        mock_model.generate_content.assert_called_once_with("Test prompt")

    def test_get_completion_missing_api_key(self):
        """Test LLM completion without API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_completion("Test prompt")
            assert result == ""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion_api_error(self, mock_model_class, mock_configure, mock_environment):
        """Test LLM completion with API error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model
        
        result = get_completion("Test prompt")
        
        assert result == ""

    @patch('google.generativeai.configure') 
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion_custom_model(self, mock_model_class, mock_configure, mock_environment):
        """Test LLM completion with custom model."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Custom model response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        result = get_completion("Test prompt", model="gemini-pro")
        
        assert result == "Custom model response"
        mock_model_class.assert_called_once_with(model_name="gemini-pro")

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion_empty_response(self, mock_model_class, mock_configure, mock_environment):
        """Test LLM completion with empty response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = ""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        result = get_completion("Test prompt")
        
        assert result == ""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion_long_prompt(self, mock_model_class, mock_configure, mock_environment):
        """Test LLM completion with very long prompt."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response to long prompt"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        long_prompt = "This is a very long prompt. " * 1000  # ~30k characters
        result = get_completion(long_prompt)
        
        assert result == "Response to long prompt"
        mock_model.generate_content.assert_called_once_with(long_prompt)


@pytest.mark.unit
@pytest.mark.llm
@pytest.mark.performance  
class TestLLMPerformance:
    """Test LLM performance characteristics."""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_llm_response_time(self, mock_model_class, mock_configure, mock_environment, performance_baseline):
        """Test LLM response time performance."""
        import time
        
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Fast response"
        
        # Simulate realistic API delay
        def slow_generate(prompt):
            time.sleep(0.1)  # Simulate network latency
            return mock_response
            
        mock_model.generate_content.side_effect = slow_generate
        mock_model_class.return_value = mock_model
        
        start_time = time.time()
        result = get_completion("Test prompt")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should be within performance baseline
        assert response_time <= performance_baseline["llm_response_time"]
        assert result == "Fast response"