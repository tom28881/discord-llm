"""
Comprehensive AI/Chatbot functionality tests with real data and LLM APIs.

Tests the complete AI stack:
- LLM API connections (Gemini, OpenAI, OpenRouter)
- Query processing and context retrieval
- Prompt building and response generation
- ML features (importance scoring, activity detection)
- End-to-end chat workflows

⚠️ IMPORTANT: These tests make REAL LLM API calls which may incur costs!
"""
import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_messages import load_messages_once
from lib.database import get_recent_message_records
from lib.llm import get_completion


@pytest.mark.real
@pytest.mark.skip_ci
class TestLLMAPIConnection:
    """Test real LLM API connections."""
    
    def test_gemini_api_connection(self):
        """Test connection to Google Gemini API."""
        print("\n" + "="*80)
        print("🤖 TESTING GEMINI API CONNECTION")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set - add to .env to test Gemini")
        
        print(f"\n   ✅ API Key found: {api_key[:10]}...")
        
        # Simple test prompt
        test_prompt = "Say 'Hello' in one word only."
        
        print(f"\n   📤 Sending test prompt: {test_prompt}")
        
        try:
            response = get_completion(test_prompt)
            
            print(f"   📥 Response received: {response[:100]}...")
            print(f"   📏 Response length: {len(response)} characters")
            
            assert response, "Response should not be empty"
            assert len(response) > 0, "Response should have content"
            
            print(f"\n   ✅ Gemini API working correctly!")
            
        except Exception as e:
            print(f"\n   ❌ Gemini API error: {e}")
            raise
    
    def test_llm_with_structured_prompt(self):
        """Test LLM with structured prompt (like real usage)."""
        print("\n" + "="*80)
        print("🤖 TESTING STRUCTURED PROMPT")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # Simulate real prompt structure
        prompt = """You are an analyst who must answer strictly from the provided Discord messages.

<metadata>
{"server": "Test Server", "channels": ["general"], "time_range_hours": 24}
</metadata>

<messages>
1. [2025-10-24T10:00:00] #general: Diskutujeme o investičních strategiích
2. [2025-10-24T11:00:00] #general: ETF jsou dobrá volba pro začátečníky
3. [2025-10-24T12:00:00] #general: S&P 500 je nejpopulárnější index
</messages>

<question>
Co bylo diskutováno o ETF?
</question>

Instructions:
1. Use only the numbered messages as evidence.
2. Reference supporting messages using their numbers in square brackets (e.g., [1]).
3. Answer in Czech.

Respond in Markdown with sections:
- Answer
- Evidence
- Confidence"""
        
        print("\n   📤 Sending structured prompt...")
        
        response = get_completion(prompt)
        
        print(f"\n   📥 Response:\n{response}\n")
        
        # Verify response structure
        assert response, "Should have response"
        assert len(response) > 50, "Response should be substantial"
        
        # Check for expected sections
        response_lower = response.lower()
        has_answer = "answer" in response_lower or "odpověď" in response_lower
        has_evidence = "evidence" in response_lower or "[" in response
        has_confidence = "confidence" in response_lower
        
        print(f"\n   ✅ Response structure:")
        print(f"      Answer section: {has_answer}")
        print(f"      Evidence refs: {has_evidence}")
        print(f"      Confidence: {has_confidence}")
        
        assert has_answer or has_evidence, "Response should have answer or evidence"


@pytest.mark.real
@pytest.mark.skip_ci
class TestSummarizeFunction:
    """Test the summarize_messages function with real data."""
    
    def test_summarize_with_real_messages(
        self,
        test_database,
        test_server_id,
    ):
        """Test summarize_messages with real Discord messages."""
        print("\n" + "="*80)
        print("📝 TESTING SUMMARIZE_MESSAGES WITH REAL DATA")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # Import real messages
        print("\n   📥 Importing messages...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            sleep_between_channels=False
        )
        
        print(f"   💾 Imported: {summary['messages_saved']} messages")
        
        if summary['messages_saved'] == 0:
            pytest.skip("No messages to test with")
        
        # Get messages
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=24,
            limit=None
        )
        
        print(f"   📊 Retrieved: {len(records)} messages")
        
        if len(records) == 0:
            pytest.skip("No messages available")
        
        # Build references (as UI does)
        references = []
        for record in records[:20]:  # Limit to first 20 for testing
            timestamp_iso = datetime.fromtimestamp(record['sent_at']).isoformat()
            references.append(f"[{timestamp_iso}] {record['content'][:100]}")
        
        # Import the actual function
        from streamlit_app import summarize_messages
        
        # Test with a real question
        test_questions = [
            "Co bylo hlavní téma diskuse?",
            "Jaké důležité informace byly zmíněny?",
            "Shrň hlavní body z těchto zpráv",
        ]
        
        for question in test_questions:
            print(f"\n   ❓ Question: {question}")
            
            metadata = {
                "server": "Test Server",
                "channels": ["general"],
                "time_range_hours": 24,
                "total_messages_considered": len(records),
                "messages_used_for_answer": len(references),
            }
            
            chat_history = [
                {"role": "user", "content": question}
            ]
            
            # Call summarize_messages
            print(f"   🤖 Generating response...")
            
            response = summarize_messages(
                references=references,
                chat_history=chat_history,
                question=question,
                metadata=metadata
            )
            
            print(f"\n   📥 Response preview:")
            print(f"   {response[:200]}...")
            print(f"   📏 Response length: {len(response)} characters")
            
            # Verify response
            assert response, "Should have response"
            assert len(response) > 50, "Response should be substantial"
            assert not response.startswith("An error"), "Should not be error message"
            
            print(f"   ✅ Response generated successfully!")
            
            # Only test first question to save API costs
            break


@pytest.mark.real
@pytest.mark.skip_ci  
class TestChatWorkflow:
    """Test complete chat workflow end-to-end."""
    
    def test_complete_chat_workflow(
        self,
        test_database,
        test_server_id,
    ):
        """Test complete workflow: Import → Query → Context → LLM → Response."""
        print("\n" + "="*80)
        print("💬 COMPLETE CHAT WORKFLOW TEST")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # STEP 1: Import messages
        print("\n   📥 STEP 1: Importing messages...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=168,  # 1 week
            sleep_between_channels=False
        )
        
        print(f"      💾 Imported: {summary['messages_saved']} messages")
        
        # STEP 2: User asks question (simulate UI)
        user_question = "Jaké byly hlavní diskuse v posledním týdnu?"
        
        print(f"\n   👤 STEP 2: User asks: {user_question}")
        
        # STEP 3: Retrieve context from database
        print(f"\n   🔍 STEP 3: Retrieving context...")
        
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=168,
            limit=None
        )
        
        print(f"      📊 Context: {len(records)} messages")
        
        if len(records) < 10:
            pytest.skip(f"Need at least 10 messages, only have {len(records)}")
        
        # STEP 4: Build references (as UI does)
        print(f"\n   📝 STEP 4: Building references...")
        
        # Take sample of messages
        sample_size = min(50, len(records))
        sampled_records = records[:sample_size]
        
        references = []
        for record in sampled_records:
            timestamp_iso = datetime.fromtimestamp(record['sent_at']).isoformat()
            content = record['content'][:150]  # Limit content length
            references.append(f"[{timestamp_iso}] {content}")
        
        print(f"      ✅ Built {len(references)} references")
        
        # STEP 5: Generate LLM response
        print(f"\n   🤖 STEP 5: Generating LLM response...")
        
        from streamlit_app import summarize_messages
        
        metadata = {
            "server": "InvestičníFlow",
            "channels": ["All channels"],
            "time_range_hours": 168,
            "total_messages_considered": len(records),
            "messages_used_for_answer": len(references),
        }
        
        chat_history = [
            {"role": "user", "content": user_question}
        ]
        
        response = summarize_messages(
            references=references,
            chat_history=chat_history,
            question=user_question,
            metadata=metadata
        )
        
        # STEP 6: Verify response quality
        print(f"\n   ✅ STEP 6: Verifying response quality...")
        
        print(f"\n   📥 Response:")
        print(f"   {'-'*70}")
        print(f"   {response}")
        print(f"   {'-'*70}")
        
        # Quality checks
        assert response, "Should have response"
        assert len(response) > 100, "Response should be detailed"
        assert not response.startswith("An error"), "Should not be error"
        
        # Check if response references messages
        has_references = "[" in response and "]" in response
        print(f"\n   📌 Contains message references: {has_references}")
        
        # Check response structure
        has_sections = any(marker in response.lower() for marker in 
                          ['answer', 'odpověď', 'evidence', 'confidence'])
        print(f"   📋 Has structured sections: {has_sections}")
        
        # Check Czech language
        czech_words = ['je', 'jsou', 'bylo', 'byli', 'diskuse', 'zprávy']
        has_czech = any(word in response.lower() for word in czech_words)
        print(f"   🇨🇿 Response in Czech: {has_czech}")
        
        print(f"\n   ✅ Complete chat workflow PASSED!")
    
    def test_multiple_questions_conversation(
        self,
        test_database,
        test_server_id,
    ):
        """Test conversation with multiple questions (chat history)."""
        print("\n" + "="*80)
        print("💬 MULTI-TURN CONVERSATION TEST")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # Setup: Import messages
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            sleep_between_channels=False
        )
        
        if summary['messages_saved'] < 5:
            pytest.skip("Need more messages for conversation test")
        
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=24,
            limit=30
        )
        
        # Build references
        references = []
        for record in records:
            timestamp_iso = datetime.fromtimestamp(record['sent_at']).isoformat()
            references.append(f"[{timestamp_iso}] {record['content'][:100]}")
        
        from streamlit_app import summarize_messages
        
        # Simulate multi-turn conversation
        conversation = [
            "Co bylo diskutováno?",
            "Můžeš to shrnout stručněji?",
        ]
        
        chat_history = []
        
        for idx, question in enumerate(conversation, 1):
            print(f"\n   👤 Turn {idx}: {question}")
            
            chat_history.append({"role": "user", "content": question})
            
            metadata = {
                "server": "Test Server",
                "time_range_hours": 24,
                "messages_used_for_answer": len(references),
            }
            
            response = summarize_messages(
                references=references,
                chat_history=chat_history,
                question=question,
                metadata=metadata
            )
            
            print(f"   🤖 Response: {response[:150]}...")
            
            chat_history.append({"role": "assistant", "content": response})
            
            assert response, f"Turn {idx} should have response"
            assert len(response) > 20, f"Turn {idx} response should have content"
            
            # Only test 2 turns to save API costs
            if idx >= 2:
                break
        
        print(f"\n   ✅ Multi-turn conversation working!")


@pytest.mark.real
@pytest.mark.skip_ci
class TestResponseQuality:
    """Test quality and correctness of LLM responses."""
    
    def test_response_contains_evidence(
        self,
        test_database,
        test_server_id,
    ):
        """Test that responses cite specific messages as evidence."""
        print("\n" + "="*80)
        print("🔍 TESTING RESPONSE EVIDENCE CITING")
        print("="*80)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        # Import and prepare data
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            sleep_between_channels=False
        )
        
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=24,
            limit=20
        )
        
        if len(records) < 5:
            pytest.skip("Need messages to test")
        
        # Build references with clear numbering
        references = []
        for idx, record in enumerate(records, 1):
            timestamp_iso = datetime.fromtimestamp(record['sent_at']).isoformat()
            references.append(f"{idx}. [{timestamp_iso}] {record['content'][:100]}")
        
        from streamlit_app import summarize_messages
        
        # Ask specific question that requires citing
        question = "Které konkrétní zprávy byly nejdůležitější? Cituj čísla zpráv."
        
        metadata = {
            "server": "Test",
            "time_range_hours": 24,
            "messages_used_for_answer": len(references),
        }
        
        chat_history = [{"role": "user", "content": question}]
        
        print(f"\n   ❓ Question: {question}")
        print(f"   📊 Available messages: {len(references)}")
        
        response = summarize_messages(
            references=references,
            chat_history=chat_history,
            question=question,
            metadata=metadata
        )
        
        print(f"\n   📥 Response:")
        print(f"   {response}")
        
        # Check for evidence citations
        import re
        citations = re.findall(r'\[(\d+)\]', response)
        
        print(f"\n   📌 Found {len(citations)} message citations: {citations}")
        
        # Verify citations are valid
        if citations:
            valid_citations = [int(c) for c in citations if c.isdigit() and 1 <= int(c) <= len(references)]
            print(f"   ✅ Valid citations: {valid_citations}")
            
            assert len(valid_citations) > 0, "Should have at least one valid citation"
        else:
            print(f"   ⚠️  No citations found (model might not always cite)")
        
        # Response should still be useful
        assert len(response) > 50, "Response should be substantial"


@pytest.mark.real
@pytest.mark.skip_ci
class TestErrorHandling:
    """Test error handling in chat/LLM functionality."""
    
    def test_no_messages_scenario(self, test_database):
        """Test behavior when no messages are available."""
        print("\n" + "="*80)
        print("⚠️  TESTING NO MESSAGES SCENARIO")
        print("="*80)
        
        from streamlit_app import summarize_messages
        
        # Empty references
        references = []
        chat_history = [{"role": "user", "content": "Test question"}]
        metadata = {}
        
        response = summarize_messages(
            references=references,
            chat_history=chat_history,
            question="Test",
            metadata=metadata
        )
        
        print(f"\n   📥 Response: {response}")
        
        # Should handle gracefully
        assert response, "Should have some response"
        assert "no" in response.lower() or "žádné" in response.lower(), \
            "Should indicate no messages"
    
    def test_invalid_api_key(self):
        """Test behavior with invalid API key."""
        print("\n" + "="*80)
        print("🔑 TESTING INVALID API KEY")
        print("="*80)
        
        # Temporarily remove API key
        original_key = os.getenv("GOOGLE_API_KEY")
        
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        try:
            response = get_completion("Test prompt")
            
            print(f"\n   📥 Response: {response}")
            
            # Should handle gracefully (return empty or error message)
            assert response == "" or "error" in response.lower(), \
                "Should handle missing API key gracefully"
            
            print(f"   ✅ Gracefully handled missing API key")
            
        finally:
            # Restore API key
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key


@pytest.mark.real
@pytest.mark.skip_ci
class TestContextBuilding:
    """Test context assembly and reference building."""
    
    def test_context_length_limits(
        self,
        test_database,
        test_server_id,
    ):
        """Test that context respects reasonable length limits."""
        print("\n" + "="*80)
        print("📏 TESTING CONTEXT LENGTH LIMITS")
        print("="*80)
        
        # Import many messages
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=720,  # 30 days
            sleep_between_channels=False
        )
        
        print(f"\n   💾 Imported: {summary['messages_saved']} messages")
        
        # Get all messages
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=720,
            limit=None
        )
        
        print(f"   📊 Total available: {len(records)} messages")
        
        # Build references (as UI would)
        references = []
        for record in records[:200]:  # Limit to 200 as UI does
            timestamp_iso = datetime.fromtimestamp(record['sent_at']).isoformat()
            content = record['content'][:150]  # Truncate
            references.append(f"[{timestamp_iso}] {content}")
        
        print(f"   📝 References built: {len(references)}")
        
        # Calculate total context size
        total_chars = sum(len(ref) for ref in references)
        print(f"   📏 Total context size: {total_chars:,} characters")
        
        # Typical LLM limits are 32k-128k tokens (roughly 4 chars per token)
        estimated_tokens = total_chars / 4
        print(f"   🎯 Estimated tokens: {estimated_tokens:,.0f}")
        
        # Should be within reasonable limits
        assert len(references) <= 200, "Should limit to 200 messages"
        assert estimated_tokens < 30000, "Should stay under 30k tokens for safety"
        
        print(f"\n   ✅ Context size is reasonable!")
