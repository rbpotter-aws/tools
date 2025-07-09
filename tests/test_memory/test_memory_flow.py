"""
Tests for complex user confirmation flows, retrieval logic, and error handling in memory.py.

This test file specifically targets areas with lower test coverage identified in the journal:
1. Complex user confirmation paths (lines 667-740)
2. Edge cases in document retrieval (lines 813-831)
3. Additional error handling scenarios (lines 855-918)
4. Complex query handling (lines 938-959)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands_tools import memory
from strands_tools.memory import MemoryFormatter, MemoryServiceClient


# ===== FIXTURES FOR TEST DATA =====
@pytest.fixture
def document_responses():
    """Fixture for various document response scenarios."""
    return {
        "get": {
            "indexed": {"documentDetails": [{"status": "INDEXED"}]},
            "processing": {"documentDetails": [{"status": "PROCESSING"}]},
            "failed": {"documentDetails": [{"status": "FAILED"}]},
            "not_found": {"documentDetails": []},
        },
        "retrieve": {
            "with_results": {
                "retrievalResults": [
                    {
                        "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                        "location": {"customDocumentLocation": {"id": "memory_20230509_12345678"}},
                        "score": 0.9,
                    }
                ]
            },
            "empty": {"retrievalResults": []},
            "multiple_results": {
                "retrievalResults": [
                    {
                        "content": {"text": "Some other document"},
                        "location": {"customDocumentLocation": {"id": "wrong_id"}},
                        "score": 0.7,
                    },
                    {
                        "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                        "location": {"customDocumentLocation": {"id": "doc123"}},
                        "score": 0.85,
                    },
                ]
            },
            "plain_text": {
                "retrievalResults": [
                    {
                        "content": {"text": "This is plain text content, not JSON"},
                        "location": {"customDocumentLocation": {"id": "doc123"}},
                        "score": 0.9,
                    }
                ]
            },
            "with_pagination": {
                "retrievalResults": [
                    {
                        "score": 0.85,
                        "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                        "location": {"customDocumentLocation": {"id": "doc123"}},
                    }
                ],
                "nextToken": "pagination_token_123",
            },
        },
        "delete": {
            "success": {"documentDetails": [{"status": "DELETED"}]},
            "in_progress": {"documentDetails": [{"status": "DELETE_IN_PROGRESS"}]},
        },
    }


@pytest.fixture
def user_inputs():
    """Fixture for various user input scenarios."""
    return {
        "confirm": ["y"],
        "cancel_with_reason": ["n", "Changed my mind"],
        "cancel_direct": ["no thanks"],
        "cancel_empty": ["", "Not ready yet"],
    }


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client."""
    client = MagicMock(spec=MemoryServiceClient)
    client.get_data_source_id.return_value = "test_ds_id"
    return client


@pytest.fixture
def mock_memory_formatter():
    """Create a mock memory formatter."""
    formatter = MagicMock(spec=MemoryFormatter)
    return formatter


# ===== USER CONFIRMATION FLOW TESTS =====
@patch.dict(
    os.environ,
    {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "false"},
)
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_store_with_user_confirmation(
    mock_get_user_input,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    user_inputs,
):
    """Test store operation with user confirmation flow."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = user_inputs["confirm"]

    # Configure store operation mocks
    doc_id = "memory_20230509_12345678"
    doc_title = "Test Title"
    mock_memory_service_client.store_document.return_value = (
        {"status": "success"},
        doc_id,
        doc_title,
    )
    mock_memory_formatter.format_store_response.return_value = [
        {"text": "✅ Successfully stored content in knowledge base:"},
        {"text": f"📝 Title: {doc_title}"},
    ]

    # Execute
    result = memory.memory(action="store", content="Test content", title=doc_title)

    # Verify
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    assert "Do you want to proceed with the store operation?" in mock_get_user_input.call_args[0][0]
    mock_memory_service_client.store_document.assert_called_once()


@patch.dict(
    os.environ,
    {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "false"},
)
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_store_with_user_cancellation_and_reason(
    mock_get_user_input,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    user_inputs,
):
    """Test store operation with user cancellation and reason flow."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = user_inputs["cancel_with_reason"]

    # Execute
    result = memory.memory(action="store", content="Test content", title="Test Title")

    # Verify
    assert result["status"] == "error"
    assert "Operation cancelled by the user" in result["content"][0]["text"]
    assert "Changed my mind" in result["content"][0]["text"]
    assert mock_get_user_input.call_count == 2
    mock_memory_service_client.store_document.assert_not_called()


@patch.dict(
    os.environ,
    {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "false"},
)
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_store_with_direct_cancellation(
    mock_get_user_input,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    user_inputs,
):
    """Test store operation with direct cancellation (non-'n' response)."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = user_inputs["cancel_direct"]

    # Execute
    result = memory.memory(action="store", content="Test content", title="Test Title")

    # Verify
    assert result["status"] == "error"
    assert "Operation cancelled by the user" in result["content"][0]["text"]
    assert "no thanks" in result["content"][0]["text"]
    assert mock_get_user_input.call_count == 1  # No second prompt for reason


@patch.dict(
    os.environ,
    {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "false"},
)
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_delete_with_document_preview_and_title(
    mock_get_user_input,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
    user_inputs,
):
    """Test delete operation with successful document preview showing title."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = user_inputs["confirm"]

    doc_id = "memory_20230509_12345678"

    # Configure mocks for preview and delete
    mock_memory_service_client.get_document.return_value = document_responses["get"]["indexed"]
    mock_memory_service_client.retrieve.return_value = document_responses["retrieve"]["with_results"]
    mock_memory_service_client.delete_document.return_value = document_responses["delete"]["success"]
    mock_memory_formatter.format_delete_response.return_value = [
        {"text": "✅ Document deletion deleted:"},
        {"text": f"🔑 Document ID: {doc_id}"},
    ]

    # Execute
    result = memory.memory(action="delete", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    mock_memory_service_client.get_document.assert_called_once()
    mock_memory_service_client.retrieve.assert_called_once()  # For title retrieval
    mock_memory_service_client.delete_document.assert_called_once()


@patch.dict(
    os.environ,
    {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "false"},
)
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_delete_with_preview_error_continues(
    mock_get_user_input,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
    user_inputs,
):
    """Test delete operation continues even when document preview fails."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = user_inputs["confirm"]

    doc_id = "memory_20230509_12345678"

    # Make get_document raise an exception
    mock_memory_service_client.get_document.side_effect = Exception("Error getting document")
    mock_memory_service_client.delete_document.return_value = document_responses["delete"]["success"]
    mock_memory_formatter.format_delete_response.return_value = [
        {"text": "✅ Document deletion deleted:"},
        {"text": f"🔑 Document ID: {doc_id}"},
    ]

    # Execute
    result = memory.memory(action="delete", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    mock_memory_service_client.get_document.assert_called_once()
    mock_memory_service_client.retrieve.assert_not_called()  # Skipped due to error
    mock_memory_service_client.delete_document.assert_called_once()


# ===== DOCUMENT RETRIEVAL EDGE CASES =====
@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("time.sleep")
def test_get_document_with_retry_success(
    mock_sleep,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document that becomes indexed after retry."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # First call: PROCESSING, Second call: INDEXED
    mock_memory_service_client.get_document.side_effect = [
        document_responses["get"]["processing"],
        document_responses["get"]["indexed"],
    ]
    mock_memory_service_client.retrieve.return_value = document_responses["retrieve"]["with_results"]
    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    assert mock_memory_service_client.get_document.call_count == 2
    mock_memory_service_client.retrieve.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("time.sleep")
def test_get_document_still_processing_after_retries(
    mock_sleep,
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document that remains in PROCESSING status after all retries."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # Always return PROCESSING
    mock_memory_service_client.get_document.return_value = document_responses["get"]["processing"]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "error"
    assert "Document is not indexed (status: PROCESSING)" in result["content"][0]["text"]
    assert mock_memory_service_client.get_document.call_count == 4  # Initial + 3 retries


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_fallback_retrieval(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document with first retrieve failing but fallback succeeding."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    mock_memory_service_client.get_document.return_value = document_responses["get"]["indexed"]

    # First retrieve: empty, Second retrieve: success
    mock_memory_service_client.retrieve.side_effect = [
        document_responses["retrieve"]["empty"],
        document_responses["retrieve"]["with_results"],
    ]

    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    assert mock_memory_service_client.retrieve.call_count == 2

    # Check that second call used doc_id as query
    second_call_args = mock_memory_service_client.retrieve.call_args_list[1]
    assert second_call_args[1]["query"] == doc_id


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_multiple_results_filters_correctly(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document correctly filters when multiple results are returned."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "doc123"

    mock_memory_service_client.get_document.return_value = document_responses["get"]["indexed"]

    # First retrieve: empty, Second retrieve: multiple results
    mock_memory_service_client.retrieve.side_effect = [
        document_responses["retrieve"]["empty"],
        document_responses["retrieve"]["multiple_results"],
    ]

    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    assert mock_memory_service_client.retrieve.call_count == 2
    mock_memory_formatter.format_get_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_plain_text_content(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document with non-JSON plain text content."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "doc123"

    mock_memory_service_client.get_document.return_value = document_responses["get"]["indexed"]
    mock_memory_service_client.retrieve.return_value = document_responses["retrieve"]["plain_text"]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "success"
    assert "Document retrieved successfully" in result["content"][0]["text"]
    assert "This is plain text content" in result["content"][3]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_all_retrieval_attempts_fail(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test get document when all retrieval attempts fail."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "doc123"

    mock_memory_service_client.get_document.return_value = document_responses["get"]["indexed"]

    # All retrieve attempts return empty
    mock_memory_service_client.retrieve.return_value = document_responses["retrieve"]["empty"]

    # Execute
    result = memory.memory(action="get", document_id=doc_id)

    # Verify
    assert result["status"] == "error"
    assert "Document found but content could not be retrieved" in result["content"][0]["text"]
    assert mock_memory_service_client.retrieve.call_count == 3  # Three different query attempts


# ===== RETRIEVE OPERATION TESTS =====
@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_validation_error(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test retrieve operation with validation error from AWS."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure retrieve to raise ValidationException
    mock_memory_service_client.retrieve.side_effect = Exception("ValidationException: The provided knowledgeBaseId")

    # Execute
    result = memory.memory(action="retrieve", query="test query")

    # Verify
    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_generic_error(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test retrieve operation with generic error."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure retrieve to raise generic error
    mock_memory_service_client.retrieve.side_effect = Exception("Some other AWS error")

    # Execute
    result = memory.memory(action="retrieve", query="test query")

    # Verify
    assert result["status"] == "error"
    assert "Error during retrieval" in result["content"][0]["text"]
    assert "Some other AWS error" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_pagination(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    document_responses,
):
    """Test retrieve operation with pagination token in response."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    mock_memory_service_client.retrieve.return_value = document_responses["retrieve"]["with_pagination"]
    mock_memory_formatter.format_retrieve_response.return_value = [
        {"text": "Retrieved 1 results with score >= 0.4:"},
        {"text": "➡️ More results available. Use next_token parameter to continue."},
        {"text": "next_token: pagination_token_123"},
    ]

    # Execute
    result = memory.memory(action="retrieve", query="test query", min_score=0.4)

    # Verify
    assert result["status"] == "success"
    assert len(result["content"]) == 3
    assert "pagination_token_123" in result["content"][2]["text"]


# ===== FORMATTER TESTS =====
def test_memory_formatter_comprehensive():
    """Test MemoryFormatter with various scenarios."""
    formatter = MemoryFormatter()

    # Test list response with S3 identifier
    s3_list_response = {
        "documentDetails": [
            {
                "identifier": {"s3": {"uri": "s3://bucket/key"}},
                "status": "INDEXED",
                "updatedAt": "2023-05-09T10:00:00Z",
            }
        ]
    }
    s3_result = formatter.format_list_response(s3_list_response)
    assert "s3://bucket/key" in s3_result[0]["text"]

    # Test retrieve response with score filtering
    retrieve_response = {
        "retrievalResults": [
            {
                "score": 0.9,
                "content": {"text": '{"title": "High Score", "content": "High content"}'},
                "location": {"customDocumentLocation": {"id": "doc1"}},
            },
            {
                "score": 0.3,  # Below threshold
                "content": {"text": '{"title": "Low Score", "content": "Low content"}'},
                "location": {"customDocumentLocation": {"id": "doc2"}},
            },
        ]
    }

    filtered_result = formatter.format_retrieve_response(retrieve_response, 0.5)
    assert "Retrieved 1 results" in filtered_result[0]["text"]
    assert "High Score" in filtered_result[0]["text"]
    assert "Low Score" not in filtered_result[0]["text"]

    # Test delete response with different statuses
    assert "deletion deleted" in formatter.format_delete_response("DELETED", "doc1", "kb1")[0]["text"]
    assert "deletion deleting" in formatter.format_delete_response("DELETING", "doc1", "kb1")[0]["text"]
    assert "deletion failed" in formatter.format_delete_response("FAILED", "doc1", "kb1")[0]["text"]


# ===== SESSION MANAGEMENT TESTS =====
@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_session")
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_memory_with_stored_session(
    mock_get_formatter,
    mock_get_client,
    mock_get_session,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test memory operations using stored session."""
    # Setup mocks
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure mocks
    mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
    mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

    # Execute with custom session key
    result = memory.memory(action="list", session_key="custom_key")

    # Verify
    assert result["status"] == "success"
    mock_get_session.assert_called_once_with("custom_key")
    mock_get_client.assert_called_once_with(region=None, session=mock_session)


# ===== EDGE CASE TESTS =====
@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_memory_with_custom_datasource_id(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test memory operations with custom datasource ID parameter."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure mocks
    mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
    mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

    # Execute with custom datasource ID
    result = memory.memory(action="list", STRANDS_KNOWLEDGE_BASE_MEMORY_DATASOURCE_ID="custom_ds789")

    # Verify
    assert result["status"] == "success"

    # Should validate and use the custom datasource ID
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb", "custom_ds789")

    # The list_documents should use the returned data source ID
    mock_memory_service_client.list_documents.assert_called_once_with("test123kb", "test_ds_id", 50, None)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
def test_memory_service_client_lazy_loading():
    """Test MemoryServiceClient lazy loading of clients."""
    with patch("boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_agent_client = MagicMock()
        mock_runtime_client = MagicMock()

        mock_session_class.return_value = mock_session

        def client_factory(service, **kwargs):
            if service == "bedrock-agent":
                return mock_agent_client
            elif service == "bedrock-agent-runtime":
                return mock_runtime_client
            raise ValueError(f"Unknown service: {service}")

        mock_session.client.side_effect = client_factory

        # Create client
        client = MemoryServiceClient(region="us-east-1")

        # Verify no clients created yet
        mock_session.client.assert_not_called()

        # Access agent_client (triggers lazy loading)
        agent = client.agent_client
        assert agent == mock_agent_client
        mock_session.client.assert_called_with("bedrock-agent", region_name="us-east-1")

        # Access again (should use cache)
        mock_session.client.reset_mock()
        agent2 = client.agent_client
        assert agent2 == agent
        mock_session.client.assert_not_called()

        # Access runtime_client (triggers lazy loading)
        runtime = client.runtime_client
        assert runtime == mock_runtime_client
        mock_session.client.assert_called_with("bedrock-agent-runtime", region_name="us-east-1")
