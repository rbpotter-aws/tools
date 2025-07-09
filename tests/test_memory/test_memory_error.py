"""
Tests for error handling in memory.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands_tools.memory import memory


# ===== FIXTURES FOR ERROR SCENARIOS =====
@pytest.fixture
def validation_test_cases():
    """Fixture for ID validation test cases."""
    return {
        "valid_kb_ids": ["test123kb", "ABC123xyz", "kb999", "ALLCAPS123"],
        "invalid_kb_ids": ["test-123-kb", "test123kb!!", "test@123", "kb-with-hyphens"],
        "valid_ds_ids": ["ds123ds", "datasource456", "DS999", "mixedCase123"],
        "invalid_ds_ids": ["ds-456-ds!!", "ds@123", "data-source-1", "ds.with.dots"],
    }


@pytest.fixture
def error_responses():
    """Fixture for various error response scenarios."""
    return {
        "data_source_errors": {
            "not_custom": {
                "status": "error",
                "content": [{"text": "❌ Provided DataSource ID: ds456 is not of type 'CUSTOM'"}],
            },
            "no_sources_found": {
                "status": "error",
                "content": [
                    {"text": "❌ Failed to get data source ID: No data sources found for knowledge base test123kb"}
                ],
            },
            "no_custom_found": {
                "status": "error",
                "content": [
                    {
                        "text": "❌ Failed to get data source ID: No CUSTOM data source found for "
                        "knowledge base test123kb"
                    }
                ],
            },
        },
        "document_errors": {
            "not_found": {"documentDetails": []},
            "not_indexed": {"documentDetails": [{"status": "PROCESSING"}]},
            "failed": {"documentDetails": [{"status": "FAILED"}]},
        },
        "api_errors": {
            "generic": Exception("API Error"),
            "validation": Exception("ValidationException: Invalid knowledgeBaseId"),
            "access_denied": Exception("AccessDeniedException: User is not authorized"),
            "throttling": Exception("ThrottlingException: Rate exceeded"),
        },
    }


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client with default behavior."""
    client = MagicMock()
    client.get_data_source_id.return_value = "ds123"  # Default to valid
    return client


# ===== ID VALIDATION TESTS =====
@pytest.mark.parametrize(
    "invalid_kb_id",
    [
        "invalid-kb-id",  # Contains hyphens
        "test123kb!!",  # Contains special characters
        "kb@123",  # Contains @
        "kb.with.dots",  # Contains dots
        "kb with spaces",  # Contains spaces
    ],
)
def test_invalid_kb_id_format(invalid_kb_id):
    """Test error handling for various invalid knowledge base ID formats."""
    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID=invalid_kb_id)

    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]
    assert invalid_kb_id in result["content"][0]["text"]


@pytest.mark.parametrize(
    "valid_kb_id",
    [
        "test123kb",
        "ABC123xyz",
        "kb999",
        "ALLCAPS123",
        "123456789",
        "a",  # Single character
    ],
)
@patch("strands_tools.memory.get_memory_service_client")
def test_valid_kb_id_format(mock_get_client, mock_memory_service_client, valid_kb_id):
    """Test that valid knowledge base IDs are accepted."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.list_documents.return_value = {"documentDetails": []}

    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID=valid_kb_id)

    # Should not have validation error
    assert "Invalid knowledge base ID format" not in str(result)


@pytest.mark.parametrize(
    "invalid_ds_id",
    [
        "invalid-ds-id",  # Contains hyphens
        "ds456ds!!",  # Contains special characters
        "ds@123",  # Contains @
        "data-source-1",  # Contains hyphens
        "ds.with.dots",  # Contains dots
    ],
)
def test_invalid_datasource_id_format(invalid_ds_id):
    """Test error handling for various invalid datasource ID formats."""
    result = memory(
        action="list",
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
        STRANDS_KNOWLEDGE_BASE_MEMORY_DATASOURCE_ID=invalid_ds_id,
    )

    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]
    assert invalid_ds_id in result["content"][0]["text"]


# ===== DATA SOURCE ERROR TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
def test_data_source_not_custom_type(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when provided data source is not CUSTOM type."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_data_source_id.return_value = error_responses["data_source_errors"]["not_custom"]

    result = memory(
        action="list",
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
        STRANDS_KNOWLEDGE_BASE_MEMORY_DATASOURCE_ID="ds456",
    )

    assert result["status"] == "error"
    assert "not of type 'CUSTOM'" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_data_source_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when getting data source ID fails with API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_data_source_id.side_effect = error_responses["api_errors"]["generic"]

    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Failed to get data source ID" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_no_data_sources_found(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when no data sources are found."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_data_source_id.return_value = error_responses["data_source_errors"][
        "no_sources_found"
    ]

    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "No data sources found" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_no_custom_data_source_found(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when no CUSTOM type data source is found."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_data_source_id.return_value = error_responses["data_source_errors"][
        "no_custom_found"
    ]

    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "No CUSTOM data source found" in result["content"][0]["text"]


# ===== ACTION-SPECIFIC VALIDATION TESTS =====
def test_missing_knowledge_base_id():
    """Test error when no knowledge base ID is provided."""
    with patch.dict(os.environ, {}, clear=True):
        result = memory(action="list")

        assert result["status"] == "error"
        assert "No knowledge base ID provided" in result["content"][0]["text"]


def test_invalid_action():
    """Test error handling for invalid action."""
    result = memory(action="invalid_action", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Invalid action: invalid_action" in result["content"][0]["text"]
    assert "Must be 'store', 'delete', 'list', 'get', or 'retrieve'" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_store_empty_content(mock_get_client):
    """Test error handling for empty content in store operation."""
    result = memory(action="store", content="", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Content cannot be empty" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_store_whitespace_only_content(mock_get_client):
    """Test error handling for whitespace-only content in store operation."""
    result = memory(action="store", content="   \n\t   ", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Content cannot be empty" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_delete_missing_document_id(mock_get_client):
    """Test error handling for missing document ID in delete operation."""
    result = memory(action="delete", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Document ID cannot be empty for delete operation" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_get_missing_document_id(mock_get_client):
    """Test error handling for missing document ID in get operation."""
    result = memory(action="get", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Document ID cannot be empty for get operation" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_missing_query(mock_get_client):
    """Test error handling for missing query in retrieve operation."""
    result = memory(action="retrieve", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "No query provided for retrieval" in result["content"][0]["text"]


# ===== PARAMETER VALIDATION TESTS =====
@pytest.mark.parametrize("invalid_min_score", [-0.1, 1.1, 2.0, -1.0])
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_invalid_min_score(mock_get_client, invalid_min_score):
    """Test error handling for invalid min_score values in retrieve operation."""
    result = memory(
        action="retrieve",
        query="test query",
        min_score=invalid_min_score,
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
    )

    assert result["status"] == "error"
    assert "min_score must be between 0.0 and 1.0" in result["content"][0]["text"]


@pytest.mark.parametrize("invalid_max_results", [0, -1, 1001, 2000])
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_invalid_max_results(mock_get_client, invalid_max_results):
    """Test error handling for invalid max_results values in retrieve operation."""
    result = memory(
        action="retrieve",
        query="test query",
        max_results=invalid_max_results,
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
    )

    assert result["status"] == "error"
    assert "max_results must be between 1 and 1000" in result["content"][0]["text"]


@pytest.mark.parametrize("invalid_max_results", [0, -1, 1001, 2000])
@patch("strands_tools.memory.get_memory_service_client")
def test_list_invalid_max_results(mock_get_client, invalid_max_results):
    """Test error handling for invalid max_results values in list operation."""
    result = memory(
        action="list",
        max_results=invalid_max_results,
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
    )

    assert result["status"] == "error"
    assert "max_results must be between 1 and 1000" in result["content"][0]["text"]


# ===== API ERROR TESTS =====
@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
def test_store_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when store operation fails with API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.store_document.side_effect = error_responses["api_errors"]["generic"]

    result = memory(
        action="store",
        content="Test content",
        title="Test Title",
        STRANDS_KNOWLEDGE_BASE_ID="test123kb",
    )

    assert result["status"] == "error"
    assert "Error during store operation" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
def test_delete_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when delete operation fails with API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.delete_document.side_effect = error_responses["api_errors"]["generic"]

    result = memory(action="delete", document_id="doc123", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Error during delete operation" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_get_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when get operation fails with API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_document.side_effect = error_responses["api_errors"]["generic"]

    result = memory(action="get", document_id="doc123", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Error retrieving document" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when retrieve operation fails with generic API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.retrieve.side_effect = error_responses["api_errors"]["generic"]

    result = memory(action="retrieve", query="test query", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Error during retrieval" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_validation_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when retrieve encounters a validation error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.retrieve.side_effect = error_responses["api_errors"]["validation"]

    result = memory(action="retrieve", query="test query", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]


# ===== DOCUMENT STATUS ERROR TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
def test_get_document_not_found(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when document is not found."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.get_document.return_value = error_responses["document_errors"]["not_found"]

    result = memory(action="get", document_id="doc123", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Document not found: doc123" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
@patch("time.sleep")  # Mock sleep to avoid test delays
def test_get_document_not_indexed_after_retries(
    mock_sleep, mock_get_client, mock_memory_service_client, error_responses
):
    """Test error handling when document is not indexed after max retries."""
    mock_get_client.return_value = mock_memory_service_client
    # Always return PROCESSING status
    mock_memory_service_client.get_document.return_value = error_responses["document_errors"]["not_indexed"]

    result = memory(action="get", document_id="doc123", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Document is not indexed (status: PROCESSING)" in result["content"][0]["text"]
    assert mock_memory_service_client.get_document.call_count == 4  # Initial + 3 retries


# ===== EDGE CASE TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
def test_delete_without_document_details(mock_get_client, mock_memory_service_client):
    """Test successful delete when response has no document details."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.delete_document.return_value = {}  # Empty response

    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
        result = memory(action="delete", document_id="doc123", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "success"
    assert "Document deletion request accepted" in result["content"][0]["text"]


@patch("strands_tools.memory.get_memory_service_client")
def test_list_api_error(mock_get_client, mock_memory_service_client, error_responses):
    """Test error handling when list operation fails with API error."""
    mock_get_client.return_value = mock_memory_service_client
    mock_memory_service_client.list_documents.side_effect = error_responses["api_errors"]["generic"]

    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID="test123kb")

    assert result["status"] == "error"
    assert "Error during list operation" in result["content"][0]["text"]
    assert "API Error" in result["content"][0]["text"]
