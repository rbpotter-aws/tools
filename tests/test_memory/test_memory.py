"""
Tests for the memory tool using the Agent interface.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands_tools import memory
from strands_tools.memory import MemoryFormatter, MemoryServiceClient


# ===== FIXTURES FOR TEST DATA =====
@pytest.fixture
def agent():
    """Create an agent with the memory tool loaded."""
    return Agent(tools=[memory])


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client with default behavior."""
    client = MagicMock(spec=MemoryServiceClient)
    client.get_data_source_id.return_value = "ds123"  # Default to valid datasource
    return client


@pytest.fixture
def mock_memory_formatter():
    """Create a mock memory formatter."""
    formatter = MagicMock(spec=MemoryFormatter)
    return formatter


@pytest.fixture
def api_responses():
    """Fixture for common API response patterns."""
    return {
        "list": {
            "empty": {"documentDetails": []},
            "with_documents": {
                "documentDetails": [
                    {
                        "identifier": {"custom": {"id": "doc123"}},
                        "status": "INDEXED",
                        "updatedAt": "2023-05-09T10:00:00Z",
                    }
                ]
            },
            "with_pagination": {
                "documentDetails": [
                    {
                        "identifier": {"custom": {"id": "doc123"}},
                        "status": "INDEXED",
                        "updatedAt": "2023-05-09T10:00:00Z",
                    }
                ],
                "nextToken": "next_page_token",
            },
        },
        "store": {
            "success": {"status": "success"},
            "doc_id": "memory_20230509_12345678",
            "doc_title": "Test Title",
        },
        "delete": {
            "success": {"documentDetails": [{"status": "DELETED"}]},
            "no_details": {},  # Empty response
        },
        "get": {
            "document_found": {"documentDetails": [{"status": "INDEXED"}]},
            "document_content": {
                "retrievalResults": [
                    {
                        "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                        "location": {"customDocumentLocation": {"id": "doc123"}},
                    }
                ]
            },
        },
        "retrieve": {
            "with_results": {
                "retrievalResults": [
                    {
                        "score": 0.85,
                        "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                        "location": {"customDocumentLocation": {"id": "memory_20230509_12345678"}},
                    }
                ]
            }
        },
    }


@pytest.fixture
def formatted_responses():
    """Fixture for formatter output patterns."""
    return {
        "list": {
            "empty": [{"text": "No documents found."}],
            "with_documents": [{"text": "Found 1 documents:"}],
        },
        "store": {
            "success": [
                {"text": "✅ Successfully stored content in knowledge base:"},
                {"text": "📝 Title: Test Title"},
            ]
        },
        "delete": {
            "success": [
                {"text": "✅ Document deletion deleted:"},
                {"text": "🔑 Document ID: doc123"},
            ]
        },
        "get": {
            "success": [
                {"text": "✅ Document retrieved successfully:"},
                {"text": "📝 Title: Test Title"},
            ]
        },
        "retrieve": {"success": [{"text": "Retrieved 1 results with score >= 0.4:"}]},
    }


@pytest.fixture
def environment_configs():
    """Fixture for different environment configurations."""
    return {
        "default": {
            "STRANDS_KNOWLEDGE_BASE_ID": "test123kb",
            "BYPASS_TOOL_CONSENT": "false",
        },
        "bypass_consent": {
            "STRANDS_KNOWLEDGE_BASE_ID": "test123kb",
            "BYPASS_TOOL_CONSENT": "true",
        },
        "custom_region": {
            "STRANDS_KNOWLEDGE_BASE_ID": "test123kb",
            "AWS_REGION": "eu-west-1",
        },
        "no_kb_id": {
            # Missing STRANDS_KNOWLEDGE_BASE_ID
            "BYPASS_TOOL_CONSENT": "true"
        },
    }


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if (
        isinstance(result, dict)
        and "content" in result
        and isinstance(result["content"], list)
        and len(result["content"]) > 0
    ):
        return result["content"][0]["text"]
    return str(result)


# ===== LIST OPERATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_list_documents_empty(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test list documents when no documents exist."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.list_documents.return_value = api_responses["list"]["empty"]
        mock_memory_formatter.format_list_response.return_value = formatted_responses["list"]["empty"]

        # Execute
        result = memory.memory(action="list")

        # Verify
        assert result["status"] == "success"
        assert "No documents found" in extract_result_text(result)
        mock_memory_service_client.list_documents.assert_called_once_with("test123kb", "ds123", 50, None)


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_list_documents_with_results(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test list documents with existing documents."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.list_documents.return_value = api_responses["list"]["with_documents"]
        mock_memory_formatter.format_list_response.return_value = formatted_responses["list"]["with_documents"]

        # Execute
        result = memory.memory(action="list")

        # Verify
        assert result["status"] == "success"
        assert "Found 1 documents:" in extract_result_text(result)


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_list_with_custom_parameters(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    environment_configs,
):
    """Test list documents with custom max_results and pagination."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
        mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

        # Execute with custom parameters
        result = memory.memory(action="list", max_results=100, next_token="custom_token")

        # Verify parameters were passed correctly
        assert result["status"] == "success"
        mock_memory_service_client.list_documents.assert_called_once_with("test123kb", "ds123", 100, "custom_token")


# ===== STORE OPERATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_store_document_with_bypass_consent(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test store document with BYPASS_TOOL_CONSENT enabled."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        store_data = api_responses["store"]
        mock_memory_service_client.store_document.return_value = (
            store_data["success"],
            store_data["doc_id"],
            store_data["doc_title"],
        )
        mock_memory_formatter.format_store_response.return_value = formatted_responses["store"]["success"]

        # Execute
        result = memory.memory(action="store", content="Test content", title="Test Title")

        # Verify
        assert result["status"] == "success"
        assert "Successfully stored content" in extract_result_text(result)
        mock_memory_service_client.store_document.assert_called_once_with(
            "test123kb", "ds123", "Test content", "Test Title"
        )


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_store_document_without_title(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test store document without providing a title."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure to capture the auto-generated title
        def store_side_effect(kb_id, ds_id, content, title):
            assert "Memory Entry" in title  # Auto-generated title
            return api_responses["store"]["success"], "doc123", title

        mock_memory_service_client.store_document.side_effect = store_side_effect
        mock_memory_formatter.format_store_response.return_value = formatted_responses["store"]["success"]

        # Execute without title
        result = memory.memory(action="store", content="Test content")

        # Verify
        assert result["status"] == "success"
        mock_memory_service_client.store_document.assert_called_once()


# ===== DELETE OPERATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_delete_document_with_bypass_consent(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test delete document with BYPASS_TOOL_CONSENT enabled."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.delete_document.return_value = api_responses["delete"]["success"]
        mock_memory_formatter.format_delete_response.return_value = formatted_responses["delete"]["success"]

        # Execute
        result = memory.memory(action="delete", document_id="doc123")

        # Verify
        assert result["status"] == "success"
        assert "Document deletion" in extract_result_text(result)
        mock_memory_service_client.delete_document.assert_called_once_with("test123kb", "ds123", "doc123")


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_delete_document_no_details_response(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    environment_configs,
):
    """Test delete document when response has no document details."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure empty response
        mock_memory_service_client.delete_document.return_value = api_responses["delete"]["no_details"]

        # Execute
        result = memory.memory(action="delete", document_id="doc123")

        # Verify
        assert result["status"] == "success"
        assert "Document deletion request accepted" in result["content"][0]["text"]


# ===== GET OPERATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_success(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test successful document retrieval."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.get_document.return_value = api_responses["get"]["document_found"]
        mock_memory_service_client.retrieve.return_value = api_responses["get"]["document_content"]
        mock_memory_formatter.format_get_response.return_value = formatted_responses["get"]["success"]

        # Execute
        result = memory.memory(action="get", document_id="doc123")

        # Verify
        assert result["status"] == "success"
        assert "Document retrieved successfully" in extract_result_text(result)
        mock_memory_service_client.get_document.assert_called_once_with("test123kb", "ds123", "doc123")
        mock_memory_service_client.retrieve.assert_called_once()


# ===== RETRIEVE OPERATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_documents(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test retrieve documents with semantic search."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.retrieve.return_value = api_responses["retrieve"]["with_results"]
        mock_memory_formatter.format_retrieve_response.return_value = formatted_responses["retrieve"]["success"]

        # Execute
        result = memory.memory(action="retrieve", query="test query")

        # Verify
        assert result["status"] == "success"
        assert "Retrieved 1 results" in extract_result_text(result)
        mock_memory_service_client.retrieve.assert_called_once_with(
            kb_id="test123kb", query="test query", max_results=50, next_token=None
        )


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_custom_parameters(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test retrieve with custom min_score and max_results."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure responses
        mock_memory_service_client.retrieve.return_value = api_responses["retrieve"]["with_results"]
        mock_memory_formatter.format_retrieve_response.return_value = formatted_responses["retrieve"]["success"]

        # Execute with custom parameters
        result = memory.memory(
            action="retrieve",
            query="test query",
            min_score=0.7,
            max_results=20,
            next_token="page_token",
        )

        # Verify
        assert result["status"] == "success"
        mock_memory_service_client.retrieve.assert_called_once_with(
            kb_id="test123kb",
            query="test query",
            max_results=20,
            next_token="page_token",
        )
        mock_memory_formatter.format_retrieve_response.assert_called_once_with(
            api_responses["retrieve"]["with_results"],
            0.7,  # min_score passed to formatter
        )


# ===== REGION CONFIGURATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
def test_custom_region_parameter(mock_get_client, environment_configs):
    """Test using a custom region parameter."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Execute with custom region
        memory.memory(action="list", region_name="ap-southeast-2")

        # Verify region was passed correctly
        mock_get_client.assert_called_once_with(region="ap-southeast-2", session=None)


@patch("strands_tools.memory.get_memory_service_client")
def test_region_from_environment(mock_get_client, environment_configs):
    """Test using region from AWS_REGION environment variable."""
    with patch.dict(os.environ, environment_configs["custom_region"]):
        # Execute without region parameter
        memory.memory(action="list")

        # Verify environment region was used
        mock_get_client.assert_called_once_with(region=None, session=None)


# ===== ERROR HANDLING TESTS =====
def test_invalid_action(environment_configs):
    """Test error handling for invalid action."""
    with patch.dict(os.environ, environment_configs["default"]):
        result = memory.memory(action="invalid")

        assert result["status"] == "error"
        assert "Invalid action" in extract_result_text(result)


def test_missing_knowledge_base_id(environment_configs):
    """Test error when no knowledge base ID is provided."""
    with patch.dict(os.environ, environment_configs["no_kb_id"]):
        result = memory.memory(action="list")

        assert result["status"] == "error"
        assert "No knowledge base ID provided" in extract_result_text(result)


def test_invalid_knowledge_base_id_format(environment_configs):
    """Test error for invalid knowledge base ID format."""
    with patch.dict(os.environ, environment_configs["default"]):
        result = memory.memory(
            action="list",
            STRANDS_KNOWLEDGE_BASE_ID="invalid-kb-id",  # Contains hyphen
        )

        assert result["status"] == "error"
        assert "Invalid knowledge base ID format" in extract_result_text(result)


@patch("strands_tools.memory.get_memory_service_client")
def test_action_specific_missing_params(mock_get_client, environment_configs):
    """Test missing action-specific parameters."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Test missing content for store
        store_result = memory.memory(action="store")
        assert store_result["status"] == "error"
        assert "Content cannot be empty" in extract_result_text(store_result)

        # Test missing document_id for delete
        delete_result = memory.memory(action="delete")
        assert delete_result["status"] == "error"
        assert "Document ID cannot be empty" in extract_result_text(delete_result)

        # Test missing document_id for get
        get_result = memory.memory(action="get")
        assert get_result["status"] == "error"
        assert "Document ID cannot be empty" in extract_result_text(get_result)

        # Test missing query for retrieve
        retrieve_result = memory.memory(action="retrieve")
        assert retrieve_result["status"] == "error"
        assert "No query provided" in extract_result_text(retrieve_result)


# ===== CLIENT AND FORMATTER TESTS =====
@patch("boto3.Session")
def test_memory_service_client_init(mock_session):
    """Test MemoryServiceClient initialization."""
    # Test with default parameters
    client = MemoryServiceClient()
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None

    # Test with custom parameters
    custom_client = MemoryServiceClient(region="us-east-1", profile_name="test-profile")
    assert custom_client.region == "us-east-1"
    assert custom_client.profile_name == "test-profile"
    mock_session.assert_called_with(profile_name="test-profile")


def test_memory_formatter_basic():
    """Test basic MemoryFormatter functionality."""
    formatter = MemoryFormatter()

    # Test format_list_response with empty response
    empty_response = {"documentDetails": []}
    empty_content = formatter.format_list_response(empty_response)
    assert "No documents found" in empty_content[0]["text"]

    # Test format_list_response with documents
    list_response = {
        "documentDetails": [
            {
                "identifier": {"custom": {"id": "doc123"}},
                "status": "INDEXED",
                "updatedAt": "2023-05-09T10:00:00Z",
            }
        ]
    }
    list_content = formatter.format_list_response(list_response)
    assert "Found 1 documents" in list_content[0]["text"]

    # Test format_store_response
    store_content = formatter.format_store_response("doc123", "test123kb", "Test Title")
    assert "Successfully stored content" in store_content[0]["text"]

    # Test format_delete_response
    delete_content = formatter.format_delete_response("DELETED", "doc123", "test123kb")
    assert "Document deletion deleted" in delete_content[0]["text"]

    # Test format_retrieve_response with no results
    empty_retrieve = {"retrievalResults": []}
    retrieve_content = formatter.format_retrieve_response(empty_retrieve, 0.4)
    assert "No results found" in retrieve_content[0]["text"]


# ===== SESSION MANAGEMENT TESTS =====
@patch("strands_tools.memory.get_memory_session")
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_memory_with_session_key(
    mock_get_formatter,
    mock_get_client,
    mock_get_session,
    mock_memory_service_client,
    mock_memory_formatter,
    environment_configs,
):
    """Test memory operations with custom session key."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure basic response
        mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
        mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

        # Execute with custom session key
        result = memory.memory(action="list", session_key="my_custom_session")

        # Verify
        assert result["status"] == "success"
        mock_get_session.assert_called_once_with("my_custom_session")
        mock_get_client.assert_called_once_with(region=None, session=mock_session)


# ===== INTEGRATION TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_full_document_lifecycle(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test complete document lifecycle: store, list, get, retrieve, delete."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        doc_id = "memory_20230509_12345678"

        # 1. Store document
        store_data = api_responses["store"]
        mock_memory_service_client.store_document.return_value = (
            store_data["success"],
            doc_id,
            store_data["doc_title"],
        )
        mock_memory_formatter.format_store_response.return_value = formatted_responses["store"]["success"]

        store_result = memory.memory(action="store", content="Test content", title="Test Title")
        assert store_result["status"] == "success"

        # 2. List documents
        mock_memory_service_client.list_documents.return_value = api_responses["list"]["with_documents"]
        mock_memory_formatter.format_list_response.return_value = formatted_responses["list"]["with_documents"]

        list_result = memory.memory(action="list")
        assert list_result["status"] == "success"

        # 3. Get specific document
        mock_memory_service_client.get_document.return_value = api_responses["get"]["document_found"]
        mock_memory_service_client.retrieve.return_value = api_responses["get"]["document_content"]
        mock_memory_formatter.format_get_response.return_value = formatted_responses["get"]["success"]

        get_result = memory.memory(action="get", document_id=doc_id)
        assert get_result["status"] == "success"

        # 4. Retrieve with query
        mock_memory_service_client.retrieve.return_value = api_responses["retrieve"]["with_results"]
        mock_memory_formatter.format_retrieve_response.return_value = formatted_responses["retrieve"]["success"]

        retrieve_result = memory.memory(action="retrieve", query="test")
        assert retrieve_result["status"] == "success"

        # 5. Delete document
        mock_memory_service_client.delete_document.return_value = api_responses["delete"]["success"]
        mock_memory_formatter.format_delete_response.return_value = formatted_responses["delete"]["success"]

        delete_result = memory.memory(action="delete", document_id=doc_id)
        assert delete_result["status"] == "success"


# ===== ADDITIONAL EDGE CASE TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_store_with_very_large_content(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test storing very large content (over 15000 characters)."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Create large content
        large_content = "A" * 20000  # 20,000 characters

        store_data = api_responses["store"]
        mock_memory_service_client.store_document.return_value = (
            store_data["success"],
            store_data["doc_id"],
            store_data["doc_title"],
        )
        mock_memory_formatter.format_store_response.return_value = formatted_responses["store"]["success"]

        # Execute
        result = memory.memory(action="store", content=large_content, title="Large Document")

        # Verify
        assert result["status"] == "success"
        mock_memory_service_client.store_document.assert_called_once()
        call_args = mock_memory_service_client.store_document.call_args[0]
        assert call_args[2] == large_content  # Full content should be stored


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_store_with_special_characters(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    api_responses,
    formatted_responses,
    environment_configs,
):
    """Test storing content with special characters and unicode."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Content with special characters
        special_content = "Special chars: @#$%^&*() Unicode: 🎉 测试 тест\n\tTabs and\nnewlines"
        special_title = "Title with émojis 🚀 and spëcial çhars"

        store_data = api_responses["store"]
        mock_memory_service_client.store_document.return_value = (
            store_data["success"],
            store_data["doc_id"],
            special_title,
        )
        mock_memory_formatter.format_store_response.return_value = formatted_responses["store"]["success"]

        # Execute
        result = memory.memory(action="store", content=special_content, title=special_title)

        # Verify
        assert result["status"] == "success"
        mock_memory_service_client.store_document.assert_called_once_with(
            "test123kb", "ds123", special_content, special_title
        )


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_list_with_empty_pagination_token(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    environment_configs,
):
    """Test list operation with empty string as next_token."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
        mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

        # Execute with empty string token (should be treated as None)
        result = memory.memory(action="list", next_token="")

        # Verify - empty string should not be passed to API
        assert result["status"] == "success"
        mock_memory_service_client.list_documents.assert_called_once_with("test123kb", "ds123", 50, "")


@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_with_complex_query(
    mock_get_client,
    mock_memory_service_client,
    environment_configs,
):
    """Test retrieve with complex query containing special characters."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_formatter = MagicMock()

        complex_query = 'complex query with "quotes" and special chars: @#$%'

        mock_memory_service_client.retrieve.return_value = {"retrievalResults": []}
        mock_formatter.format_retrieve_response.return_value = [{"text": "No results found."}]

        with patch("strands_tools.memory.get_memory_formatter", return_value=mock_formatter):
            # Execute
            result = memory.memory(action="retrieve", query=complex_query)

            # Verify
            assert result["status"] == "success"
            mock_memory_service_client.retrieve.assert_called_once()
            call_args = mock_memory_service_client.retrieve.call_args[1]
            assert call_args["query"] == complex_query


@patch("strands_tools.memory.get_memory_service_client")
def test_api_throttling_error(
    mock_get_client,
    environment_configs,
):
    """Test handling of AWS throttling errors."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Simulate throttling error
        mock_client.get_data_source_id.side_effect = Exception("ThrottlingException: Rate exceeded")

        # Execute
        result = memory.memory(action="list")

        # Verify
        assert result["status"] == "error"
        assert "ThrottlingException" in extract_result_text(result)


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_exact_min_score_boundary(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    environment_configs,
):
    """Test retrieve with min_score at exact boundaries (0.0 and 1.0)."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        mock_memory_service_client.retrieve.return_value = {"retrievalResults": []}
        mock_memory_formatter.format_retrieve_response.return_value = [{"text": "No results found."}]

        # Test with min_score = 0.0
        result = memory.memory(action="retrieve", query="test", min_score=0.0)
        assert result["status"] == "success"

        # Test with min_score = 1.0
        result = memory.memory(action="retrieve", query="test", min_score=1.0)
        assert result["status"] == "success"


@patch("strands_tools.memory.get_memory_service_client")
def test_get_data_source_id_caching(
    mock_get_client,
    environment_configs,
):
    """Test that data source ID is properly retrieved and used."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_data_source_id.return_value = "cached_ds_id"

        # First call
        memory.memory(action="list")

        # Verify get_data_source_id was called
        mock_client.get_data_source_id.assert_called_once_with("test123kb", None)

        # The list_documents should use the returned data source ID
        mock_client.list_documents.assert_called_once_with("test123kb", "cached_ds_id", 50, None)


def test_extract_result_text_with_various_structures():
    """Test the extract_result_text helper function with different result structures."""
    # Test with proper structure
    result1 = {"content": [{"text": "Success message"}]}
    assert extract_result_text(result1) == "Success message"

    # Test with string result
    result2 = "Direct string result"
    assert extract_result_text(result2) == "Direct string result"

    # Test with dict without content
    result3 = {"status": "error", "message": "Error occurred"}
    assert extract_result_text(result3) == str(result3)

    # Test with empty content list
    result4 = {"content": []}
    assert extract_result_text(result4) == "{'content': []}"

    # Test with None
    result5 = None
    assert extract_result_text(result5) == "None"


# ===== ENVIRONMENT VARIABLE TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_default_environment_variables(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test using default environment variables for max_results and min_score."""
    with patch.dict(
        os.environ,
        {
            "STRANDS_KNOWLEDGE_BASE_ID": "test123kb",
            "MEMORY_DEFAULT_MAX_RESULTS": "100",
            "MEMORY_DEFAULT_MIN_SCORE": "0.8",
        },
    ):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        mock_memory_service_client.retrieve.return_value = {"retrievalResults": []}
        mock_memory_formatter.format_retrieve_response.return_value = [{"text": "No results found."}]

        # Execute without specifying max_results or min_score
        result = memory.memory(action="retrieve", query="test")

        # Verify defaults from environment were used
        assert result["status"] == "success"
        mock_memory_service_client.retrieve.assert_called_once_with(
            kb_id="test123kb", query="test", max_results=100, next_token=None
        )
        mock_memory_formatter.format_retrieve_response.assert_called_once_with({"retrievalResults": []}, 0.8)


@patch("strands_tools.memory.get_memory_service_client")
def test_missing_aws_region_defaults(
    mock_get_client,
):
    """Test that missing AWS_REGION defaults to us-west-2."""
    with patch.dict(
        os.environ,
        {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"},
        clear=True,  # Clear all env vars
    ):
        # Execute
        memory.memory(action="list")

        # Verify default region was used
        mock_get_client.assert_called_once_with(region=None, session=None)


# ===== SESSION MANAGEMENT EDGE CASES =====
def test_set_and_get_memory_session():
    """Test the session store functionality directly."""
    from strands_tools.memory import (
        _SESSION_STORE,
        get_memory_session,
        set_memory_session,
    )

    # Clear any existing sessions
    _SESSION_STORE.clear()

    # Test setting and getting a session
    mock_session = MagicMock()
    set_memory_session(mock_session, "test_key")

    retrieved_session = get_memory_session("test_key")
    assert retrieved_session == mock_session

    # Test getting non-existent session
    assert get_memory_session("non_existent") is None

    # Test clearing a session
    set_memory_session(None, "test_key")
    assert get_memory_session("test_key") is None


@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_memory_with_multiple_sessions(
    mock_get_formatter,
    mock_get_client,
    environment_configs,
):
    """Test using multiple stored sessions with different keys."""
    from strands_tools.memory import set_memory_session

    with patch.dict(os.environ, environment_configs["default"]):
        # Setup multiple sessions
        session1 = MagicMock()
        session2 = MagicMock()
        set_memory_session(session1, "session1")
        set_memory_session(session2, "session2")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_formatter = MagicMock()
        mock_get_formatter.return_value = mock_formatter

        mock_client.list_documents.return_value = {"documentDetails": []}
        mock_formatter.format_list_response.return_value = [{"text": "No documents found."}]

        # Use first session
        result1 = memory.memory(action="list", session_key="session1")
        assert result1["status"] == "success"
        assert mock_get_client.call_args_list[0][1]["session"] == session1

        # Use second session
        result2 = memory.memory(action="list", session_key="session2")
        assert result2["status"] == "success"
        assert mock_get_client.call_args_list[1][1]["session"] == session2


# ===== UNICODE AND ENCODING TESTS =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_unicode_content_handling(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
    environment_configs,
):
    """Test handling of various unicode characters in content."""
    with patch.dict(os.environ, environment_configs["bypass_consent"]):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Various unicode test cases
        unicode_content = """
        English: Hello World
        Chinese: 你好世界
        Japanese: こんにちは世界
        Korean: 안녕하세요 세계
        Arabic: مرحبا بالعالم
        Hebrew: שלום עולם
        Russian: Привет мир
        Emoji: 🌍🌎🌏 🚀 ⭐ 🎉
        Math: ∑ ∏ ∫ √ ∞ ≈ ≠ ≤ ≥
        Symbols: © ® ™ € £ ¥ § ¶ † ‡
        """

        mock_memory_service_client.store_document.return_value = (
            {"status": "success"},
            "doc123",
            "Unicode Test",
        )
        mock_memory_formatter.format_store_response.return_value = [{"text": "✅ Successfully stored content"}]

        # Execute
        result = memory.memory(action="store", content=unicode_content, title="Unicode Test")

        # Verify
        assert result["status"] == "success"
        # Verify the unicode content was passed correctly
        call_args = mock_memory_service_client.store_document.call_args[0]
        assert unicode_content in call_args[2]


# ===== FINAL CLEANUP TEST =====
def test_cleanup_session_store():
    """Test cleanup of session store to avoid memory leaks."""
    from strands_tools.memory import _SESSION_STORE, set_memory_session

    # Add some test sessions
    for i in range(5):
        set_memory_session(MagicMock(), f"test_session_{i}")

    # Verify sessions were added
    assert len(_SESSION_STORE) >= 5

    # Clean up all test sessions
    for i in range(5):
        set_memory_session(None, f"test_session_{i}")

    # Verify cleanup
    for i in range(5):
        assert f"test_session_{i}" not in _SESSION_STORE


# ===== ADDITIONAL TESTS FOR DATA SOURCE ERROR MESSAGES =====
@patch("strands_tools.memory.get_memory_service_client")
def test_data_source_error_message_format(
    mock_get_client,
    environment_configs,
):
    """Test that data source error messages match expected format."""
    with patch.dict(os.environ, environment_configs["default"]):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return error response with correct punctuation
        mock_client.get_data_source_id.return_value = {
            "status": "error",
            "content": [{"text": "❌ Provided DataSource ID: ds456 is not of type 'CUSTOM'"}],
        }

        # Execute
        result = memory.memory(action="list", STRANDS_KNOWLEDGE_BASE_MEMORY_DATASOURCE_ID="ds456")

        # Verify - the error message includes a period
        assert result["status"] == "error"
        assert "not of type 'CUSTOM'" in result["content"][0]["text"]


# ===== ADDITIONAL TESTS FOR CUSTOM DATASOURCE ID =====
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_memory_with_valid_custom_datasource_id(
    mock_get_formatter,
    mock_get_client,
    mock_memory_service_client,
    mock_memory_formatter,
):
    """Test memory operations with valid custom datasource ID parameter."""
    with patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"}):
        # Setup mocks
        mock_get_client.return_value = mock_memory_service_client
        mock_get_formatter.return_value = mock_memory_formatter

        # Configure mocks
        mock_memory_service_client.list_documents.return_value = {"documentDetails": []}
        mock_memory_formatter.format_list_response.return_value = [{"text": "No documents found."}]

        # Execute with valid datasource ID (alphanumeric only)
        result = memory.memory(action="list", STRANDS_KNOWLEDGE_BASE_MEMORY_DATASOURCE_ID="customds789")

        # Verify
        assert result["status"] == "success"

        # Should validate and use the custom datasource ID
        mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb", "customds789")
