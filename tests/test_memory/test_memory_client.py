"""
Tests for the MemoryServiceClient class in memory.py.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands_tools.memory import MemoryServiceClient


# ===== FIXTURES FOR AWS RESPONSES =====
@pytest.fixture
def aws_responses():
    """Comprehensive AWS response fixtures for different scenarios."""
    return {
        "list_data_sources": {
            "custom_only": {"dataSourceSummaries": [{"dataSourceId": "ds123"}]},
            "mixed_types": {
                "dataSourceSummaries": [
                    {"dataSourceId": "ds123"},  # Will be CUSTOM
                    {"dataSourceId": "ds456"},  # Will be S3
                    {"dataSourceId": "ds789"},  # Will be CUSTOM
                ]
            },
            "no_custom": {
                "dataSourceSummaries": [
                    {"dataSourceId": "ds111"},  # Will be S3
                    {"dataSourceId": "ds222"},  # Will be WEB_CRAWLER
                ]
            },
            "empty": {"dataSourceSummaries": []},
            "with_pagination": {
                "first_page": {
                    "dataSourceSummaries": [
                        {"dataSourceId": "ds111"},
                        {"dataSourceId": "ds222"},
                    ],
                    "nextToken": "token123",
                },
                "second_page": {
                    "dataSourceSummaries": [
                        {"dataSourceId": "ds333"}  # This will be CUSTOM
                    ]
                },
            },
        },
        "get_data_source": {
            "custom": {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}},
            "s3": {"dataSource": {"dataSourceConfiguration": {"type": "S3"}}},
            "web_crawler": {"dataSource": {"dataSourceConfiguration": {"type": "WEB_CRAWLER"}}},
        },
        "get_document": {
            "indexed": {
                "documentDetails": [
                    {
                        "identifier": {"custom": {"id": "doc123"}},
                        "status": "INDEXED",
                        "updatedAt": "2023-05-09T10:00:00Z",
                    }
                ]
            },
            "processing": {
                "documentDetails": [
                    {
                        "identifier": {"custom": {"id": "doc123"}},
                        "status": "PROCESSING",
                        "updatedAt": "2023-05-09T10:00:00Z",
                    }
                ]
            },
            "not_found": {"documentDetails": []},
        },
    }


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
def mock_boto3_session():
    """Create a mock boto3 session with properly configured clients."""
    session = MagicMock()

    # Create separate mocks for agent and runtime clients
    agent_client = MagicMock()
    runtime_client = MagicMock()

    # Configure session.client to return appropriate client based on service name
    def client_factory(service_name, **kwargs):
        if service_name == "bedrock-agent":
            return agent_client
        elif service_name == "bedrock-agent-runtime":
            return runtime_client
        else:
            raise ValueError(f"Unexpected service: {service_name}")

    session.client.side_effect = client_factory

    # Attach clients to session for easy access in tests
    session.agent_client = agent_client
    session.runtime_client = runtime_client

    return session


# ===== CLIENT INITIALIZATION TESTS =====
@patch("boto3.Session")
def test_client_init_default(mock_session_class):
    """Test client initialization with default parameters."""
    session_instance = MagicMock()
    mock_session_class.return_value = session_instance

    client = MemoryServiceClient()

    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None
    mock_session_class.assert_called_once_with()


@patch("boto3.Session")
def test_client_init_custom_region(mock_session_class):
    """Test client initialization with custom region."""
    session_instance = MagicMock()
    mock_session_class.return_value = session_instance

    client = MemoryServiceClient(region="us-east-1")

    assert client.region == "us-east-1"
    assert client.profile_name is None
    mock_session_class.assert_called_once_with()


@patch("boto3.Session")
def test_client_init_custom_profile(mock_session_class):
    """Test client initialization with custom profile."""
    session_instance = MagicMock()
    mock_session_class.return_value = session_instance

    client = MemoryServiceClient(profile_name="test-profile")

    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name == "test-profile"
    mock_session_class.assert_called_once_with(profile_name="test-profile")


def test_client_init_with_session():
    """Test client initialization with pre-configured session."""
    mock_session = MagicMock()

    client = MemoryServiceClient(session=mock_session)

    assert client.session == mock_session


# ===== PROPERTY TESTS =====
def test_agent_client_property(mock_boto3_session):
    """Test the agent_client property lazy loading."""
    client = MemoryServiceClient(session=mock_boto3_session)

    # Access the property
    result = client.agent_client

    # Verify client was created
    mock_boto3_session.client.assert_called_once_with("bedrock-agent", region_name=client.region)
    assert result == mock_boto3_session.agent_client

    # Verify same client is returned on second access (cached)
    mock_boto3_session.client.reset_mock()
    result2 = client.agent_client
    assert result is result2
    mock_boto3_session.client.assert_not_called()


def test_runtime_client_property(mock_boto3_session):
    """Test the runtime_client property lazy loading."""
    client = MemoryServiceClient(session=mock_boto3_session)

    # Access the property
    result = client.runtime_client

    # Verify client was created
    mock_boto3_session.client.assert_called_once_with("bedrock-agent-runtime", region_name=client.region)
    assert result == mock_boto3_session.runtime_client

    # Verify same client is returned on second access (cached)
    mock_boto3_session.client.reset_mock()
    result2 = client.runtime_client
    assert result is result2
    mock_boto3_session.client.assert_not_called()


# ===== FIND CUSTOM DATA SOURCE TESTS =====
def test_find_custom_data_source_id_success(mock_boto3_session, aws_responses):
    """Test find_custom_data_source_id method finding CUSTOM type data source."""
    # Configure mocks
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["mixed_types"]

    # Set up get_data_source responses for mixed types
    mock_boto3_session.agent_client.get_data_source.side_effect = [
        aws_responses["get_data_source"]["custom"],  # ds123 is CUSTOM
        # Stops here since we found CUSTOM
    ]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.find_custom_data_source_id("test123kb")

    assert result == "ds123"
    mock_boto3_session.agent_client.list_data_sources.assert_called_once_with(knowledgeBaseId="test123kb")
    assert mock_boto3_session.agent_client.get_data_source.call_count == 1


def test_find_custom_data_source_id_third_is_custom(mock_boto3_session, aws_responses):
    """Test find_custom_data_source_id when third data source is CUSTOM."""
    # Configure mocks
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["mixed_types"]

    # Set up responses where third one is CUSTOM
    mock_boto3_session.agent_client.get_data_source.side_effect = [
        aws_responses["get_data_source"]["s3"],  # ds123 is S3
        aws_responses["get_data_source"]["web_crawler"],  # ds456 is WEB_CRAWLER
        aws_responses["get_data_source"]["custom"],  # ds789 is CUSTOM
    ]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.find_custom_data_source_id("test123kb")

    assert result == "ds789"
    assert mock_boto3_session.agent_client.get_data_source.call_count == 3


def test_find_custom_data_source_id_with_pagination(mock_boto3_session, aws_responses):
    """Test find_custom_data_source_id method with pagination."""
    # Configure paginated responses
    mock_boto3_session.agent_client.list_data_sources.side_effect = [
        aws_responses["list_data_sources"]["with_pagination"]["first_page"],
        aws_responses["list_data_sources"]["with_pagination"]["second_page"],
    ]

    # All on first page are not CUSTOM, third one (on second page) is CUSTOM
    mock_boto3_session.agent_client.get_data_source.side_effect = [
        aws_responses["get_data_source"]["s3"],  # ds111
        aws_responses["get_data_source"]["web_crawler"],  # ds222
        aws_responses["get_data_source"]["custom"],  # ds333
    ]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.find_custom_data_source_id("test456kb")

    assert result == "ds333"

    # Verify pagination calls
    calls = mock_boto3_session.agent_client.list_data_sources.call_args_list
    assert len(calls) == 2
    assert calls[0][1] == {"knowledgeBaseId": "test456kb"}
    assert calls[1][1] == {"knowledgeBaseId": "test456kb", "nextToken": "token123"}


def test_find_custom_data_source_id_no_custom_found(mock_boto3_session, aws_responses):
    """Test find_custom_data_source_id when no CUSTOM data source exists."""
    # Configure mocks with no CUSTOM type
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["no_custom"]

    mock_boto3_session.agent_client.get_data_source.side_effect = [
        aws_responses["get_data_source"]["s3"],
        aws_responses["get_data_source"]["web_crawler"],
    ]

    client = MemoryServiceClient(session=mock_boto3_session)

    with pytest.raises(ValueError, match=r"No CUSTOM data source found"):
        client.find_custom_data_source_id("test123kb")


def test_find_custom_data_source_id_no_sources(mock_boto3_session, aws_responses):
    """Test find_custom_data_source_id when no data sources exist."""
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["empty"]

    client = MemoryServiceClient(session=mock_boto3_session)

    with pytest.raises(ValueError, match=r"No data sources found"):
        client.find_custom_data_source_id("test123kb")


# ===== GET DATA SOURCE ID TESTS =====
def test_get_data_source_id_with_valid_custom_ds_id(mock_boto3_session, aws_responses):
    """Test get_data_source_id with provided ds_id that is CUSTOM type."""
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.get_data_source_id("test123kb", "ds456")

    assert result == "ds456"
    mock_boto3_session.agent_client.get_data_source.assert_called_once_with(
        dataSourceId="ds456", knowledgeBaseId="test123kb"
    )


def test_get_data_source_id_with_non_custom_ds_id(mock_boto3_session, aws_responses):
    """Test get_data_source_id with provided ds_id that is not CUSTOM type."""
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["s3"]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.get_data_source_id("test123kb", "ds456")

    assert result["status"] == "error"
    assert "not of type 'CUSTOM'" in result["content"][0]["text"]


def test_get_data_source_id_without_ds_id(mock_boto3_session, aws_responses):
    """Test get_data_source_id without ds_id parameter, should call find_custom_data_source_id."""
    # Configure for find_custom_data_source_id flow
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.get_data_source_id("test123kb")

    assert result == "ds123"
    mock_boto3_session.agent_client.list_data_sources.assert_called_once()


def test_get_data_source_id_api_error(mock_boto3_session):
    """Test get_data_source_id when API call fails."""
    mock_boto3_session.agent_client.get_data_source.side_effect = Exception("API Error")

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.get_data_source_id("test123kb", "ds456")

    assert result["status"] == "error"
    assert "API Error" in result["content"][0]["text"]


# ===== DOCUMENT OPERATION TESTS =====
def test_list_documents_with_defaults(mock_boto3_session, aws_responses):
    """Test list_documents method with default parameters."""
    # Configure find_custom_data_source_id flow
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]

    client = MemoryServiceClient(session=mock_boto3_session)
    client.list_documents("test123kb")

    mock_boto3_session.agent_client.list_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="test123kb", dataSourceId="ds123"
    )


def test_list_documents_with_params(mock_boto3_session):
    """Test list_documents method with all parameters."""
    client = MemoryServiceClient(session=mock_boto3_session)
    client.list_documents("test123kb", "ds456", 10, "token123")

    mock_boto3_session.agent_client.list_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="test123kb",
        dataSourceId="ds456",
        maxResults=10,
        nextToken="token123",
    )


def test_get_document(mock_boto3_session, aws_responses):
    """Test get_document method."""
    # Configure find_custom_data_source_id flow
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]

    client = MemoryServiceClient(session=mock_boto3_session)
    client.get_document("test123kb", None, "doc123")

    mock_boto3_session.agent_client.get_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="test123kb",
        dataSourceId="ds123",
        documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
    )


def test_store_document(mock_boto3_session, aws_responses):
    """Test store_document method."""
    # Configure find_custom_data_source_id flow
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]
    mock_boto3_session.agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    client = MemoryServiceClient(session=mock_boto3_session)
    response, doc_id, doc_title = client.store_document("test123kb", None, "test content", "Test Title")

    assert response == {"status": "success"}
    assert "memory_" in doc_id
    assert doc_title == "Test Title"

    # Verify API call
    call_args = mock_boto3_session.agent_client.ingest_knowledge_base_documents.call_args[1]
    assert call_args["knowledgeBaseId"] == "test123kb"
    assert call_args["dataSourceId"] == "ds123"
    assert len(call_args["documents"]) == 1

    # Verify document content structure
    doc = call_args["documents"][0]
    content_json = doc["content"]["custom"]["inlineContent"]["textContent"]["data"]
    content_data = json.loads(content_json)
    assert content_data["title"] == "Test Title"
    assert content_data["action"] == "store"
    assert content_data["content"] == "test content"


def test_store_document_no_title(mock_boto3_session, aws_responses):
    """Test store_document method with auto-generated title."""
    # Configure mocks
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]
    mock_boto3_session.agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    client = MemoryServiceClient(session=mock_boto3_session)
    response, doc_id, doc_title = client.store_document("test123kb", None, "test content")

    assert "Strands Memory" in doc_title

    # Verify the title was used in the document
    call_args = mock_boto3_session.agent_client.ingest_knowledge_base_documents.call_args[1]
    doc = call_args["documents"][0]
    content_json = doc["content"]["custom"]["inlineContent"]["textContent"]["data"]
    content_data = json.loads(content_json)
    assert content_data["title"] == doc_title


def test_delete_document(mock_boto3_session, aws_responses):
    """Test delete_document method."""
    # Configure mocks
    mock_boto3_session.agent_client.list_data_sources.return_value = aws_responses["list_data_sources"]["custom_only"]
    mock_boto3_session.agent_client.get_data_source.return_value = aws_responses["get_data_source"]["custom"]
    mock_boto3_session.agent_client.delete_knowledge_base_documents.return_value = {"status": "success"}

    client = MemoryServiceClient(session=mock_boto3_session)
    response = client.delete_document("test123kb", None, "doc123")

    assert response == {"status": "success"}
    mock_boto3_session.agent_client.delete_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="test123kb",
        dataSourceId="ds123",
        documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
    )


def test_retrieve(mock_boto3_session):
    """Test retrieve method."""
    mock_boto3_session.runtime_client.retrieve.return_value = {"retrievalResults": []}

    client = MemoryServiceClient(session=mock_boto3_session)
    result = client.retrieve("test123kb", "test query", 10)

    assert result == {"retrievalResults": []}
    mock_boto3_session.runtime_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="test123kb",
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 10},
        },
    )


def test_retrieve_with_token(mock_boto3_session):
    """Test retrieve method with pagination token."""
    client = MemoryServiceClient(session=mock_boto3_session)
    client.retrieve("test123kb", "test query", 10, "token123")

    call_args = mock_boto3_session.runtime_client.retrieve.call_args[1]
    assert call_args["nextToken"] == "token123"
