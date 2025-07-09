"""
Tests for the MemoryFormatter class in memory.py.
"""

import json

import pytest
from strands_tools.memory import MemoryFormatter


# ===== FIXTURES FOR FORMATTER TESTS =====
@pytest.fixture
def formatter():
    """Create a MemoryFormatter instance."""
    return MemoryFormatter()


@pytest.fixture
def list_responses():
    """Fixture for various list response scenarios."""
    return {
        "empty": {"documentDetails": []},
        "with_custom_ids": {
            "documentDetails": [
                {
                    "identifier": {"custom": {"id": "doc123"}},
                    "status": "INDEXED",
                    "updatedAt": "2023-05-09T10:00:00Z",
                },
                {
                    "identifier": {"custom": {"id": "doc456"}},
                    "status": "INGESTING",
                    "updatedAt": "2023-05-09T11:00:00Z",
                },
            ]
        },
        "with_s3_ids": {
            "documentDetails": [
                {
                    "identifier": {"s3": {"uri": "s3://bucket/file1.txt"}},
                    "status": "INDEXED",
                    "updatedAt": "2023-05-09T10:00:00Z",
                },
                {
                    "identifier": {"s3": {"uri": "s3://bucket/file2.txt"}},
                    "status": "PROCESSING",
                    "updatedAt": "2023-05-09T11:00:00Z",
                },
            ]
        },
        "with_mixed_ids": {
            "documentDetails": [
                {
                    "identifier": {"custom": {"id": "doc123"}},
                    "status": "INDEXED",
                    "updatedAt": "2023-05-09T10:00:00Z",
                },
                {
                    "identifier": {"s3": {"uri": "s3://bucket/file.txt"}},
                    "status": "INDEXED",
                    "updatedAt": "2023-05-09T11:00:00Z",
                },
                {
                    "identifier": {},  # Missing identifier details
                    "status": "FAILED",
                    "updatedAt": "2023-05-09T12:00:00Z",
                },
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
            "nextToken": "token123",
        },
    }


@pytest.fixture
def retrieve_responses():
    """Fixture for various retrieve response scenarios."""
    return {
        "empty": {"retrievalResults": []},
        "single_result": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc123"}},
                    "content": {"text": "This is test content"},
                }
            ]
        },
        "multiple_results": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc123"}},
                    "content": {"text": "First document content"},
                },
                {
                    "score": 0.85,
                    "location": {"customDocumentLocation": {"id": "doc456"}},
                    "content": {"text": "Second document content"},
                },
                {
                    "score": 0.75,
                    "location": {"customDocumentLocation": {"id": "doc789"}},
                    "content": {"text": "Third document content"},
                },
            ]
        },
        "with_json_content": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc123"}},
                    "content": {
                        "text": json.dumps(
                            {
                                "title": "Test Document",
                                "content": "This is structured content",
                                "metadata": {
                                    "author": "Test Author",
                                    "date": "2023-05-09",
                                },
                            }
                        )
                    },
                }
            ]
        },
        "mixed_scores": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc_high"}},
                    "content": {"text": "High score content"},
                },
                {
                    "score": 0.45,
                    "location": {"customDocumentLocation": {"id": "doc_medium"}},
                    "content": {"text": "Medium score content"},
                },
                {
                    "score": 0.15,
                    "location": {"customDocumentLocation": {"id": "doc_low"}},
                    "content": {"text": "Low score content"},
                },
            ]
        },
        "with_pagination": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc123"}},
                    "content": {"text": "This is test content"},
                }
            ],
            "nextToken": "retrieve_token_456",
        },
        "long_content": {
            "retrievalResults": [
                {
                    "score": 0.95,
                    "location": {"customDocumentLocation": {"id": "doc123"}},
                    "content": {"text": "A" * 200},  # Long content for preview testing
                }
            ]
        },
    }


@pytest.fixture
def document_metadata():
    """Fixture for document metadata scenarios."""
    return {
        "basic": {"document_id": "doc123", "kb_id": "kb456", "title": "Test Document"},
        "with_special_chars": {
            "document_id": "doc_special_123",
            "kb_id": "kb_special_456",
            "title": "Document with Special Characters: @#$%",
        },
        "with_unicode": {
            "document_id": "doc_unicode_123",
            "kb_id": "kb_unicode_456",
            "title": "Document with Unicode: 🎉 测试 тест",
        },
    }


# ===== LIST RESPONSE FORMATTING TESTS =====
def test_format_list_response_empty(formatter, list_responses):
    """Test formatting an empty list response."""
    result = formatter.format_list_response(list_responses["empty"])

    assert len(result) == 1
    assert result[0]["text"] == "No documents found."


def test_format_list_response_with_custom_ids(formatter, list_responses):
    """Test formatting a list response with custom document IDs."""
    result = formatter.format_list_response(list_responses["with_custom_ids"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Found 2 documents:" in text
    assert "doc123" in text
    assert "doc456" in text
    assert "INDEXED" in text
    assert "INGESTING" in text
    assert "2023-05-09T10:00:00Z" in text
    assert "2023-05-09T11:00:00Z" in text


def test_format_list_response_with_s3_ids(formatter, list_responses):
    """Test formatting a list response with S3 identifiers."""
    result = formatter.format_list_response(list_responses["with_s3_ids"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Found 2 documents:" in text
    assert "s3://bucket/file1.txt" in text
    assert "s3://bucket/file2.txt" in text
    assert "PROCESSING" in text


def test_format_list_response_with_mixed_ids(formatter, list_responses):
    """Test formatting a list response with mixed identifier types."""
    result = formatter.format_list_response(list_responses["with_mixed_ids"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Found 3 documents:" in text
    assert "doc123" in text
    assert "s3://bucket/file.txt" in text
    assert "FAILED" in text
    # The third document with missing identifier should be skipped in details


def test_format_list_response_with_pagination(formatter, list_responses):
    """Test formatting a list response with pagination token."""
    result = formatter.format_list_response(list_responses["with_pagination"])

    assert len(result) == 3
    assert "Found 1 documents:" in result[0]["text"]
    assert "More results available" in result[1]["text"]
    assert "token123" in result[2]["text"]


# ===== GET RESPONSE FORMATTING TESTS =====
def test_format_get_response_basic(formatter, document_metadata):
    """Test formatting a basic get document response."""
    metadata = document_metadata["basic"]
    content_data = {
        "title": metadata["title"],
        "content": "This is the document content",
    }

    result = formatter.format_get_response(metadata["document_id"], metadata["kb_id"], content_data)

    assert len(result) == 5
    assert "Document retrieved successfully" in result[0]["text"]
    assert metadata["title"] in result[1]["text"]
    assert metadata["document_id"] in result[2]["text"]
    assert metadata["kb_id"] in result[3]["text"]
    assert "This is the document content" in result[4]["text"]


def test_format_get_response_missing_fields(formatter, document_metadata):
    """Test formatting get response with missing content fields."""
    metadata = document_metadata["basic"]
    content_data = {
        # Missing title
        "content": "Content without title"
    }

    result = formatter.format_get_response(metadata["document_id"], metadata["kb_id"], content_data)

    assert "Unknown" in result[1]["text"]  # Default title
    assert "Content without title" in result[4]["text"]


def test_format_get_response_with_special_chars(formatter, document_metadata):
    """Test formatting get response with special characters."""
    metadata = document_metadata["with_special_chars"]
    content_data = {
        "title": metadata["title"],
        "content": "Content with special chars: <>&\"'",
    }

    result = formatter.format_get_response(metadata["document_id"], metadata["kb_id"], content_data)

    assert metadata["title"] in result[1]["text"]
    assert "Content with special chars: <>&\"'" in result[4]["text"]


# ===== STORE RESPONSE FORMATTING TESTS =====
def test_format_store_response_basic(formatter, document_metadata):
    """Test formatting a basic store document response."""
    metadata = document_metadata["basic"]

    result = formatter.format_store_response(metadata["document_id"], metadata["kb_id"], metadata["title"])

    assert len(result) == 4
    assert "Successfully stored content" in result[0]["text"]
    assert metadata["title"] in result[1]["text"]
    assert metadata["document_id"] in result[2]["text"]
    assert metadata["kb_id"] in result[3]["text"]


def test_format_store_response_with_unicode(formatter, document_metadata):
    """Test formatting store response with unicode characters."""
    metadata = document_metadata["with_unicode"]

    result = formatter.format_store_response(metadata["document_id"], metadata["kb_id"], metadata["title"])

    assert metadata["title"] in result[1]["text"]
    assert "🎉" in result[1]["text"]
    assert "测试" in result[1]["text"]
    assert "тест" in result[1]["text"]


# ===== DELETE RESPONSE FORMATTING TESTS =====
@pytest.mark.parametrize(
    "status,expected_text",
    [
        ("DELETED", "Document deletion deleted"),
        ("DELETING", "Document deletion deleting"),
        ("DELETE_IN_PROGRESS", "Document deletion delete in progress"),
        ("FAILED", "Document deletion failed"),
        ("ERROR", "Document deletion failed"),
        ("UNKNOWN", "Document deletion failed"),
    ],
)
def test_format_delete_response_various_statuses(formatter, status, expected_text):
    """Test formatting delete response with various status values."""
    result = formatter.format_delete_response(status, "doc123", "kb456")

    assert expected_text in result[0]["text"]
    assert "doc123" in result[1]["text"]
    assert "kb456" in result[2]["text"]


# ===== RETRIEVE RESPONSE FORMATTING TESTS =====
def test_format_retrieve_response_empty(formatter, retrieve_responses):
    """Test formatting an empty retrieve response."""
    result = formatter.format_retrieve_response(retrieve_responses["empty"])

    assert len(result) == 1
    assert result[0]["text"] == "No results found that meet the score threshold."


def test_format_retrieve_response_single_result(formatter, retrieve_responses):
    """Test formatting a retrieve response with single result."""
    result = formatter.format_retrieve_response(retrieve_responses["single_result"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Retrieved 1 results" in text
    assert "0.9500" in text
    assert "doc123" in text
    assert "This is test content" in text


def test_format_retrieve_response_multiple_results(formatter, retrieve_responses):
    """Test formatting a retrieve response with multiple results."""
    result = formatter.format_retrieve_response(retrieve_responses["multiple_results"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Retrieved 3 results" in text
    assert "doc123" in text
    assert "doc456" in text
    assert "doc789" in text
    assert "First document content" in text
    assert "Second document content" in text
    assert "Third document content" in text


def test_format_retrieve_response_with_json_content(formatter, retrieve_responses):
    """Test formatting a retrieve response with JSON content."""
    result = formatter.format_retrieve_response(retrieve_responses["with_json_content"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Test Document" in text  # Title should be extracted
    assert "This is structured content" in text


def test_format_retrieve_response_with_score_filter(formatter, retrieve_responses):
    """Test formatting a retrieve response with score filtering."""
    # Filter out results below 0.5
    result = formatter.format_retrieve_response(retrieve_responses["mixed_scores"], 0.5)

    assert len(result) == 1
    text = result[0]["text"]
    assert "Retrieved 1 results with score >= 0.5" in text
    assert "doc_high" in text
    assert "doc_medium" not in text  # Score 0.45, below threshold
    assert "doc_low" not in text  # Score 0.15, below threshold


def test_format_retrieve_response_with_high_threshold(formatter, retrieve_responses):
    """Test formatting retrieve response with very high score threshold."""
    # Filter with threshold of 0.99 (no results should match)
    result = formatter.format_retrieve_response(retrieve_responses["mixed_scores"], 0.99)

    assert len(result) == 1
    assert "No results found that meet the score threshold" in result[0]["text"]


def test_format_retrieve_response_with_pagination(formatter, retrieve_responses):
    """Test formatting a retrieve response with pagination."""
    result = formatter.format_retrieve_response(retrieve_responses["with_pagination"])

    assert len(result) == 3
    assert "Retrieved 1 results" in result[0]["text"]
    assert "More results available" in result[1]["text"]
    assert "retrieve_token_456" in result[2]["text"]


def test_format_retrieve_response_content_preview(formatter, retrieve_responses):
    """Test formatting retrieve response with long content (preview truncation)."""
    result = formatter.format_retrieve_response(retrieve_responses["long_content"])

    assert len(result) == 1
    text = result[0]["text"]
    assert "Retrieved 1 results" in text
    assert "A" * 150 in text  # First 150 characters
    assert "..." in text  # Truncation indicator
    assert len(text) < 1000  # Reasonable length limit


def test_format_retrieve_response_invalid_json_content(formatter):
    """Test formatting retrieve response with invalid JSON content."""
    response = {
        "retrievalResults": [
            {
                "score": 0.95,
                "location": {"customDocumentLocation": {"id": "doc123"}},
                "content": {"text": "{invalid json}"},  # Invalid JSON
            }
        ]
    }

    result = formatter.format_retrieve_response(response)

    assert len(result) == 1
    text = result[0]["text"]
    assert "doc123" in text
    assert "{invalid json}" in text  # Should include raw content


# ===== EDGE CASE TESTS =====
def test_format_responses_with_none_values(formatter):
    """Test formatter methods handle None values gracefully."""
    # Test with None/missing values in various places

    # List response with None values
    list_response = {
        "documentDetails": [
            {
                "identifier": {"custom": {"id": None}},  # None ID
                "status": None,  # None status
                "updatedAt": None,  # None timestamp
            }
        ]
    }
    result = formatter.format_list_response(list_response)
    assert "Found 1 documents" in result[0]["text"]

    # Get response with None content
    result = formatter.format_get_response("doc123", "kb456", {"content": None})
    assert "No content available" in result[4]["text"]

    # Retrieve response with missing fields
    retrieve_response = {
        "retrievalResults": [
            {
                # Missing score
                "location": {"customDocumentLocation": {"id": "doc123"}},
                "content": {"text": "Content"},
            }
        ]
    }
    result = formatter.format_retrieve_response(retrieve_response)
    assert "0.0000" in result[0]["text"]  # Default score


def test_format_retrieve_response_missing_location(formatter):
    """Test retrieve response formatting with missing location data."""
    response = {
        "retrievalResults": [
            {
                "score": 0.95,
                # Missing location entirely
                "content": {"text": "Content without location"},
            }
        ]
    }

    result = formatter.format_retrieve_response(response)
    text = result[0]["text"]
    assert "unknown" in text  # Default document ID
    assert "Content without location" in text
