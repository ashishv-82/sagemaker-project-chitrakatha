"""
Custom exception hierarchy for Project Chitrakatha.
"""

class ChitrakathaBaseError(Exception):
    """Base exception for all Chitrakatha errors."""
    pass


class SageMakerPipelineError(ChitrakathaBaseError):
    """Raised when a SageMaker Pipeline step fails or is misconfigured."""
    pass


class BedrockEmbeddingError(ChitrakathaBaseError):
    """Raised when a Bedrock Titan embedding API call fails."""
    pass


class BedrockSynthesisError(ChitrakathaBaseError):
    """Raised when a Bedrock Claude synthesis (RAFT pair generation) call fails."""
    pass


class S3VectorError(ChitrakathaBaseError):
    """Raised when interacting with S3 Vectors index fails."""
    pass


class DataIngestionError(ChitrakathaBaseError):
    """Raised when raw data fails schema validation or Unicode expectations."""
    pass
