"""Custom exceptions for llm-optimizer."""


class LLMOptimizerError(Exception):
    """Base exception for all llm-optimizer errors."""


class ServerNotReadyError(LLMOptimizerError):
    pass
