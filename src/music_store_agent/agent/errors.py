"""Stable agent failure codes and sanitized Brazilian Portuguese messages."""

from enum import StrEnum

from groq import APITimeoutError
from httpx import TimeoutException


class ErrorCode(StrEnum):
    """Machine-readable operational failures without private diagnostic details."""

    CONFIGURATION = "configuration"
    MISSING_API_KEY = "missing_api_key"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DATABASE = "database"
    DEMO_UNVERIFIED = "demo_unverified"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    EMPTY_CATALOG = "empty_catalog"
    MISSING_POLICY = "missing_policy"
    ORDER_UNVERIFIED = "order_unverified"
    EXECUTION_LIMIT = "execution_limit"
    INVALID_INPUT = "invalid_input"
    EMPTY_RESPONSE = "empty_response"


PUBLIC_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.CONFIGURATION: "A configuração do assistente é inválida.",
    ErrorCode.MISSING_API_KEY: "Configure GROQ_API_KEY para iniciar o assistente.",
    ErrorCode.AUTHENTICATION: "Não foi possível autenticar o acesso ao assistente.",
    ErrorCode.RATE_LIMIT: (
        "O limite de uso do assistente foi atingido. Tente mais tarde."
    ),
    ErrorCode.PROVIDER_UNAVAILABLE: "O assistente está temporariamente indisponível.",
    ErrorCode.PROVIDER_TIMEOUT: "O assistente está temporariamente indisponível.",
    ErrorCode.QUOTA_EXHAUSTED: "O assistente está temporariamente indisponível.",
    ErrorCode.DATABASE: "Não foi possível consultar os dados da loja agora.",
    ErrorCode.DEMO_UNVERIFIED: "A origem dos dados de demonstração não foi confirmada.",
    ErrorCode.INVALID_ARGUMENTS: (
        "Os dados da consulta são inválidos. Confira os campos."
    ),
    ErrorCode.NOT_FOUND: "Nenhum resultado foi encontrado para esta consulta.",
    ErrorCode.EMPTY_CATALOG: "O catálogo não está disponível para consulta.",
    ErrorCode.MISSING_POLICY: (
        "O índice de políticas não está disponível para consulta."
    ),
    ErrorCode.ORDER_UNVERIFIED: "Não foi possível localizar ou verificar esse pedido.",
    ErrorCode.EXECUTION_LIMIT: (
        "O limite desta consulta foi atingido. Simplifique a pergunta."
    ),
    ErrorCode.INVALID_INPUT: (
        "Envie apenas texto de demonstração, sem dados pessoais reais."
    ),
    ErrorCode.EMPTY_RESPONSE: "Não foi possível obter uma resposta. Tente novamente.",
}


class AgentFailure(RuntimeError):
    """Carry a safe error code without retaining raw provider or database messages."""

    code: ErrorCode

    def __init__(self, code: ErrorCode) -> None:
        """Initialize a sanitized failure.

        Args:
            code: Stable application-level failure classification.
        """
        self.code = code
        super().__init__(PUBLIC_MESSAGES[code])


def provider_failure(error: Exception) -> AgentFailure:
    """Classify provider status codes without exposing diagnostic strings.

    Args:
        error: Groq SDK or transport error, potentially wrapped by the integration.

    Returns:
        A sanitized failure; unknown transport failures mean temporary unavailability.
    """
    current: BaseException | None = error
    for _ in range(4):
        if current is None:
            break
        code = getattr(current, "status_code", None)
        # Groq exposes HTTP status_code; transport timeouts have no status response.
        if isinstance(current, (APITimeoutError, TimeoutError, TimeoutException)) or (
            code in (408, 504)
        ):
            return AgentFailure(ErrorCode.PROVIDER_TIMEOUT)
        if code == 402:
            return AgentFailure(ErrorCode.QUOTA_EXHAUSTED)
        if code in (401, 403):
            return AgentFailure(ErrorCode.AUTHENTICATION)
        if code == 429:
            return AgentFailure(ErrorCode.RATE_LIMIT)
        if code in (400, 404, 413, 422):
            return AgentFailure(ErrorCode.CONFIGURATION)
        current = current.__cause__ or current.__context__
    return AgentFailure(ErrorCode.PROVIDER_UNAVAILABLE)
