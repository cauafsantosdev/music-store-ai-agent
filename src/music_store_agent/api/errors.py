"""Explicit service-to-HTTP error translation with sanitized response bodies."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from music_store_agent.agent.errors import PUBLIC_MESSAGES, AgentFailure, ErrorCode
from music_store_agent.agent.service import AgentReply
from music_store_agent.api.schemas import ErrorDetail, ErrorResponse

STATUS_CODES: dict[str, int] = {
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.INVALID_ARGUMENTS: 422,
    ErrorCode.SESSION_UNVERIFIED: 404,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.RATE_LIMIT: 429,
    ErrorCode.EXECUTION_LIMIT: 429,
    ErrorCode.QUOTA_EXHAUSTED: 429,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.ORDER_UNVERIFIED: 404,
    **{
        code: 503
        for code in (
            ErrorCode.CONFIGURATION,
            ErrorCode.MISSING_API_KEY,
            ErrorCode.AUTHENTICATION,
            ErrorCode.PROVIDER_UNAVAILABLE,
            ErrorCode.DATABASE,
            ErrorCode.DEMO_UNVERIFIED,
            ErrorCode.EMPTY_CATALOG,
            ErrorCode.MISSING_POLICY,
            ErrorCode.EMPTY_RESPONSE,
            ErrorCode.STATE_UNINITIALIZED,
        )
    },
    "unauthorized": 401,
    "invalid_request": 422,
    "request_too_large": 413,
    "resource_not_found": 404,
    "method_not_allowed": 405,
    "internal_error": 500,
}
MESSAGES: dict[str, str] = {
    **PUBLIC_MESSAGES,
    "unauthorized": "Informe uma credencial de acesso válida.",
    "invalid_request": "Os dados da solicitação são inválidos.",
    "request_too_large": "A solicitação excede o tamanho permitido.",
    "resource_not_found": "Recurso não encontrado.",
    "method_not_allowed": "Método não permitido.",
    "internal_error": "Não foi possível processar a solicitação agora.",
    ErrorCode.STATE_UNINITIALIZED: "O serviço de conversas está indisponível.",
    ErrorCode.MISSING_API_KEY: "O assistente ainda não está disponível.",
}


class APIError(Exception):
    """Carry only an allowlisted public HTTP failure without raw diagnostic text."""

    code: str
    status_code: int

    def __init__(self, code: str) -> None:
        """Resolve an error identifier without echoing unknown or sensitive values.

        Args:
            code: Existing service code or a fixed HTTP boundary code.
        """
        self.code = code if code in STATUS_CODES else "internal_error"
        self.status_code = STATUS_CODES[self.code]
        super().__init__(MESSAGES[self.code])

    def envelope(self) -> ErrorResponse:
        """Build the public error model.

        Returns:
            A stable sanitized JSON-compatible error envelope.
        """
        return ErrorResponse(error=ErrorDetail(code=self.code, message=str(self)))


def error_response(error: APIError) -> JSONResponse:
    """Encode a fixed error response with capability-safe caching rules.

    Args:
        error: Sanitized HTTP failure.

    Returns:
        The typed envelope and appropriate HTTP status.
    """
    headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
    if error.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        error.envelope().model_dump(), status_code=error.status_code, headers=headers
    )


def checked_reply(reply: AgentReply) -> AgentReply:
    """Reject infrastructure failures instead of presenting them as HTTP success.

    Args:
        reply: Existing service execution result.

    Returns:
        A successful or sanitized business-level no-result response.

    Raises:
        APIError: For execution, infrastructure, or unrecognized service failures.
    """
    if reply.code == "ok":
        return reply
    if reply.code in (ErrorCode.NOT_FOUND, ErrorCode.ORDER_UNVERIFIED):
        return AgentReply(
            text=MESSAGES[reply.code], code=reply.code, tool_calls=reply.tool_calls
        )
    raise APIError(reply.code)


async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
    """Translate an HTTP boundary failure without logging request data.

    Args:
        request: Uninspected incoming request.
        error: Allowlisted failure.

    Returns:
        A sanitized error response.
    """
    return error_response(error)


async def handle_agent_error(request: Request, error: AgentFailure) -> JSONResponse:
    """Translate existing domain failures consistently.

    Args:
        request: Uninspected incoming request.
        error: Existing sanitized service failure.

    Returns:
        A controlled HTTP error response.
    """
    return error_response(APIError(error.code.value))


async def handle_validation_error(
    request: Request, error: RequestValidationError | ResponseValidationError
) -> JSONResponse:
    """Discard validation diagnostics that could echo bodies, credentials or paths.

    Args:
        request: Uninspected incoming request.
        error: Framework validation failure; values are never serialized.

    Returns:
        A sanitized request error or server-side response validation error.
    """
    code = (
        "internal_error"
        if isinstance(error, ResponseValidationError)
        else "invalid_request"
    )
    return error_response(APIError(code))


async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
    """Sanitize framework routing errors without exposing arbitrary detail strings.

    Args:
        request: Uninspected incoming request.
        error: Framework HTTP exception.

    Returns:
        A consistent public error response.
    """
    codes = {
        400: "invalid_request",
        401: "unauthorized",
        404: "resource_not_found",
        405: "method_not_allowed",
        413: "request_too_large",
        422: "invalid_request",
        429: ErrorCode.RATE_LIMIT,
        503: ErrorCode.PROVIDER_UNAVAILABLE,
        504: ErrorCode.PROVIDER_TIMEOUT,
    }
    return error_response(APIError(codes.get(error.status_code, "internal_error")))
