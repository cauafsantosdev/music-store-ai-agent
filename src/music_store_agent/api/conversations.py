"""Thin versioned HTTP routes delegating to the durable application service."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.service import AgentReply, Conversation, StoreAgent
from music_store_agent.api.dependencies import (
    authorized_conversation,
    get_agent,
    reject_query_parameters,
)
from music_store_agent.api.errors import APIError, checked_reply
from music_store_agent.api.schemas import (
    ConversationCreated,
    ConversationHistory,
    EmptyRequest,
    ErrorResponse,
    MessageRequest,
)

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
    dependencies=[Depends(reject_query_parameters)],
    responses={
        status: {"model": ErrorResponse}
        for status in (401, 404, 413, 422, 429, 500, 503, 504)
    },
)


@router.post("", status_code=201, response_model=ConversationCreated)
def create_conversation(
    agent: Annotated[StoreAgent, Depends(get_agent)],
    payload: EmptyRequest | None = None,
) -> ConversationCreated:
    """Create a durable conversation and return its capability exactly once.

    Args:
        payload: Optional empty creation payload; no internal overrides are accepted.
        agent: Existing durable application service.

    Returns:
        The conversation UUID and opaque access capability.
    """
    conversation = agent.create_conversation()
    return ConversationCreated(
        conversation_id=conversation.conversation_id,
        access_token=conversation.access_token.get_secret_value(),
    )


@router.get("/{conversation_id}", response_model=ConversationHistory)
def get_history(
    conversation: Annotated[Conversation, Depends(authorized_conversation)],
    agent: Annotated[StoreAgent, Depends(get_agent)],
) -> ConversationHistory:
    """Retrieve only authorized display history, never native checkpoint messages.

    Args:
        conversation: Previously authorized capability handle.
        agent: Existing durable application service.

    Returns:
        Filtered user and assistant text.
    """
    return ConversationHistory(
        conversation_id=conversation.conversation_id,
        messages=agent.get_conversation_history(
            conversation.conversation_id, conversation.access_token
        ),
    )


@router.post("/{conversation_id}/messages", response_model=AgentReply)
def send_message(
    payload: MessageRequest,
    conversation: Annotated[Conversation, Depends(authorized_conversation)],
    agent: Annotated[StoreAgent, Depends(get_agent)],
) -> AgentReply:
    """Execute a single bounded turn through the synchronous application service.

    FastAPI offloads this blocking route to its worker pool. No prompt creation,
    pricing logic or checkpoint manipulation occurs at the HTTP boundary.

    Args:
        payload: Validated synthetic input.
        conversation: Authorized capability handle.
        agent: Existing durable service.

    Returns:
        A typed completed reply or a translated HTTP failure.
    """
    return checked_reply(
        agent.send_message(
            conversation.conversation_id, conversation.access_token, payload.message
        )
    )


def event_frame(event: str, payload: dict[str, object]) -> str:
    """Frame one SSE event with JSON escaping rather than interpolated user text.

    Args:
        event: Fixed event identifier controlled by the application.
        payload: Public JSON-compatible event data.

    Returns:
        A complete SSE frame terminated by a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/{conversation_id}/stream",
    response_class=StreamingResponse,
    description=(
        "Deliver processing, completed-message/error, and done SSE events. "
        "This endpoint does not provide model-token streaming."
    ),
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def stream_message(
    payload: MessageRequest,
    conversation: Annotated[Conversation, Depends(authorized_conversation)],
    agent: Annotated[StoreAgent, Depends(get_agent)],
) -> StreamingResponse:
    """Start status delivery before one offloaded, complete service execution.

    Credentials and input are validated before headers. Once execution begins,
    disconnect cannot safely interrupt synchronous PostgreSQL transactions. The
    bounded worker may commit after disconnect; clients must not automatically
    retry. No detached task or fabricated model-token fragments are created.

    Args:
        payload: Validated synthetic input.
        conversation: Previously authorized capability handle.
        agent: Existing durable service.

    Returns:
        An unbuffered SSE response with stable JSON event payloads.
    """

    async def events() -> AsyncIterator[str]:
        """Deliver one completed result or sanitized error, followed by completion.

        Yields:
            Status, completed message/error and done frames, never raw tool events.
        """
        yield event_frame("status", {"status": "processing"})
        failure: APIError | None = None
        reply: AgentReply | None = None
        try:
            # Default thread offloading waits for bounded transactional work instead
            # of abandoning an indefinitely running background task on disconnect.
            reply = checked_reply(
                await run_in_threadpool(
                    agent.send_message,
                    conversation.conversation_id,
                    conversation.access_token,
                    payload.message,
                )
            )
        except AgentFailure as error:
            failure = APIError(error.code.value)
        except APIError as error:
            failure = error
        except Exception:
            failure = APIError("internal_error")
        if failure is not None:
            yield event_frame(
                "error",
                {
                    **failure.envelope().model_dump(),
                    "status_code": failure.status_code,
                },
            )
        elif reply is not None:
            yield event_frame("message", reply.model_dump())
        yield event_frame("done", {"status": "failed" if failure else "completed"})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
    )


@router.delete("/{conversation_id}", status_code=204, response_class=Response)
def delete_conversation(
    conversation: Annotated[Conversation, Depends(authorized_conversation)],
    agent: Annotated[StoreAgent, Depends(get_agent)],
) -> Response:
    """Delete only the authorized session through the existing transaction service.

    Args:
        conversation: Previously authorized capability handle.
        agent: Existing durable service.

    Returns:
        An empty no-content response.
    """
    agent.delete_conversation(conversation.conversation_id, conversation.access_token)
    return Response(status_code=204)
