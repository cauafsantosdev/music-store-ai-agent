"""Explicit GroqCloud and LangChain create_agent factories."""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy import Engine

from music_store_agent.agent.admission import AdmissionController
from music_store_agent.agent.config import AgentSettings, load_settings
from music_store_agent.agent.demo import dataset_mode, verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.memory import ConversationStore, create_state_store
from music_store_agent.agent.prompts import SYSTEM_PROMPT
from music_store_agent.agent.service import StoreAgent
from music_store_agent.agent.tool_registry import build_tools, tool_failure


def create_live_model(settings: AgentSettings) -> ChatGroq:
    """Construct the GroqCloud client lazily without making a model request.

    Args:
        settings: Validated model, key, timeout, and output limits.

    Returns:
        A GroqCloud model with bounded output and no automatic retries or paid tiers.

    Raises:
        AgentFailure: If the key is missing or client configuration fails.
    """
    if (
        settings.groq_api_key is None
        or not settings.groq_api_key.get_secret_value().strip()
    ):
        raise AgentFailure(ErrorCode.MISSING_API_KEY)
    try:
        # Retain SDK model defaults; never request paid tiers or parallel tool calls.
        return ChatGroq(
            model=settings.ai_model,
            api_key=settings.groq_api_key,
            # The Groq API replaces max_tokens with max_completion_tokens.
            model_kwargs={"max_completion_tokens": settings.ai_max_output_tokens},
            timeout=settings.ai_provider_timeout_seconds,
            max_retries=0,
        )
    except Exception:
        raise AgentFailure(ErrorCode.CONFIGURATION) from None


def create_store_agent(
    engine: Engine,
    *,
    settings: AgentSettings | None = None,
    model: BaseChatModel | None = None,
    state_store: ConversationStore | None = None,
) -> StoreAgent:
    """Create a bounded tool agent only after synthetic provenance is verified.

    Args:
        engine: Explicit, confirmed synthetic namespace; never a fallback schema.
        settings: Validated environment settings, or lazily read when omitted.
        model: Controlled chat model for offline tests; omitted for live GroqCloud.
        state_store: Explicit initialized store, or an owned default pool.

    Returns:
        A durable service; close() or a context manager releases its owned pool.

    Raises:
        AgentFailure: If data provenance, API key, or model configuration is invalid.
    """
    settings = settings or load_settings()
    if dataset_mode(engine) != settings.ai_dataset_mode:
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    verify_demo(engine)
    model = model if model is not None else create_live_model(settings)
    owns_store = state_store is None
    if state_store is None:
        state_store = (
            create_state_store(settings.ai_state_schema, dataset_mode="challenge")
            if settings.ai_dataset_mode == "challenge"
            else create_state_store(settings.ai_state_schema)
        )
    else:
        if state_store.dataset_mode != settings.ai_dataset_mode:
            raise AgentFailure(ErrorCode.SESSION_UNVERIFIED)
        state_store.check_ready()
    admission = AdmissionController(state_store, settings)
    try:
        admission.check_ready()
    except BaseException:
        if owns_store:
            state_store.close()
        raise

    @wrap_model_call
    def guard_model(
        request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Recheck data before every external model request, not only at startup.

        Args:
            request: Framework-managed model context and native messages.
            handler: Next supported model invocation handler.

        Returns:
            Unmodified native model response.

        Raises:
            AgentFailure: If the selected dataset is no longer confirmed synthetic.
        """
        verify_demo(engine)
        if len(request.state["messages"]) > settings.ai_max_history_messages:
            raise AgentFailure(ErrorCode.EXECUTION_LIMIT)
        admission.model_call()
        return handler(request)

    @wrap_tool_call
    def guard_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Sanitize unexpected framework/tool errors while retaining call associations.

        Args:
            request: Framework-validated tool invocation context.
            handler: Next tool execution handler.

        Returns:
            A native tool response or a generic safe error linked to the same call ID.
        """
        try:
            return handler(request)
        except Exception:
            import json

            return ToolMessage(
                content=json.dumps(
                    tool_failure(ErrorCode.INVALID_ARGUMENTS), ensure_ascii=False
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )

    def build_graph(saver: PostgresSaver) -> CompiledStateGraph:
        """Bind an official saver to an authorized transaction.

        Args:
            saver: Saver sharing the operation's outer transaction.

        Returns:
            A create_agent graph preserving all existing execution limits.
        """
        return create_agent(
            model=model,
            tools=build_tools(engine),
            system_prompt=SYSTEM_PROMPT,
            checkpointer=saver,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=settings.ai_max_model_calls, exit_behavior="error"
                ),
                ToolCallLimitMiddleware(
                    run_limit=settings.ai_max_tool_calls, exit_behavior="error"
                ),
                guard_model,
                guard_tool,
            ],
        )

    return StoreAgent(
        engine,
        build_graph,
        settings,
        state_store,
        owns_store=owns_store,
        admission=admission,
    )
