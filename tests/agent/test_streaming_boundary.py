"""Native streaming visibility evidence using real tools and transaction-bound state."""

from collections.abc import Iterator

from conftest import ScriptedModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore


class LateToolStreamingModel(ScriptedModel):
    """Emit ambiguous early text before a later native tool-call chunk."""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        """Expose native early text and final text through the same model node.

        Args:
            messages: Framework-managed messages including real tool evidence.
            stop: Unused framework stop sequences.
            run_manager: Framework streaming callback context.
            kwargs: Native invocation options.

        Yields:
            Controlled incremental model output, never a frontend typing simulation.
        """
        self.seen.append(list(messages))
        if any(isinstance(item, ToolMessage) for item in messages):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="Resposta fictícia confirmada.")
            )
        else:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="untrusted-intermediate-marker")
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": "hidden-reasoning-marker"},
                    tool_call_chunks=[
                        {
                            "name": "get_product_details",
                            "args": '{"product_id":101}',
                            "id": "native-probe-tool",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            )


def test_native_message_stream_does_not_identify_final_text_before_late_tool_calls(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Demonstrate why pre-commit model chunks are unsuitable for public SSE deltas.

    Args:
        demo_engine: Fingerprint-verified isolated catalog.
        state_store: Official checkpointer restricted to disposable test storage.
        offline_settings: Bounded configuration without credentials or network.
    """
    model = LateToolStreamingModel(responses=[])
    with create_store_agent(
        demo_engine, settings=offline_settings, model=model, state_store=state_store
    ) as agent:
        conversation = agent.create_conversation()
        config = {"configurable": {"thread_id": str(conversation.conversation_id)}}
        with state_store.authorized(
            conversation.conversation_id, conversation.access_token.get_secret_value()
        ) as connection:
            graph = agent._build_graph(state_store.saver(connection))
            stream = graph.stream(
                {
                    "messages": [
                        HumanMessage(content="Consulte o produto fictício 101.")
                    ]
                },
                config,
                stream_mode="messages",
            )
            first, first_metadata = next(stream)
            assert first.content == "untrusted-intermediate-marker"
            assert first.tool_call_chunks == [] and first.response_metadata == {}
            # A separate connection cannot observe a committed turn at this point.
            with state_store.transaction() as observer:
                assert state_store.saver(observer).get_tuple(config) is None
            remaining = list(stream)
            assert any(
                isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks
                for chunk, _ in remaining
            )
            final, final_metadata = next(
                (chunk, metadata)
                for chunk, metadata in remaining
                if chunk.content == "Resposta fictícia confirmada."
            )
            assert (
                first_metadata["langgraph_node"]
                == final_metadata["langgraph_node"]
                == "model"
            )
            assert final.tool_call_chunks == []
        public = agent.get_conversation_history(
            conversation.conversation_id, conversation.access_token
        )
        assert [item.text for item in public] == [
            "Consulte o produto fictício 101.",
            "Resposta fictícia confirmada.",
        ]
        assert len(model.seen) == 2
