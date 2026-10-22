# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridges agents-library tracing callbacks to opentelemetry-util-genai.

The agents library exposes a public extension API
(:func:`agents.tracing.add_trace_processor`) for plugging custom
:class:`TracingProcessor` implementations into its own tracing system.
``Trace.start()`` / ``Span.start()`` invoke the registered processors'
``on_*_start`` callbacks *synchronously* on whichever asyncio task
started the agents-library span:

* Workflow (``Trace``) and agent (``AgentSpanData``) spans start in the
  ``Runner.run`` task itself.
* Function tool (``FunctionSpanData``) spans start inside the per-tool
  ``asyncio.Task`` the agents library creates for tool dispatch. That
  sub-task inherits a snapshot of the run-loop context (so workflow +
  agent are already active in OTel contextvars).
* MCP list tools (``MCPListToolsSpanData``) spans start when tool
  discovery requests are dispatched to an MCP server.

Because every ``*_end`` callback fires on the same task as its
matching ``*_start``, util-genai's auto-``attach()`` / ``detach()`` of
OTel context is balanced and no context tokens leak across tasks.
OTel's natural parent tracking nests the tree:

    workflow > invoke_agent > [chat from openai instrumentation,
                               execute_tool, list_tools]

LLM-level spans (``chat`` / ``responses`` / ``embeddings``) are not
emitted here — ``opentelemetry-instrumentation-genai-openai`` patches
the openai SDK directly and produces those.
"""

from __future__ import annotations

import weakref
from typing import Any

from agents.tracing import Span, Trace, TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
)

try:
    from agents.tracing.span_data import MCPListToolsSpanData
except ImportError:  # pragma: no cover
    MCPListToolsSpanData = None  # type: ignore[assignment]

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GenAiOutputTypeValues,
)
from opentelemetry.semconv._incubating.attributes.server_attributes import (
    SERVER_ADDRESS,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai._invocation import (
    GenAIInvocation,
    _json_dumps_unless_str,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    LocalAgentInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.utils import (
    object_to_any_value,
    tool_arguments_to_any_value,
)
from opentelemetry.util.types import AttributeValue

# Non-semconv attribute: surfaces the workflow name on the workflow span
# so callers can query/filter by it. util-genai's WorkflowInvocation
# only puts the name in the span name, not as an attribute.
_WORKFLOW_NAME_ATTR = "gen_ai.workflow.name"


def _is_mcp_list_tools_span(span_data: Any) -> bool:
    if MCPListToolsSpanData is not None and isinstance(
        span_data, MCPListToolsSpanData
    ):
        return True
    if getattr(span_data, "type", None) == "mcp_tools":
        return True
    return (
        getattr(span_data, "__class__", None) is not None
        and span_data.__class__.__name__ == "MCPListToolsSpanData"
    )


class _MCPListToolsInvocation(GenAIInvocation):
    """Invocation for MCP list_tools discovery calls."""

    def __init__(
        self,
        handler: TelemetryHandler,
        server: str | None = None,
    ) -> None:
        server_str = str(server) if server else None
        span_name = f"list_tools {server_str}" if server_str else "list_tools"
        start_attributes: dict[str, AttributeValue] = {
            GEN_AI_OPERATION_NAME: "list_tools",
            GEN_AI_TOOL_TYPE: "function",
        }
        if server_str:
            start_attributes[GEN_AI_TOOL_NAME] = server_str
            start_attributes[SERVER_ADDRESS] = server_str

        super().__init__(
            tracer=handler._tracer,
            instruments=handler._instruments,
            logger=handler._logger,
            completion_hook=handler._completion_hook,
            operation_name="list_tools",
            span_name=span_name,
            span_kind=SpanKind.CLIENT,
            start_attributes=start_attributes,
            content_capturing_mode=handler._content_capturing_mode,
        )
        self.tool_result: Any = None

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        attrs: dict[str, AttributeValue] = {
            GEN_AI_OUTPUT_TYPE: GenAiOutputTypeValues.JSON.value,
        }
        if self.tool_result is not None:
            serialized = _json_dumps_unless_str(self.tool_result)
            if serialized is not None:
                if self.should_capture_content:
                    attrs[GEN_AI_TOOL_DEFINITIONS] = serialized
                if self._should_capture_content_on_span:
                    attrs[GEN_AI_TOOL_CALL_RESULT] = serialized
        attrs.update(self.attributes)
        self.span.set_attributes(attrs)


class GenAITracingProcessor(TracingProcessor):
    """Translate agents-library tracing into util-genai invocations.

    Stateful only for span lifetime: each in-flight Trace/Span has one
    entry in a :class:`weakref.WeakKeyDictionary` keyed by the
    agents-library object itself. Entries are removed on ``*_end`` or
    garbage-collected with the agents-library span/trace if the library
    drops it before ``end`` (which it shouldn't, but the weak reference
    is belt-and-suspenders against any future leak).
    """

    def __init__(self, handler: TelemetryHandler, provider: str) -> None:
        self._handler = handler
        self._provider = provider
        self._invocations: weakref.WeakKeyDictionary[
            Any,
            WorkflowInvocation
            | LocalAgentInvocation
            | ToolInvocation
            | _MCPListToolsInvocation,
        ] = weakref.WeakKeyDictionary()

    def on_trace_start(self, trace: Trace) -> None:
        # ``trace.name`` comes from ``RunConfig.workflow_name`` (default
        # "Agent workflow"). Callers customize it via the agents library's
        # own ``Runner.run(..., run_config=RunConfig(workflow_name=...))``;
        # we don't expose a second knob.
        #
        # ``group_id`` is public on ``TraceImpl`` but not declared on the
        # ``Trace`` ABC, hence the getattr.
        group_id: str | None = getattr(trace, "group_id", None)
        invocation = self._handler.workflow(
            name=trace.name,
            conversation_id=group_id,
        )
        if trace.name:
            invocation.attributes[_WORKFLOW_NAME_ATTR] = trace.name
        self._invocations[trace] = invocation

    def on_trace_end(self, trace: Trace) -> None:
        invocation = self._invocations.pop(trace, None)
        if invocation is not None:
            invocation.stop()

    def on_span_start(self, span: Span[Any]) -> None:
        span_data = span.span_data
        if isinstance(span_data, AgentSpanData):
            invocation = self._handler.invoke_local_agent(
                agent_name=span_data.name,
            )
            self._invocations[span] = invocation
            return
        if isinstance(span_data, FunctionSpanData):
            invocation = self._handler.tool(
                name=span_data.name,
                tool_type="function",
            )
            self._invocations[span] = invocation
            return
        if _is_mcp_list_tools_span(span_data):
            server: str | None = getattr(span_data, "server", None)
            invocation = _MCPListToolsInvocation(
                self._handler,
                server=server,
            )
            self._invocations[span] = invocation
            return
        # Other span_data types (GenerationSpanData, ResponseSpanData,
        # HandoffSpanData, GuardrailSpanData, Speech/TranscriptionSpanData)
        # are intentionally ignored. LLM-level spans come from the openai
        # instrumentation; the rest have no semconv yet.

    def on_span_end(self, span: Span[Any]) -> None:
        invocation = self._invocations.pop(span, None)
        if invocation is None:
            return
        if (
            isinstance(invocation, ToolInvocation)
            and isinstance(span.span_data, FunctionSpanData)
            and invocation.should_capture_content
        ):
            arguments = span.span_data.input
            if arguments:
                invocation.arguments = tool_arguments_to_any_value(arguments)
            invocation.tool_result = object_to_any_value(span.span_data.output)
        elif isinstance(
            invocation, _MCPListToolsInvocation
        ) and _is_mcp_list_tools_span(span.span_data):
            invocation.tool_result = getattr(span.span_data, "result", None)
        # SpanError is a mapping, not a raised exception, so it never
        # reaches util-genai's exception path on its own.
        span_error = getattr(span, "error", None)
        if span_error is not None:
            invocation.fail(
                Error(
                    type=ErrorTypeValues.OTHER.value,
                    message=str(span_error),
                )
            )
            return
        invocation.stop()

    def shutdown(self) -> None:
        for invocation in list(self._invocations.values()):
            try:
                invocation.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        self._invocations.clear()

    def force_flush(self) -> None:  # pragma: no cover - nothing to flush
        pass
