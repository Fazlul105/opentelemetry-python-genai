# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.attributes.mcp_attributes import (
    MCP_METHOD_NAME,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    FunctionToolDefinition,
    GenericToolDefinition,
)


class TestDiscoveryHandler(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

    def test_discovery_default_attributes_and_span_name(self) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.name, "tools/list")
        self.assertIs(span.kind, SpanKind.CLIENT)
        assert span.attributes is not None
        self.assertEqual(span.attributes[MCP_METHOD_NAME], "tools/list")
        self.assertNotIn("gen_ai.operation.name", span.attributes)
        self.assertNotIn("server.address", span.attributes)
        self.assertNotIn("gen_ai.tool.name", span.attributes)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_TYPE, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, span.attributes)

    def test_discovery_with_explicit_server_address(self) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery(
            server_address="localhost", server_port=8080
        )
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        self.assertEqual(span.attributes["server.address"], "localhost")
        self.assertEqual(span.attributes["server.port"], 8080)

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.SPAN_ONLY,
    )
    def test_discovery_structured_tool_definitions_capture(
        self, _mock_cap: MagicMock
    ) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.tool_definitions = [
            "read_file",
            GenericToolDefinition(name="write_file", type="function"),
            FunctionToolDefinition(
                name="list_dir",
                description="List directory",
                parameters={"type": "object"},
            ),
        ]
        invocation.tool_result = ["read_file", "write_file", "list_dir"]
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        definitions = json.loads(
            span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
        )
        self.assertEqual(len(definitions), 3)
        self.assertEqual(
            definitions[0], {"name": "read_file", "type": "function"}
        )
        self.assertEqual(
            definitions[1], {"name": "write_file", "type": "function"}
        )
        self.assertEqual(
            definitions[2],
            {
                "name": "list_dir",
                "description": "List directory",
                "parameters": {"type": "object"},
                "type": "function",
            },
        )
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_TYPE, span.attributes)

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.EVENT_ONLY,
    )
    def test_discovery_omits_content_in_event_only_mode(
        self, _mock_cap: MagicMock
    ) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.tool_definitions = ["read_file"]
        invocation.tool_result = ["read_file"]
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        self.assertNotIn(GenAI.GEN_AI_TOOL_DEFINITIONS, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_TYPE, span.attributes)

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.NO_CONTENT,
    )
    def test_discovery_forwards_to_completion_hook_without_span_leak(
        self, _mock_cap: MagicMock
    ) -> None:
        hook = MagicMock()
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            completion_hook=hook,
        )
        invocation = handler.discovery()
        invocation.tool_definitions = ["read_file"]
        invocation.tool_result = ["read_file"]
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        self.assertNotIn(GenAI.GEN_AI_TOOL_DEFINITIONS, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_TYPE, span.attributes)

        hook.on_completion.assert_called_once()
        kwargs = hook.on_completion.call_args.kwargs
        self.assertEqual(len(kwargs["tool_definitions"]), 1)
        self.assertEqual(kwargs["tool_definitions"][0].name, "read_file")

    def test_discovery_error_recording(self) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.fail(Error(type="_OTHER", message="connection refused"))

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        assert span.attributes is not None
        self.assertEqual(span.attributes["error.type"], "_OTHER")

    def test_mcp_list_tools_alias(self) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.mcp_list_tools()
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.name, "tools/list")
        assert span.attributes is not None
        self.assertEqual(span.attributes[MCP_METHOD_NAME], "tools/list")
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_TYPE, span.attributes)
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, span.attributes)

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.SPAN_ONLY,
    )
    def test_discovery_normalizes_mapping_input_schema_and_empty_parameters(
        self, _mock_cap: MagicMock
    ) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.tool_definitions = [
            {
                "name": "tool_with_input_schema_camel",
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
            {
                "name": "tool_with_input_schema_snake",
                "input_schema": {
                    "type": "object",
                    "properties": {"v": {"type": "integer"}},
                },
            },
            {
                "name": "tool_with_empty_parameters",
                "parameters": {},
            },
        ]
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        definitions = json.loads(
            span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
        )
        self.assertEqual(len(definitions), 3)
        self.assertEqual(
            definitions[0]["name"], "tool_with_input_schema_camel"
        )
        self.assertEqual(
            definitions[0]["parameters"],
            {"type": "object", "properties": {"q": {"type": "string"}}},
        )
        self.assertEqual(
            definitions[1]["name"], "tool_with_input_schema_snake"
        )
        self.assertEqual(
            definitions[1]["parameters"],
            {"type": "object", "properties": {"v": {"type": "integer"}}},
        )
        self.assertEqual(definitions[2]["name"], "tool_with_empty_parameters")
        self.assertEqual(definitions[2]["parameters"], {})

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.SPAN_ONLY,
    )
    def test_discovery_normalizes_object_input_schema_and_empty_parameters(
        self, _mock_cap: MagicMock
    ) -> None:
        class ObjectWithCamelInputSchema:
            name = "obj_camel"
            description = "Camel desc"
            inputSchema = {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }

        class ObjectWithSnakeInputSchema:
            name = "obj_snake"
            input_schema = {
                "type": "object",
                "properties": {"b": {"type": "string"}},
            }

        class ObjectWithEmptyParameters:
            name = "obj_empty_params"
            parameters = {}

        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        invocation = handler.discovery()
        invocation.tool_definitions = [
            ObjectWithCamelInputSchema(),
            ObjectWithSnakeInputSchema(),
            ObjectWithEmptyParameters(),
        ]
        invocation.stop()

        (span,) = self.span_exporter.get_finished_spans()
        assert span.attributes is not None
        definitions = json.loads(
            span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
        )
        self.assertEqual(len(definitions), 3)
        self.assertEqual(definitions[0]["name"], "obj_camel")
        self.assertEqual(
            definitions[0]["parameters"],
            {"type": "object", "properties": {"a": {"type": "string"}}},
        )
        self.assertEqual(definitions[1]["name"], "obj_snake")
        self.assertEqual(
            definitions[1]["parameters"],
            {"type": "object", "properties": {"b": {"type": "string"}}},
        )
        self.assertEqual(definitions[2]["name"], "obj_empty_params")
        self.assertEqual(definitions[2]["parameters"], {})

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.SPAN_ONLY,
    )
    def test_discovery_skips_building_content_attributes_when_span_not_recording(
        self, _mock_cap: MagicMock
    ) -> None:
        from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

        sampler_provider = TracerProvider(sampler=ALWAYS_OFF)
        hook = MagicMock()
        handler = TelemetryHandler(
            tracer_provider=sampler_provider,
            completion_hook=hook,
        )
        invocation = handler.discovery()
        self.assertFalse(invocation.span.is_recording())
        invocation.tool_definitions = [
            {
                "name": "sampled_out_tool",
                "parameters": {"type": "object"},
            }
        ]
        with patch(
            "opentelemetry.util.genai._discovery_invocation.get_content_attributes"
        ) as mock_get_content:
            invocation.stop()
            mock_get_content.assert_not_called()

        hook.on_completion.assert_called_once()
        kwargs = hook.on_completion.call_args.kwargs
        self.assertEqual(len(kwargs["tool_definitions"]), 1)
        self.assertEqual(
            kwargs["tool_definitions"][0].name, "sampled_out_tool"
        )
