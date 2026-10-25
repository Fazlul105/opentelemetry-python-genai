# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: agent discovers tools from an MCP server."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest import mock

from agents import Agent, Runner
from agents.mcp.server import _MCPServerWithClientSession
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListToolsResult,
)
from mcp.types import (
    Tool as MCPTool,
)

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

DEFAULT_MODEL = "gpt-4o-mini"


class DummyMCPServer(_MCPServerWithClientSession):
    def __init__(self) -> None:
        super().__init__(
            cache_tools_list=False,
            client_session_timeout_seconds=None,
        )
        self._name = "mcp_discovery_server"
        self.tool = MCPTool(
            name="get_temperature",
            description="Retrieve temperature for a city",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    def create_streams(self) -> Any:
        pass

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self.session = mock.MagicMock()

    async def cleanup(self) -> None:
        pass

    async def _list_tools_page(
        self, session: Any, cursor: str | None = None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[self.tool])

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return CallToolResult(content=[])

    async def list_prompts(
        self, run_context: Any = None, agent: Any = None
    ) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        run_context: Any = None,
        agent: Any = None,
    ) -> GetPromptResult:
        return GetPromptResult(messages=[])


class DiscoveryScenario(Scenario):
    expected_spans = {
        "invoke_workflow": 1,
        "invoke_agent": 1,
    }
    expected_metrics = (
        "gen_ai.invoke_agent.duration",
        "gen_ai.invoke_workflow.duration",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        key_override = (
            {}
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                OpenAIAgentsInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("discovery_conformance.yaml"):
                    server = DummyMCPServer()

                    async def _run() -> None:
                        await server.connect()
                        agent = Agent(
                            name="discovery_agent",
                            instructions="Answer user questions using tools if needed.",
                            mcp_servers=[server],
                            model=DEFAULT_MODEL,
                        )
                        await Runner.run(
                            agent, "What is the weather in Barcelona?"
                        )

                    asyncio.run(_run())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        mcp_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry and entry["span"].get("name") == "tools/list"
        ]
        assert len(mcp_spans) == 1, (
            f"Expected exactly 1 tools/list discovery span; saw {len(mcp_spans)}"
        )
        mcp_span = mcp_spans[0]
        attributes = {
            attr["name"]: attr["value"]
            for attr in mcp_span.get("attributes", [])
        }
        assert attributes.get("mcp.method.name") == "tools/list"
