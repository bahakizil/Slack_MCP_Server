#!/usr/bin/env python3
"""
Slack MCP Server
An autonomous AI assistant that manages Slack workspaces and researches web information.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import McpError
from openai import OpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from autonomous_agent import AutonomousAgent

load_dotenv()


class SlackMCPServer:
    """A Slack MCP Server implementation."""

    def __init__(self):
        self.mcp = FastMCP("Slack AI Assistant")
        self.slack_client = self._init_slack()
        self.openai_client = self._init_openai()
        self.autonomous_agent = AutonomousAgent()
        self._register_tools()
        self._register_routes()

    def _init_slack(self) -> Optional[WebClient]:
        """Initialize Slack client."""
        token = os.getenv("SLACK_BOT_TOKEN")
        if not token:
            return None

        try:
            client = WebClient(token=token)
            client.auth_test()
            return client
        except SlackApiError:
            return None

    def _init_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _register_tools(self) -> None:
        """Register all MCP tools."""
        self._register_slack_tools()
        self._register_ai_tools()

    def _register_slack_tools(self) -> None:
        """Register Slack workspace tools."""

        @self.mcp.tool
        def send_slack_message(channel: str, message: str) -> str:
            """Send message to Slack channel."""
            if not self.slack_client:
                raise McpError("Slack not connected")

            try:
                response = self.slack_client.chat_postMessage(
                    channel=channel.lstrip("#"), text=message
                )
                return f"Message sent: {response['ts']}"
            except SlackApiError as e:
                raise McpError(f"Error: {e.response['error']}") from e

        @self.mcp.tool
        def get_slack_channels() -> List[Dict[str, Any]]:
            """Get list of Slack channels."""
            if not self.slack_client:
                raise McpError("Slack not connected")

            try:
                response = self.slack_client.conversations_list(exclude_archived=True)
                return response["channels"]
            except SlackApiError as e:
                raise McpError(f"Error: {e.response['error']}") from e

        @self.mcp.tool
        def get_slack_messages(channel: str, limit: int = 50) -> List[Dict[str, Any]]:
            """Get messages from Slack channel."""
            if not self.slack_client:
                raise McpError("Slack not connected")

            try:
                channel_id = self._resolve_channel(channel.lstrip("#"))
                if not channel_id:
                    raise McpError(f"Channel '{channel}' not found")

                response = self.slack_client.conversations_history(
                    channel=channel_id, limit=min(limit, 100)
                )

                messages = []
                for msg in response["messages"]:
                    user_name = self._get_user_name(msg.get("user", ""))
                    messages.append(
                        {
                            "user": user_name,
                            "text": msg.get("text", ""),
                            "timestamp": msg.get("ts", ""),
                            "datetime": self._format_timestamp(msg.get("ts")),
                        }
                    )

                return messages
            except SlackApiError as e:
                raise McpError(f"Error: {e.response['error']}") from e

    def _register_ai_tools(self) -> None:
        """Register AI-powered tools."""

        @self.mcp.tool
        def ask_ai(question: str, context: Optional[str] = None) -> str:
            """Ask AI a question."""
            if not self.openai_client:
                raise McpError("OpenAI not connected")

            try:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."}
                ]

                if context:
                    messages.append({"role": "system", "content": f"Context: {context}"})

                messages.append({"role": "user", "content": question})

                response = self.openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                )

                return response.choices[0].message.content
            except Exception as e:
                raise McpError(f"AI Error: {e}") from e

        @self.mcp.tool
        async def autonomous_assistant(
            request: str,
            channel: Optional[str] = None,
            send_to_slack: bool = False,
        ) -> str:
            """Autonomous AI that discovers and uses tools dynamically."""
            try:
                # Prepare context
                context = {}
                if channel:
                    context["slack_channel"] = channel
                    context["can_send_to_slack"] = send_to_slack

                # Let autonomous agent handle everything
                result = await self.autonomous_agent.execute(request, context)

                # Send to Slack if requested
                if send_to_slack and channel and self.slack_client:
                    try:
                        self.slack_client.chat_postMessage(
                            channel=channel.lstrip("#"),
                            text=f"**Autonomous AI**\n\n{result}",
                        )
                        return f"{result}\n\nSent to #{channel}"
                    except SlackApiError:
                        return f"{result}\n\nFailed to send to Slack"

                return result

            except Exception as e:
                raise McpError(f"Autonomous error: {e}") from e

    def _register_routes(self) -> None:
        """Register additional HTTP endpoints."""

        @self.mcp.get("/health")
        def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "slack": bool(self.slack_client),
                    "openai": bool(self.openai_client),
                },
            }

        @self.mcp.get("/api/v1/info")
        def server_info() -> Dict[str, Any]:
            """Server information endpoint."""
            tools = self.mcp.list_tools()
            return {
                "name": "Slack MCP Server",
                "version": "1.0.0",
                "description": "Autonomous AI Slack assistant with MCP tools",
                "tools_count": len(tools),
                "tools": [
                    {"name": tool["name"], "description": tool.get("description", "")}
                    for tool in tools
                ],
                "services": {
                    "slack_connected": bool(self.slack_client),
                    "openai_connected": bool(self.openai_client),
                },
            }

    def _resolve_channel(self, channel_name: str) -> Optional[str]:
        """Get channel ID from name."""
        try:
            response = self.slack_client.conversations_list()
            for ch in response["channels"]:
                if ch["name"] == channel_name:
                    return ch["id"]
        except Exception:
            pass
        return None

    def _get_user_name(self, user_id: str) -> str:
        """Get user display name."""
        if not user_id or not self.slack_client:
            return "Unknown"

        try:
            response = self.slack_client.users_info(user=user_id)
            user = response["user"]
            return user.get("real_name") or user.get("name", "Unknown")
        except Exception:
            return user_id

    def _format_timestamp(self, ts: Optional[str]) -> str:
        """Format timestamp to readable format."""
        if not ts:
            return ""
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ts

    def run(self) -> None:
        """Run the MCP server."""
        print("Starting Slack MCP Server")
        print(
            f"Server running on http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8001')}"
        )
        print("Available tools:")

        # List available tools
        tools = self.mcp.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")

        self.mcp.run(
            transport="streamable-http",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8001")),
        )


def main() -> None:
    """Entry point."""
    server = SlackMCPServer()
    server.run()


if __name__ == "__main__":
    main()