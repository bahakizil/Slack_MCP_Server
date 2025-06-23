import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from fastmcp import FastMCP, Client
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackMCPServer:
    def __init__(self):
        self.mcp = FastMCP("Slack MCP Server 🚀")
        self.slack_client = None
        self.openai_client = None
        self.mcp_clients = {}  # Store connected MCP clients
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.use_aws_secrets = os.getenv("USE_AWS_SECRETS", "false").lower() == "true"
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize clients
        self._init_slack_client()
        self._init_openai_client()
        self._init_mcp_configs()
        self._init_integrations()
        
        # Register tools
        self._register_tools()
        
    def _get_secret_from_aws(self, secret_name: str) -> str:
        """Get secret from AWS Secrets Manager"""
        try:
            secrets_client = boto3.client('secretsmanager', region_name=self.aws_region)
            response = secrets_client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except ClientError as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return ""
    
    def _init_slack_client(self):
        """Initialize Slack client with token from environment or AWS Secrets Manager"""
        if self.use_aws_secrets:
            slack_token = self._get_secret_from_aws("slack-mcp/slack-bot-token")
        else:
            slack_token = os.getenv("SLACK_BOT_TOKEN")
        
        if slack_token:
            self.slack_client = WebClient(token=slack_token)
            logger.info("✅ Slack client initialized successfully")
        else:
            logger.warning("⚠️ No Slack token found. Some features may not work.")
    
    def _init_openai_client(self):
        """Initialize OpenAI client with API key from environment or AWS Secrets Manager"""
        if self.use_aws_secrets:
            openai_api_key = self._get_secret_from_aws("slack-mcp/openai-api-key")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
        else:
            logger.warning("⚠️ No OpenAI API key found. AI chat features may not work.")
    
    def _init_mcp_configs(self):
        """Initialize MCP client configurations from environment"""
        # Load MCP server configurations from environment variable
        mcp_servers_config = os.getenv("MCP_SERVERS_CONFIG", "{}")
        try:
            self.mcp_server_configs = json.loads(mcp_servers_config)
            if self.mcp_server_configs:
                logger.info(f"✅ Loaded {len(self.mcp_server_configs)} MCP server configurations")
            else:
                logger.info("ℹ️ No MCP server configurations found. Use add_mcp_server tool to add servers.")
        except json.JSONDecodeError:
            logger.warning("⚠️ Invalid MCP_SERVERS_CONFIG format. Expected JSON.")
            self.mcp_server_configs = {}
    
    async def _connect_to_mcp_server(self, server_name: str, config: Dict[str, Any]) -> bool:
        """Connect to an MCP server"""
        try:
            if server_name in self.mcp_clients:
                return True  # Already connected
            
            client = Client(config.get("url") or config.get("command"))
            await client.__aenter__()  # Initialize connection
            self.mcp_clients[server_name] = client
            logger.info(f"✅ Connected to MCP server: {server_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server {server_name}: {str(e)}")
            return False
    
    async def _disconnect_mcp_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.mcp_clients:
            try:
                await self.mcp_clients[server_name].__aexit__(None, None, None)
                del self.mcp_clients[server_name]
                logger.info(f"✅ Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"❌ Error disconnecting from {server_name}: {str(e)}")
    
    async def _get_all_external_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from connected MCP servers"""
        all_tools = []
        for server_name, client in self.mcp_clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    tool_info = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema
                    }
                    all_tools.append(tool_info)
            except Exception as e:
                logger.error(f"❌ Error getting tools from {server_name}: {str(e)}")
        return all_tools
    
    def _init_integrations(self):
        """Initialize API integrations"""
        # Initialize Trello API configuration
        if self.use_aws_secrets:
            self.trello_api_key = self._get_secret_from_aws("slack-mcp/trello-api-key")
            self.trello_token = self._get_secret_from_aws("slack-mcp/trello-token")
            self.tavily_api_key = self._get_secret_from_aws("slack-mcp/tavily-api-key")
        else:
            self.trello_api_key = os.getenv("TRELLO_API_KEY")
            self.trello_token = os.getenv("TRELLO_TOKEN")
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if self.trello_api_key and self.trello_token:
            logger.info("✅ Trello API credentials loaded")
        else:
            logger.warning("⚠️ Trello API credentials not found. Trello features will not work.")
        
        if self.tavily_api_key:
            logger.info("✅ Tavily API key loaded")
        else:
            logger.warning("⚠️ Tavily API key not found. Use external Tavily MCP server instead.")
    
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.mcp.tool
        def send_slack_message(channel: str, message: str, thread_ts: Optional[str] = None) -> str:
            """
            Send a message to a Slack channel
            
            Args:
                channel: The Slack channel ID or name (e.g., '#general' or 'C1234567890')
                message: The message text to send
                thread_ts: Optional thread timestamp to reply to a specific message
            
            Returns:
                Success message with timestamp or error message
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                # Remove # if present in channel name
                if channel.startswith('#'):
                    channel = channel[1:]
                
                response = self.slack_client.chat_postMessage(
                    channel=channel,
                    text=message,
                    thread_ts=thread_ts
                )
                
                return f"✅ Message sent successfully! Timestamp: {response['ts']}"
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error sending message: {str(e)}"
        
        @self.mcp.tool
        def get_slack_channels(types: str = "public_channel,private_channel") -> str:
            """
            Get list of Slack channels
            
            Args:
                types: Comma-separated list of channel types (public_channel, private_channel, mpim, im)
            
            Returns:
                JSON string with channel information
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                response = self.slack_client.conversations_list(
                    types=types,
                    exclude_archived=True
                )
                
                channels = []
                for channel in response['channels']:
                    channels.append({
                        'id': channel['id'],
                        'name': channel['name'],
                        'is_private': channel.get('is_private', False),
                        'is_member': channel.get('is_member', False),
                        'num_members': channel.get('num_members', 0),
                        'purpose': channel.get('purpose', {}).get('value', ''),
                        'topic': channel.get('topic', {}).get('value', '')
                    })
                
                return json.dumps({
                    'success': True,
                    'channels': channels,
                    'total_count': len(channels)
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error getting channels: {str(e)}"
        
        @self.mcp.tool
        def get_slack_messages(channel: str, limit: int = 100, oldest: Optional[str] = None) -> str:
            """
            Get messages from a Slack channel
            
            Args:
                channel: The Slack channel ID or name
                limit: Number of messages to retrieve (max 1000)
                oldest: Only messages after this timestamp (Unix timestamp)
            
            Returns:
                JSON string with message information
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                # Remove # if present in channel name
                if channel.startswith('#'):
                    channel = channel[1:]
                
                # Get channel ID if name was provided
                if not channel.startswith('C'):
                    channels_response = self.slack_client.conversations_list()
                    channel_id = None
                    for ch in channels_response['channels']:
                        if ch['name'] == channel:
                            channel_id = ch['id']
                            break
                    
                    if not channel_id:
                        return f"❌ Channel '{channel}' not found"
                    channel = channel_id
                
                response = self.slack_client.conversations_history(
                    channel=channel,
                    limit=min(limit, 1000),
                    oldest=oldest
                )
                
                messages = []
                for message in response['messages']:
                    # Get user info if available
                    user_name = message.get('user', 'Unknown')
                    if 'user' in message and message['user']:
                        try:
                            user_info = self.slack_client.users_info(user=message['user'])
                            user_name = user_info['user']['real_name'] or user_info['user']['name']
                        except:
                            pass
                    
                    messages.append({
                        'text': message.get('text', ''),
                        'user': user_name,
                        'timestamp': message.get('ts'),
                        'datetime': datetime.fromtimestamp(float(message.get('ts', 0))).isoformat() if message.get('ts') else None,
                        'type': message.get('type', 'message'),
                        'thread_ts': message.get('thread_ts'),
                        'reply_count': message.get('reply_count', 0)
                    })
                
                return json.dumps({
                    'success': True,
                    'channel': channel,
                    'messages': messages,
                    'message_count': len(messages)
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error getting messages: {str(e)}"
        
        @self.mcp.tool
        def get_slack_user_info(user_id: str) -> str:
            """
            Get information about a Slack user
            
            Args:
                user_id: The Slack user ID
            
            Returns:
                JSON string with user information
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                response = self.slack_client.users_info(user=user_id)
                user = response['user']
                
                user_info = {
                    'id': user['id'],
                    'name': user['name'],
                    'real_name': user.get('real_name', ''),
                    'display_name': user.get('profile', {}).get('display_name', ''),
                    'email': user.get('profile', {}).get('email', ''),
                    'title': user.get('profile', {}).get('title', ''),
                    'phone': user.get('profile', {}).get('phone', ''),
                    'is_admin': user.get('is_admin', False),
                    'is_owner': user.get('is_owner', False),
                    'is_bot': user.get('is_bot', False),
                    'timezone': user.get('tz', ''),
                    'status': user.get('profile', {}).get('status_text', ''),
                    'avatar': user.get('profile', {}).get('image_72', '')
                }
                
                return json.dumps({
                    'success': True,
                    'user': user_info
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error getting user info: {str(e)}"
        
        @self.mcp.tool
        def search_slack_messages(query: str, count: int = 20, sort: str = "timestamp") -> str:
            """
            Search for messages in Slack workspace
            
            Args:
                query: Search query string
                count: Number of results to return (max 100)
                sort: Sort order (timestamp, score)
            
            Returns:
                JSON string with search results
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                response = self.slack_client.search_messages(
                    query=query,
                    count=min(count, 100),
                    sort=sort
                )
                
                results = []
                for match in response.get('messages', {}).get('matches', []):
                    results.append({
                        'text': match.get('text', ''),
                        'user': match.get('user', ''),
                        'username': match.get('username', ''),
                        'channel': match.get('channel', {}).get('name', ''),
                        'timestamp': match.get('ts', ''),
                        'datetime': datetime.fromtimestamp(float(match.get('ts', 0))).isoformat() if match.get('ts') else None,
                        'permalink': match.get('permalink', '')
                    })
                
                return json.dumps({
                    'success': True,
                    'query': query,
                    'results': results,
                    'total_count': response.get('messages', {}).get('total', 0)
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error searching messages: {str(e)}"
        
        @self.mcp.tool
        def create_slack_channel(name: str, is_private: bool = False) -> str:
            """
            Create a new Slack channel
            
            Args:
                name: Channel name (without #)
                is_private: Whether to create a private channel
            
            Returns:
                Success message with channel info or error message
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            try:
                response = self.slack_client.conversations_create(
                    name=name,
                    is_private=is_private
                )
                
                channel = response['channel']
                return json.dumps({
                    'success': True,
                    'message': f"✅ Channel created successfully!",
                    'channel': {
                        'id': channel['id'],
                        'name': channel['name'],
                        'is_private': channel.get('is_private', False),
                        'created': datetime.fromtimestamp(channel.get('created', 0)).isoformat()
                    }
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error creating channel: {str(e)}"
        
        @self.mcp.tool
        def ask_ai_question(question: str, context: Optional[str] = None) -> str:
            """
            Ask a question to OpenAI GPT model
            
            Args:
                question: The question to ask the AI
                context: Optional context to provide to the AI
            
            Returns:
                AI's response to the question
            """
            if not self.openai_client:
                return "❌ OpenAI client not initialized. Please check your OPENAI_API_KEY."
            
            try:
                # Prepare the messages
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."}
                ]
                
                if context:
                    messages.append({"role": "system", "content": f"Additional context: {context}"})
                
                messages.append({"role": "user", "content": question})
                
                # Make the API call
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                return f"🤖 AI Response:\n\n{ai_response}"
                
            except Exception as e:
                return f"❌ Error getting AI response: {str(e)}"
        
        @self.mcp.tool
        def send_ai_response_to_slack(channel: str, question: str, context: Optional[str] = None, thread_ts: Optional[str] = None) -> str:
            """
            Ask AI a question and send the response directly to a Slack channel
            
            Args:
                channel: The Slack channel ID or name
                question: The question to ask the AI
                context: Optional context to provide to the AI
                thread_ts: Optional thread timestamp to reply to a specific message
            
            Returns:
                Success message or error message
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            if not self.openai_client:
                return "❌ OpenAI client not initialized. Please check your OPENAI_API_KEY."
            
            try:
                # Get AI response first
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant integrated into Slack. Provide clear, concise, and accurate responses. Keep responses conversational and appropriate for a team chat environment."}
                ]
                
                if context:
                    messages.append({"role": "system", "content": f"Additional context: {context}"})
                
                messages.append({"role": "user", "content": question})
                
                # Get AI response
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                # Format the message for Slack
                slack_message = f"🤖 *AI Assistant*\n\n**Question:** {question}\n\n**Answer:** {ai_response}"
                
                # Remove # if present in channel name
                if channel.startswith('#'):
                    channel = channel[1:]
                
                # Send to Slack
                slack_response = self.slack_client.chat_postMessage(
                    channel=channel,
                    text=slack_message,
                    thread_ts=thread_ts
                )
                
                return f"✅ AI response sent to #{channel} successfully! Timestamp: {slack_response['ts']}"
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        @self.mcp.tool
        def analyze_slack_conversation(channel: str, limit: int = 50, analysis_type: str = "summary") -> str:
            """
            Analyze a Slack conversation using AI
            
            Args:
                channel: The Slack channel ID or name to analyze
                limit: Number of recent messages to analyze (max 100)
                analysis_type: Type of analysis (summary, sentiment, action_items, key_topics)
            
            Returns:
                AI analysis of the conversation
            """
            if not self.slack_client:
                return "❌ Slack client not initialized. Please check your SLACK_BOT_TOKEN."
            
            if not self.openai_client:
                return "❌ OpenAI client not initialized. Please check your OPENAI_API_KEY."
            
            try:
                # Get recent messages from the channel
                if channel.startswith('#'):
                    channel = channel[1:]
                
                # Get channel ID if name was provided
                if not channel.startswith('C'):
                    channels_response = self.slack_client.conversations_list()
                    channel_id = None
                    for ch in channels_response['channels']:
                        if ch['name'] == channel:
                            channel_id = ch['id']
                            break
                    
                    if not channel_id:
                        return f"❌ Channel '{channel}' not found"
                    channel = channel_id
                
                # Get messages
                response = self.slack_client.conversations_history(
                    channel=channel,
                    limit=min(limit, 100)
                )
                
                if not response['messages']:
                    return "❌ No messages found in the channel"
                
                # Format messages for AI analysis
                conversation_text = ""
                for message in reversed(response['messages']):  # Reverse to get chronological order
                    if message.get('text'):
                        user_name = message.get('user', 'Unknown')
                        try:
                            if message.get('user'):
                                user_info = self.slack_client.users_info(user=message['user'])
                                user_name = user_info['user']['real_name'] or user_info['user']['name']
                        except:
                            pass
                        
                        timestamp = datetime.fromtimestamp(float(message.get('ts', 0))).strftime('%Y-%m-%d %H:%M')
                        conversation_text += f"[{timestamp}] {user_name}: {message['text']}\n"
                
                # Prepare AI prompt based on analysis type
                analysis_prompts = {
                    "summary": "Please provide a concise summary of this Slack conversation, highlighting the main topics discussed and key points made.",
                    "sentiment": "Analyze the sentiment and tone of this Slack conversation. Identify the overall mood and any emotional patterns.",
                    "action_items": "Extract any action items, tasks, decisions, or follow-ups mentioned in this Slack conversation.",
                    "key_topics": "Identify and categorize the key topics and themes discussed in this Slack conversation."
                }
                
                prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])
                
                # Get AI analysis
                ai_response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": f"You are analyzing a Slack team conversation. {prompt}"},
                        {"role": "user", "content": f"Here's the conversation:\n\n{conversation_text}"}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                analysis = ai_response.choices[0].message.content
                
                return json.dumps({
                    'success': True,
                    'channel': channel,
                    'analysis_type': analysis_type,
                    'message_count': len(response['messages']),
                    'analysis': analysis
                }, indent=2)
                
            except SlackApiError as e:
                return f"❌ Slack API Error: {e.response['error']}"
            except Exception as e:
                return f"❌ Error analyzing conversation: {str(e)}"
        
        @self.mcp.tool
        def add_mcp_server(server_name: str, url_or_command: str, description: Optional[str] = None) -> str:
            """
            Add a new MCP server connection
            
            Args:
                server_name: Unique name for the MCP server
                url_or_command: Server URL (for HTTP) or command (for stdio)
                description: Optional description of the server
            
            Returns:
                Success message or error
            """
            try:
                if server_name in self.mcp_server_configs:
                    return f"❌ MCP server '{server_name}' already exists. Use update_mcp_server to modify."
                
                config = {
                    "url": url_or_command if url_or_command.startswith(("http://", "https://")) else None,
                    "command": url_or_command if not url_or_command.startswith(("http://", "https://")) else None,
                    "description": description or f"MCP server: {server_name}"
                }
                
                # Clean up None values
                config = {k: v for k, v in config.items() if v is not None}
                
                self.mcp_server_configs[server_name] = config
                
                return json.dumps({
                    'success': True,
                    'message': f"✅ MCP server '{server_name}' added successfully",
                    'server_name': server_name,
                    'config': config,
                    'note': "Server will be available for connection. Use connect_mcp_server to establish connection."
                }, indent=2)
                
            except Exception as e:
                return f"❌ Error adding MCP server: {str(e)}"
        
        @self.mcp.tool
        def list_mcp_servers() -> str:
            """
            List all configured MCP servers and their connection status
            
            Returns:
                JSON with server configurations and status
            """
            try:
                servers_info = []
                for server_name, config in self.mcp_server_configs.items():
                    is_connected = server_name in self.mcp_clients
                    server_info = {
                        'name': server_name,
                        'config': config,
                        'connected': is_connected,
                        'status': '🟢 Connected' if is_connected else '🔴 Disconnected'
                    }
                    servers_info.append(server_info)
                
                return json.dumps({
                    'success': True,
                    'total_servers': len(self.mcp_server_configs),
                    'connected_servers': len(self.mcp_clients),
                    'servers': servers_info
                }, indent=2)
                
            except Exception as e:
                return f"❌ Error listing MCP servers: {str(e)}"
        
        @self.mcp.tool
        async def connect_mcp_server(server_name: str) -> str:
            """
            Connect to a configured MCP server
            
            Args:
                server_name: Name of the MCP server to connect to
            
            Returns:
                Connection status message
            """
            try:
                if server_name not in self.mcp_server_configs:
                    return f"❌ MCP server '{server_name}' not found. Use add_mcp_server first."
                
                if server_name in self.mcp_clients:
                    return f"✅ Already connected to MCP server '{server_name}'"
                
                config = self.mcp_server_configs[server_name]
                success = await self._connect_to_mcp_server(server_name, config)
                
                if success:
                    # Get available tools from the server
                    try:
                        tools = await self.mcp_clients[server_name].list_tools()
                        tool_names = [tool.name for tool in tools] if tools else []
                        
                        return json.dumps({
                            'success': True,
                            'message': f"✅ Successfully connected to MCP server '{server_name}'",
                            'server_name': server_name,
                            'available_tools': tool_names,
                            'tool_count': len(tool_names)
                        }, indent=2)
                    except Exception as e:
                        return f"✅ Connected to '{server_name}' but couldn't list tools: {str(e)}"
                else:
                    return f"❌ Failed to connect to MCP server '{server_name}'"
                
            except Exception as e:
                return f"❌ Error connecting to MCP server: {str(e)}"
        
        @self.mcp.tool
        async def disconnect_mcp_server(server_name: str) -> str:
            """
            Disconnect from an MCP server
            
            Args:
                server_name: Name of the MCP server to disconnect from
            
            Returns:
                Disconnection status message
            """
            try:
                if server_name not in self.mcp_clients:
                    return f"❌ Not connected to MCP server '{server_name}'"
                
                await self._disconnect_mcp_server(server_name)
                return f"✅ Disconnected from MCP server '{server_name}'"
                
            except Exception as e:
                return f"❌ Error disconnecting from MCP server: {str(e)}"
        
        @self.mcp.tool
        async def list_external_tools() -> str:
            """
            List all available tools from connected MCP servers
            
            Returns:
                JSON with all external tools and their descriptions
            """
            try:
                all_tools = await self._get_all_external_tools()
                
                if not all_tools:
                    return json.dumps({
                        'success': True,
                        'message': "No external MCP tools available",
                        'connected_servers': len(self.mcp_clients),
                        'total_tools': 0,
                        'tools': []
                    }, indent=2)
                
                # Group tools by server
                tools_by_server = {}
                for tool in all_tools:
                    server = tool['server']
                    if server not in tools_by_server:
                        tools_by_server[server] = []
                    tools_by_server[server].append({
                        'name': tool['name'],
                        'description': tool['description']
                    })
                
                return json.dumps({
                    'success': True,
                    'connected_servers': len(self.mcp_clients),
                    'total_tools': len(all_tools),
                    'tools_by_server': tools_by_server,
                    'all_tools': all_tools
                }, indent=2)
                
            except Exception as e:
                return f"❌ Error listing external tools: {str(e)}"
        
        @self.mcp.tool
        async def call_external_tool(server_name: str, tool_name: str, arguments: str = "{}") -> str:
            """
            Call a tool from an external MCP server
            
            Args:
                server_name: Name of the MCP server
                tool_name: Name of the tool to call
                arguments: JSON string with tool arguments
            
            Returns:
                Tool execution result
            """
            try:
                if server_name not in self.mcp_clients:
                    return f"❌ Not connected to MCP server '{server_name}'. Use connect_mcp_server first."
                
                # Parse arguments
                try:
                    args = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    return f"❌ Invalid arguments JSON format: {arguments}"
                
                client = self.mcp_clients[server_name]
                result = await client.call_tool(tool_name, args)
                
                return json.dumps({
                    'success': True,
                    'server': server_name,
                    'tool': tool_name,
                    'arguments': args,
                    'result': result
                }, indent=2)
                
            except Exception as e:
                return f"❌ Error calling external tool: {str(e)}"
        
        @self.mcp.tool
        async def ask_ai_with_external_tools(question: str, use_external_tools: bool = True, context: Optional[str] = None) -> str:
            """
            Ask AI a question with access to external MCP tools
            
            Args:
                question: The question to ask
                use_external_tools: Whether to include external tools in context
                context: Additional context
            
            Returns:
                AI response with information about available external tools
            """
            if not self.openai_client:
                return "❌ OpenAI client not initialized. Please check your OPENAI_API_KEY."
            
            try:
                # Prepare context with external tools if requested
                full_context = context or ""
                
                if use_external_tools and self.mcp_clients:
                    external_tools = await self._get_all_external_tools()
                    if external_tools:
                        tools_context = "\n\nAvailable External MCP Tools:\n"
                        for tool in external_tools:
                            tools_context += f"- {tool['server']}.{tool['name']}: {tool['description']}\n"
                        full_context += tools_context
                
                # Prepare messages
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant with access to various tools and services through MCP (Model Context Protocol). You can suggest using specific tools when relevant to the user's question."}
                ]
                
                if full_context:
                    messages.append({"role": "system", "content": f"Additional context and available tools: {full_context}"})
                
                messages.append({"role": "user", "content": question})
                
                # Get AI response
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                # Add info about available tools
                tools_info = ""
                if use_external_tools and self.mcp_clients:
                    tools_info = f"\n\n🔧 **Available External Tools**: {len(await self._get_all_external_tools())} tools from {len(self.mcp_clients)} connected MCP servers"
                
                return f"🤖 **AI Response**:\n\n{ai_response}{tools_info}"
                
            except Exception as e:
                return f"❌ Error getting AI response: {str(e)}"
        
        @self.mcp.tool
        def get_trello_boards() -> str:
            """
            Get all Trello boards for the authenticated user
            
            Returns:
                JSON with board information
            """
            if not self.trello_api_key or not self.trello_token:
                return "❌ Trello API credentials not configured. Please set TRELLO_API_KEY and TRELLO_TOKEN."
            
            try:
                url = "https://api.trello.com/1/members/me/boards"
                query = {
                    'key': self.trello_api_key,
                    'token': self.trello_token
                }
                
                response = requests.get(url, params=query)
                response.raise_for_status()
                
                boards = response.json()
                board_info = []
                
                for board in boards:
                    board_info.append({
                        'id': board['id'],
                        'name': board['name'],
                        'url': board['url'],
                        'closed': board['closed'],
                        'description': board.get('desc', ''),
                        'organization': board.get('idOrganization')
                    })
                
                return json.dumps({
                    'success': True,
                    'total_boards': len(board_info),
                    'boards': board_info
                }, indent=2)
                
            except requests.RequestException as e:
                return f"❌ Trello API Error: {str(e)}"
            except Exception as e:
                return f"❌ Error getting Trello boards: {str(e)}"
        
        @self.mcp.tool
        def get_trello_board_lists(board_id: str) -> str:
            """
            Get all lists from a specific Trello board
            
            Args:
                board_id: The Trello board ID
            
            Returns:
                JSON with list information
            """
            if not self.trello_api_key or not self.trello_token:
                return "❌ Trello API credentials not configured."
            
            try:
                url = f"https://api.trello.com/1/boards/{board_id}/lists"
                query = {
                    'key': self.trello_api_key,
                    'token': self.trello_token
                }
                
                response = requests.get(url, params=query)
                response.raise_for_status()
                
                lists = response.json()
                list_info = []
                
                for lst in lists:
                    list_info.append({
                        'id': lst['id'],
                        'name': lst['name'],
                        'closed': lst['closed'],
                        'position': lst['pos']
                    })
                
                return json.dumps({
                    'success': True,
                    'board_id': board_id,
                    'total_lists': len(list_info),
                    'lists': list_info
                }, indent=2)
                
            except requests.RequestException as e:
                return f"❌ Trello API Error: {str(e)}"
            except Exception as e:
                return f"❌ Error getting board lists: {str(e)}"
        
        @self.mcp.tool
        def get_trello_list_cards(list_id: str) -> str:
            """
            Get all cards from a specific Trello list
            
            Args:
                list_id: The Trello list ID
            
            Returns:
                JSON with card information
            """
            if not self.trello_api_key or not self.trello_token:
                return "❌ Trello API credentials not configured."
            
            try:
                url = f"https://api.trello.com/1/lists/{list_id}/cards"
                query = {
                    'key': self.trello_api_key,
                    'token': self.trello_token
                }
                
                response = requests.get(url, params=query)
                response.raise_for_status()
                
                cards = response.json()
                card_info = []
                
                for card in cards:
                    card_info.append({
                        'id': card['id'],
                        'name': card['name'],
                        'description': card.get('desc', ''),
                        'url': card['url'],
                        'closed': card['closed'],
                        'due_date': card.get('due'),
                        'position': card['pos'],
                        'labels': [label['name'] for label in card.get('labels', [])]
                    })
                
                return json.dumps({
                    'success': True,
                    'list_id': list_id,
                    'total_cards': len(card_info),
                    'cards': card_info
                }, indent=2)
                
            except requests.RequestException as e:
                return f"❌ Trello API Error: {str(e)}"
            except Exception as e:
                return f"❌ Error getting list cards: {str(e)}"
        
        @self.mcp.tool
        def create_trello_card(list_id: str, name: str, description: Optional[str] = None, due_date: Optional[str] = None) -> str:
            """
            Create a new card in a Trello list
            
            Args:
                list_id: The Trello list ID where to create the card
                name: The card name/title
                description: Optional card description
                due_date: Optional due date (YYYY-MM-DD format)
            
            Returns:
                JSON with created card information
            """
            if not self.trello_api_key or not self.trello_token:
                return "❌ Trello API credentials not configured."
            
            try:
                url = "https://api.trello.com/1/cards"
                query = {
                    'key': self.trello_api_key,
                    'token': self.trello_token,
                    'idList': list_id,
                    'name': name
                }
                
                if description:
                    query['desc'] = description
                
                if due_date:
                    query['due'] = due_date
                
                response = requests.post(url, params=query)
                response.raise_for_status()
                
                card = response.json()
                
                return json.dumps({
                    'success': True,
                    'message': f"✅ Card '{name}' created successfully",
                    'card': {
                        'id': card['id'],
                        'name': card['name'],
                        'url': card['url'],
                        'description': card.get('desc', ''),
                        'due_date': card.get('due')
                    }
                }, indent=2)
                
            except requests.RequestException as e:
                return f"❌ Trello API Error: {str(e)}"
            except Exception as e:
                return f"❌ Error creating Trello card: {str(e)}"
        
        @self.mcp.tool
        def search_trello_cards(query: str, limit: int = 10) -> str:
            """
            Search for Trello cards across all boards
            
            Args:
                query: Search query string
                limit: Maximum number of results (max 1000)
            
            Returns:
                JSON with search results
            """
            if not self.trello_api_key or not self.trello_token:
                return "❌ Trello API credentials not configured."
            
            try:
                url = "https://api.trello.com/1/search"
                params = {
                    'key': self.trello_api_key,
                    'token': self.trello_token,
                    'query': query,
                    'modelTypes': 'cards',
                    'cards_limit': min(limit, 1000)
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                results = response.json()
                cards = results.get('cards', [])
                
                card_info = []
                for card in cards:
                    card_info.append({
                        'id': card['id'],
                        'name': card['name'],
                        'description': card.get('desc', ''),
                        'url': card['url'],
                        'board_name': card.get('board', {}).get('name', ''),
                        'list_name': card.get('list', {}).get('name', ''),
                        'due_date': card.get('due'),
                        'labels': [label['name'] for label in card.get('labels', [])]
                    })
                
                return json.dumps({
                    'success': True,
                    'query': query,
                    'total_results': len(card_info),
                    'cards': card_info
                }, indent=2)
                
            except requests.RequestException as e:
                return f"❌ Trello API Error: {str(e)}"
            except Exception as e:
                return f"❌ Error searching Trello cards: {str(e)}"
    
    def run(self, **kwargs):
        """Run the FastMCP server"""
        self.mcp.run(**kwargs)

# Create and run the server
if __name__ == "__main__":
    server = SlackMCPServer()
    server.run()