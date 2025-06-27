# ⚡ Quick Start Guide - Autonomous AI Slack Assistant

## 🎯 What You're Building

An **autonomous AI system** that:
- 🧠 **Thinks and plans** its own execution strategies
- 📱 **Manages Slack workspaces** intelligently  
- 🌐 **Researches web information** automatically
- 🔄 **Combines multiple data sources** for comprehensive answers

## 🚀 30-Second Setup

### 1. Install & Configure

```bash
# Clone and install
git clone <repo-url>
cd Slack_MCP_Server-main
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### 2. Add Your API Keys

Edit `.env`:
```bash
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token    # From slack.com/api/apps
OPENAI_API_KEY=sk-your-openai-api-key        # From platform.openai.com
TAVILY_API_KEY=tvly-your-tavily-api-key      # From tavily.com (optional)
```

### 3. Start the System

```bash
# Terminal 1: Web Search Server
python tavily_mcp_server.py

# Terminal 2: Main AI Server  
python main.py
```

**✅ Success Output:**
```
✅ Slack client initialized
✅ OpenAI client initialized  
🚀 Starting Slack MCP Server
INFO: Uvicorn running on http://0.0.0.0:8001
```

## 🎮 Instant Usage

### Claude Desktop Integration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "autonomous-ai": {
      "command": "python",
      "args": ["/path/to/main.py"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-...",
        "OPENAI_API_KEY": "sk-...",
        "TAVILY_API_KEY": "tvly-..."
      }
    }
  }
}
```

### Try These Commands

**Simple Message:**
```
send_slack_message(channel="general", message="Hello AI! 👋")
```

**Autonomous Intelligence:**
```
autonomous_slack_agent(
  user_request="Summarize #engineering discussions and research mentioned technologies",
  channel="engineering", 
  send_to_slack=True
)
```

**What Happens:**
1. 🔍 AI discovers available tools (Slack + Web search)
2. 🧠 AI creates execution plan: "Get Slack data → Research topics → Synthesize"  
3. ⚡ AI executes plan automatically
4. 📨 AI delivers comprehensive summary to Slack

## 🧠 The Magic: How AI "Thinks"

### Traditional Chatbot (Hard-coded):
```python
if "summary" in message:
    get_slack_messages()
if "research" in message:  
    search_web()
```

### Our Autonomous AI:
```python
# AI creates its own plan:
{
  "reasoning": "User wants meeting summary + tech research",
  "steps": [
    {"tool": "get_slack_messages", "purpose": "Get conversation context"},
    {"tool": "search_web", "query": "React 18 features"},
    {"tool": "synthesize", "purpose": "Combine findings"}
  ]
}
```

**The AI literally plans its own execution strategy!** 🤯

## 🔧 Available Tools

### 📱 Slack Management (6 tools)
- `send_slack_message` - Send messages
- `get_slack_channels` - List channels  
- `get_slack_messages` - Get chat history
- `search_slack_messages` - Search workspace
- `create_slack_channel` - Create channels
- `analyze_slack_conversation` - AI analysis

### 🤖 AI Intelligence (2 tools)
- `ask_ai_question` - Simple AI queries
- `autonomous_slack_agent` - **🌟 Full autonomy**

### 🌐 Web Research (3 tools)  
- `search_web` - General web search
- `search_news` - News search
- `research_topic` - Deep research

## 💡 Pro Usage Examples

### Meeting Summary + Tech Research
```
"Analyze yesterday's #product-meeting channel and research current trends for any technologies discussed. Create a strategic report."
```

### Multi-Channel Intelligence  
```
"Compare discussions across #design, #engineering, and #marketing. What are the common themes and blockers?"
```

### Real-Time Research
```
"What are the latest developments in AI since our last team discussion about GPT models?"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   SLACK MCP     │    │   TAVILY MCP    │  
│   (Port 8001)   │    │   (Port 8002)   │
│                 │    │                 │
│ • Slack Tools   │    │ • Web Search    │
│ • AI Tools      │    │ • News Search   │  
│ • Orchestration │    │ • Research      │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┐   ┌───────┘
                     │   │
              ┌─────────────────┐
              │ AUTONOMOUS      │
              │ AGENT           │
              │                 │  
              │ 🧠 Thinks       │
              │ 📋 Plans        │
              │ ⚡ Executes     │
              │ 🔬 Synthesizes  │
              └─────────────────┘
```

## 🚨 Troubleshooting

**Server won't start:**
```bash
# Check ports
lsof -i :8001
lsof -i :8002

# Test API keys  
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

**Slack connection fails:**
```bash
# Verify bot token
curl -H "Authorization: Bearer xoxb-..." https://slack.com/api/auth.test
```

**Tools not showing in Claude:**
- Use absolute paths in config
- Restart Claude Desktop
- Check environment variables

## 🎉 You're Ready!

Your AI system is now:
- ✅ **Thinking autonomously**
- ✅ **Connected to Slack**  
- ✅ **Researching the web**
- ✅ **Available in Claude Desktop**

**Try the autonomous agent and watch it plan its own execution!** 🚀

---

**Next Steps:**
- Read the [Complete Guide](COMPLETE_GUIDE.md) for advanced features
- Experiment with complex multi-step requests
- Add your own MCP servers to extend capabilities