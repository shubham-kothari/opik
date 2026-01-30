"""
Simple Single-Agent Workflow with Opik Tracing

This demonstrates a basic agent workflow with:
- 1 Personal Assistant Agent
- 2 simple tools (Note Taker, Weather Checker)
- Full Opik tracing

Requirements:
- OPENROUTER_API_KEY environment variable
- Opik running locally
- python-dotenv (optional, for .env file support)

Run: python simple_agent.py
"""

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_FILE_LOADED = True
except ImportError:
    ENV_FILE_LOADED = False

import os
from typing import TypedDict, Annotated, Sequence
from operator import add
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from opik import configure
from opik.integrations.langchain import OpikTracer
from opik.guardrails import Guardrail, PointGuardAi
from opik import exceptions

# Configure Opik for local usage
configure(use_local=True)

# Validate required environment variables
if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is required.\n"
        "Please set it in your .env file or export it:\n"
        "  export OPENROUTER_API_KEY='your-api-key-here'\n"
        "Or create a .env file (see .env.example)"
    )

# Configure OpenRouter LLM
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-70b-instruct",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# Initialize PointGuard guardrails (optional)
guardrails_enabled = False
guard = None
policy_name = os.environ.get("POINTGUARDAI_POLICY_NAME")

if policy_name:
    try:
        guard = Guardrail(
            guards=[PointGuardAi(policy_name=policy_name)],
            guardrail_timeout=30
        )
        guardrails_enabled = True
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize PointGuard: {e}")
        print("   Continuing without guardrails...")

print("🚀 Simple Agent Workflow Starting...")
print("� Environment: {'.env loaded' if ENV_FILE_LOADED else 'system variables'}")
print("📊 Model: meta-llama/llama-3.1-70b-instruct")
print("🔍 Opik tracing: Enabled")
print("🔑 API Key: {'✓ Set' if os.environ.get('OPENROUTER_API_KEY') else '✗ Missing'}")
if guardrails_enabled:
    print("🛡️  PointGuard: Enabled (Policy: {policy_name})")
else:
    print("🛡️  PointGuard: Disabled")
    print("-" * 60)

# ============================================================================
# SIMPLE TOOLS
# ============================================================================

# In-memory note storage
notes = []

@tool
def save_note(content: str) -> str:
    """Save a note to memory. Use this to remember important information."""
    note_id = len(notes) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = {
        "id": note_id,
        "content": content,
        "timestamp": timestamp
    }
    notes.append(note)
    return f"✅ Note #{note_id} saved at {timestamp}: {content}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location. Returns mock weather data."""
    # Mock weather data
    weather_data = {
        "new york": "☀️ Sunny, 72°F (22°C), Light breeze",
        "london": "🌧️ Rainy, 55°F (13°C), Moderate wind",
        "tokyo": "⛅ Partly cloudy, 68°F (20°C), Calm",
        "paris": "☁️ Cloudy, 60°F (16°C), Light wind",
        "sydney": "🌤️ Mostly sunny, 78°F (26°C), Gentle breeze"
    }
    
    location_lower = location.lower()
    for city in weather_data:
        if city in location_lower:
            return f"Weather in {location}: {weather_data[city]}"
    
    # Default response for unknown locations
    return f"Weather in {location}: ⛅ Partly cloudy, 65°F (18°C), Light breeze (simulated)"

# Collect tools
tools = [save_note, get_weather]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Simple state for the agent."""
    messages: Annotated[Sequence[BaseMessage], add]
    user_input: str
    final_response: str

# ============================================================================
# AGENT NODE
# ============================================================================

def assistant_agent(state: AgentState):
    """Personal Assistant Agent: Handles user requests using available tools."""
    print("\n🤖 ASSISTANT: Processing request...")
    
    user_input = state["user_input"]
    
    # Generate unique correlation key for this request
    correlation_key = "Opik-test"
    
    # Validate input with PointGuard (if enabled)
    if guardrails_enabled and guard:
        try:
            user_input = guard.validate_and_get_input(user_input, correlation_key=correlation_key)
            print("   ✅ Input validation passed")
        except exceptions.GuardrailValidationFailed as e:
            print(f"   ❌ Input blocked: {e.failed_validations}")
            return {
                "messages": [],
                "final_response": f"⚠️ Your request was blocked by content policy: {e.failed_validations}"
            }
    
    system_prompt = """You are a helpful Personal Assistant with access to tools.
    
    Available tools:
    - save_note: Save important information to memory
    - get_weather: Check weather for any location
    
    When the user asks about weather, use get_weather.
    When the user wants to remember something, use save_note.
    Be friendly and helpful!"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    # Invoke with tools
    response = llm_with_tools.invoke(messages)
    
    # Check if tools were called
    if response.tool_calls:
        print("   🔧 Using {len(response.tool_calls)} tool(s)...")
        tool_results = tool_node.invoke({"messages": [response]})
        final_messages = messages + [response] + tool_results["messages"]
        final_response = llm.invoke(final_messages)
        print("   ✅ Response generated")
        response_content = final_response.content
    else:
        print("   ✅ Response generated")
        response_content = response.content
    
    # Validate output with PointGuard (if enabled)
    if guardrails_enabled and guard:
        try:
            response_content = guard.validate_and_get_output(
                user_input, response_content, correlation_key=correlation_key
            )
        except exceptions.GuardrailValidationFailed as e:
            print(f"   ❌ Output blocked: {e.failed_validations}")
            return {
                "messages": [response] if not response.tool_calls else [response, final_response],
                "final_response": f"⚠️ The response was blocked by content policy: {e.failed_validations}"
            }
    
    return {
        "messages": [response] if not response.tool_calls else [response, final_response],
        "final_response": response_content
    }

# ============================================================================
# WORKFLOW DEFINITION
# ============================================================================

# Build simple workflow
workflow = StateGraph(AgentState)
workflow.add_node("assistant", assistant_agent)
workflow.set_entry_point("assistant")
workflow.add_edge("assistant", END)

# Compile
app = workflow.compile()

# ============================================================================
# EXECUTION
# ============================================================================

def run_query(question: str, tracer: OpikTracer):
    """Run a single query through the workflow."""
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    
    initial_state = {
        "messages": [],
        "user_input": question,
        "final_response": ""
    }
    
    result = app.invoke(
        initial_state,
        config={"callbacks": [tracer]}
    )
    
    print(f"\n💬 Response: {result['final_response']}")
    print(f"{'='*60}")
    
    return result

if __name__ == "__main__":
    # Create Opik tracer
    tracer = OpikTracer(graph=app.get_graph(xray=True))
    
    print("\n🎬 Running test queries...\n")
    
    # Test 1: Weather query (uses get_weather tool)
    run_query("What's the weather like in New York? 469-12-4453", tracer)
    run_query("What's the weather like in New York? abc@gmail.com", tracer)
    run_query("What's the weather like in New York?", tracer)

    # # Test 2: Note-taking (uses save_note tool)
    # run_query("Remember that I have a meeting with Sarah at 3pm tomorrow", tracer)
    
    # # Test 3: Combined query (may use both tools)
    # run_query("Check Tokyo weather and save a note that I'm planning a trip there", tracer)
    
    # # Test 4: Simple query (no tools needed)
    # run_query("Tell me a fun fact about penguins", tracer)
    
    print("\n✅ All queries completed!")
    print("📊 Check your Opik dashboard for trace visualization")
    print("📝 Notes saved: {len(notes)}")
    if notes:
        print("\nSaved notes:")
        for note in notes:
            print(f"  - #{note['id']}: {note['content']}")
    print()
