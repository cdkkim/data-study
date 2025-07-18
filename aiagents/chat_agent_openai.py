'''
https://openai.github.io/openai-agents-python/agents/
'''
import re
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_date
from agents import Agent, Runner, WebSearchTool, function_tool, ModelSettings


@dataclass
class ChatContext:
    """Context object to hold user session data and dependencies."""
    user_name: str = "User"
    session_id: str = "default"
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


@function_tool
def current_date(query: str) -> str:
    """Return the date that matches the query in YYYY-MM-DD format.

    Understands: today, tomorrow, yesterday, ISO/US dates, and
    relative expressions like '1 year ago', 'in 3 weeks', '2 months ago'.

    Args:
        query: The date query to parse (e.g., "today", "2 weeks ago", "2024-01-15")
    """
    q = query.strip().lower()
    now = datetime.now()

    # Simple keywords
    if q in {"today", "current date", "date"}:
        return now.strftime("%Y-%m-%d")
    if q == "tomorrow":
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if q == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Relative "in 2 weeks" / "3 months ago"
    m = re.match(r"^\s*(?:in\s+)?(\d+)\s+(day|week|month|year)s?\s*(ago)?\s*$", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        ago = m.group(3) is not None

        delta_kwargs = {
            "days": n if unit == "day" else 0,
            "weeks": n if unit == "week" else 0,
            "months": n if unit == "month" else 0,
            "years": n if unit == "year" else 0,
        }
        dt = now - relativedelta(**delta_kwargs) if ago else now + relativedelta(**delta_kwargs)
        return dt.strftime("%Y-%m-%d")

    # Fallback to free-form parsing
    try:
        dt = parse_date(query, default=now)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        return f"Could not parse date from query: {query!r}. Error: {str(e)}"


class ChatAgent:
    """Main chat agent using OpenAI Agents framework."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        """Initialize the chat agent with configuration."""

        # Create model settings
        model_settings = ModelSettings(
            temperature=temperature,
            max_tokens=1000
        )

        # Create the main agent
        self.agent = Agent[ChatContext](
            name="Assistant",
            instructions=(
                "You are a helpful AI assistant with access to date calculation "
                "and web search tools. Use the tools when appropriate to provide "
                "accurate and up-to-date information.\n\n"
                "Guidelines:\n"
                "- Be conversational and friendly\n"
                "- Use the current_date tool for any date-related queries\n"
                "- Use web search for current events or information you need to look up\n"
                "- Provide clear, helpful responses\n"
                "- If you use tools, explain what you found"
            ),
            model=model,
            model_settings=model_settings,
            tools=[
                current_date,
                WebSearchTool()
            ]
        )

    async def chat(self, message: str, context: Optional[ChatContext] = None) -> str:
        """Send a message to the agent and return the response."""
        if context is None:
            context = ChatContext()

        try:
            # Add user message to context history
            context.conversation_history.append({"role": "user", "content": message})

            # Run the agent
            result = await Runner.run(
                self.agent,
                context=context,
                input=message,
                max_turns=10
            )

            # Add assistant response to context history
            context.conversation_history.append({"role": "assistant", "content": result.final_output})

            return result.final_output

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            context.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def create_specialized_agent(self, name: str, instructions: str, tools: list = None) -> Agent:
        """Create a specialized agent for specific tasks."""
        if tools is None:
            tools = [current_date]  # Default to date tool

        return Agent[ChatContext](
            name=name,
            instructions=instructions,
            model="gpt-4o",
            tools=tools
        )


class MultiAgentChat:
    """Example of multi-agent orchestration with handoffs."""

    def __init__(self):
        # Create specialized agents
        self.date_agent = Agent[ChatContext](
            name="Date Agent",
            instructions=(
                "You are a date calculation specialist. Help users with any "
                "date-related queries, calculations, or conversions. Always use "
                "the current_date tool for accurate results."
            ),
            tools=[current_date]
        )

        self.search_agent = Agent[ChatContext](
            name="Search Agent",
            instructions=(
                "You are a web search specialist. Help users find current "
                "information, news, and facts from the web. Always use web "
                "search to provide up-to-date information."
            ),
            tools=[WebSearchTool(max_results=3)]
        )

        # Create orchestrator agent with handoffs
        self.orchestrator = Agent[ChatContext](
            name="Orchestrator",
            instructions=(
                "You are a helpful assistant that coordinates with specialist agents. "
                "- For date-related queries, handoff to the Date Agent\n"
                "- For web search queries, handoff to the Search Agent\n"
                "- For simple questions, answer directly\n"
                "- Always be helpful and conversational"
            ),
            handoffs=[self.date_agent, self.search_agent]
        )

    async def chat(self, message: str, context: Optional[ChatContext] = None) -> str:
        """Chat using the orchestrator with specialized agents."""
        if context is None:
            context = ChatContext()

        try:
            result = await Runner.run(
                self.orchestrator,
                context=context,
                input=message,
                max_turns=15
            )

            return result.final_output

        except Exception as e:
            return f"Error processing message: {str(e)}"


async def interactive_chat():
    """Interactive chat loop for single agent."""
    print("🤖 Chat Agent (OpenAI Agents SDK)")
    print("Commands: 'quit'/'exit'/'q' to end, 'reset' to clear context")
    print("-" * 60)

    agent = ChatAgent()
    context = ChatContext()

    while True:
        try:
            user_input = input(f"\n{context.user_name}: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            if user_input.lower() == "reset":
                context = ChatContext()
                print("🔄 Context reset.")
                continue

            if not user_input:
                print("Please enter a message.")
                continue

            # Show thinking indicator
            print("🤔 Thinking...")

            # Get response
            response = await agent.chat(user_input, context)
            print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


async def multi_agent_chat():
    """Interactive chat loop for multi-agent system."""
    print("🤖🔀 Multi-Agent Chat System")
    print("The orchestrator will route your queries to specialized agents")
    print("Commands: 'quit'/'exit'/'q' to end")
    print("-" * 60)

    multi_agent = MultiAgentChat()
    context = ChatContext()

    while True:
        try:
            user_input = input(f"\n{context.user_name}: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            if not user_input:
                print("Please enter a message.")
                continue

            print("🤔 Routing to appropriate agent...")

            response = await multi_agent.chat(user_input, context)
            print(f"\n🤖 System: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


async def main():
    """Main function to choose between single agent and multi-agent chat."""
    print("Choose chat mode:")
    print("1. Single Agent (default)")
    print("2. Multi-Agent with handoffs")

    try:
        choice = input("Enter choice (1-2): ").strip()

        if choice == "2":
            await multi_agent_chat()
        else:
            await interactive_chat()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())