'''
https://github.com/agno-agi/agno
'''
import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_date
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools import tool


@tool
def current_date_tool(query: str) -> str:
    """
    Return the date that matches the query in YYYY-MM-DD format.
    Understands: today, tomorrow, yesterday, ISO/US dates, and
    relative expressions like '1 year ago', 'in 3 weeks', '2 months ago'.

    Args:
        query: Date query string to parse

    Returns:
        Date in YYYY-MM-DD format
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
        raise ValueError(f"Could not parse date from query: {query!r}") from e


class AgnoChat:
    """
    A chatbot implementation using the Agno framework.
    Provides date parsing tools and web search capabilities with persistent memory.
    """

    def __init__(self,
                 model_id: str = "gpt-4o",
                 user_name: str = "User",
                 session_id: Optional[str] = None):
        """
        Initialize the Agno chatbot.

        Args:
            model_id: OpenAI model to use
            user_name: Name of the user for personalization
            session_id: Session identifier for memory persistence
        """
        self.user_name = user_name
        self.session_id = session_id or "default_session"

        # Initialize the agent with tools and memory
        self.agent = Agent(
            name="Assistant",
            model=OpenAIChat(id=model_id),
            tools=[
                current_date_tool,
                DuckDuckGoTools()
            ],
            instructions=[
                f"You are a helpful assistant chatting with {user_name}.",
                "Use the current_date_tool when users ask about dates or need date calculations.",
                "Use web search when you need current information or facts you're unsure about.",
                "Be conversational and helpful.",
                "Always provide accurate information and cite sources when using web search."
            ],
            # memory=SqliteMemoryDb(
            #     table_name="chat_sessions",
            #     db_file=f"chat_memory_{self.session_id}.db"
            # ),
            show_tool_calls=True,
            markdown=True,
            debug_mode=False
        )

    def chat(self, message: str, stream: bool = True) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: User input message
            stream: Whether to stream the response

        Returns:
            Agent's response as string
        """
        if stream:
            # Stream the response for better user experience
            response = self.agent.run(message, stream=True)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            # Get complete response
            response = self.agent.run(message)
            return response.content if hasattr(response, 'content') else str(response)

    def print_response(self, message: str, stream: bool = True):
        """
        Print the agent's response to the console.

        Args:
            message: User input message
            stream: Whether to stream the response
        """
        self.agent.print_response(message, stream=stream)

    def clear_memory(self):
        """Clear the conversation memory for this session."""
        if self.agent.memory:
            self.agent.memory.clear()

    def get_conversation_history(self):
        """Get the conversation history for this session."""
        if self.agent.memory:
            return self.agent.memory.get_all()
        return []


def main():
    """
    Main function to run the interactive chatbot.
    """
    print("🤖 Agno Chatbot - Type 'quit', 'exit', or 'q' to stop")
    print("=" * 50)

    # Initialize the chatbot
    chatbot = AgnoChat(
        model_id="gpt-4o",
        user_name="User",
        session_id="interactive_session"
    )

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                print("Please enter a message.")
                continue

            # Special commands
            if user_input.lower() == "clear":
                chatbot.clear_memory()
                print("🧹 Memory cleared!")
                continue

            if user_input.lower() == "history":
                history = chatbot.get_conversation_history()
                if history:
                    print("\n📜 Conversation History:")
                    for entry in history[-5:]:  # Show last 5 entries
                        print(f"  {entry}")
                else:
                    print("No conversation history found.")
                continue

            print("\n🤖 Assistant:", end=" ")
            chatbot.print_response(user_input, stream=True)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()