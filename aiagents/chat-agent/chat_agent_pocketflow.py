'''
https://github.com/The-Pocket
'''
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_date
from pocketflow import Node, Flow
import openai
import requests
import json
import yaml


class DateToolNode(Node):
    """Node for handling date-related queries."""

    def prep(self, shared):
        """Extract date query from the user message."""
        messages = shared.get('messages', [])
        if not messages:
            return ""

        user_message = messages[-1]['content']
        print(f"🗓️  Processing date query: {user_message}")
        return user_message

    def exec(self, query):
        """Convert common date queries into an ISO string (YYYY-MM-DD)."""
        q = query.strip().lower()
        now = datetime.now()

        # Simple keywords - Fixed the logic for "what is today"
        if any(keyword in q for keyword in ["today", "current date", "date", "what is today"]):
            return now.strftime("%Y-%m-%d")
        if "tomorrow" in q:
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if "yesterday" in q:
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")

        # Relative "in 2 weeks" / "3 months ago"
        m = re.match(r"^\s*(?:in\s+)?(\d+)\s+(day|week|month|year)s?\s*(?:ago)?\s*$", q)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            ago = "ago" in q
            delta_kwargs = {
                "days": n if unit == "day" else 0,
                "weeks": n if unit == "week" else 0,
                "months": n if unit == "month" else 0,
                "years": n if unit == "year" else 0,
            }
            dt = now - relativedelta(**delta_kwargs) if ago else now + relativedelta(**delta_kwargs)
            return dt.strftime("%Y-%m-%d")

        # Fallback to free-form parsing - but be more lenient
        try:
            dt = parse_date(query, default=now)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # If all parsing fails, just return today's date for date-related queries
            print(f"⚠️  Could not parse specific date from '{query}', returning today's date")
            return now.strftime("%Y-%m-%d")

    def post(self, shared, prep_res, exec_res):
        """Store the date result and proceed to next step."""
        shared['tool_results'] = shared.get('tool_results', [])
        shared['tool_results'].append(f"Current date: {exec_res}")
        print(f"📅 Date result: {exec_res}")

        # Check if we need both tools
        decision = shared.get('decision', '')
        if decision == "both":
            return "search"  # Go to search tool next
        else:
            return "llm"  # Go directly to LLM


class SearchToolNode(Node):
    """Node for handling search queries."""

    def prep(self, shared):
        """Extract search query from the user message."""
        messages = shared.get('messages', [])
        if not messages:
            return ""

        user_message = messages[-1]['content']
        print(f"🔍 Processing search query: {user_message}")
        return user_message

    def exec(self, query):
        """Search the web using a simple search API (placeholder)."""
        # This is a placeholder - you would implement actual search here
        # You could use DuckDuckGo, SerpAPI, Tavily, or any other search service
        try:
            # Example with a simple web search (replace with actual implementation)
            # For demo purposes, returning a placeholder
            print("🌐 Performing web search...")
            return f"Search results for '{query}': [Placeholder - implement with your preferred search API like Tavily, SerpAPI, or DuckDuckGo]"
        except Exception as e:
            return f"Search failed: {str(e)}"

    def post(self, shared, prep_res, exec_res):
        """Store the search result and proceed to LLM."""
        shared['tool_results'] = shared.get('tool_results', [])
        shared['tool_results'].append(f"Search results: {exec_res}")
        print(f"🔎 Search complete")
        return "llm"


class DecisionNode(Node):
    """Node that decides which tools to use based on the user query."""

    def prep(self, shared):
        """Prepare the user message for analysis."""
        messages = shared.get('messages', [])
        if not messages:
            return ""

        user_message = messages[-1]['content']
        print(f"🤔 Analyzing query: {user_message}")
        return user_message

    def exec(self, user_message):
        """Analyze the query and decide which tools to use."""
        q = user_message.lower()

        # Check for date-related keywords
        date_keywords = ['date', 'today', 'tomorrow', 'yesterday', 'ago', 'week', 'month', 'year', 'when']
        needs_date = any(keyword in q for keyword in date_keywords)

        # Check for search-related keywords
        search_keywords = ['search', 'find', 'what is', 'who is', 'latest', 'news', 'current', 'recent']
        needs_search = any(keyword in q for keyword in search_keywords)

        if needs_date and needs_search:
            return "both"
        elif needs_date:
            return "date"
        elif needs_search:
            return "search"
        else:
            return "llm"

    def post(self, shared, prep_res, exec_res):
        """Store the decision and route to appropriate tool."""
        shared['decision'] = exec_res
        print(f"🎯 Decision: {exec_res}")
        return exec_res


class LLMNode(Node):
    """Node for handling LLM interactions."""

    def prep(self, shared):
        """Prepare the context for the LLM call."""
        messages = shared.get('messages', [])
        tool_results = shared.get('tool_results', [])

        if not messages:
            return "", ""

        user_message = messages[-1]['content']
        context = ""

        if tool_results:
            context = f"Tool results: {'; '.join(tool_results)}\n\n"

        print(f"🤖 Calling LLM with context length: {len(context)}")
        return user_message, context

    def exec(self, inputs):
        """Call the LLM to generate a response."""
        user_message, context = inputs

        system_message = """You are a helpful assistant. You have access to date and search tools.
        If tool results are provided, use them to answer the user's question accurately.
        Be conversational and helpful in your responses."""

        try:
            client = openai.OpenAI()  # Assumes OPENAI_API_KEY is set in environment
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{context}{user_message}"}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            assistant_message = response.choices[0].message.content
            return assistant_message

        except Exception as e:
            # Fallback response if OpenAI API fails
            if context and "Current date:" in context:
                date_info = [line for line in context.split(';') if 'Current date:' in line]
                if date_info:
                    current_date = date_info[0].split(': ')[1].strip()
                    return f"Today's date is {current_date}."

            return f"I apologize, but I encountered an error: {str(e)}"

    def post(self, shared, prep_res, exec_res):
        """Store the LLM response and complete the flow."""
        messages = shared.get('messages', [])
        messages.append({"role": "assistant", "content": exec_res})
        shared['messages'] = messages
        shared['last_response'] = exec_res

        # Clear tool results for next interaction
        shared['tool_results'] = []

        print(f"✅ Response generated")
        return "done"


def main():
    """Main chat loop using PocketFlow with separate tool nodes."""

    # Create nodes
    decision_node = DecisionNode()
    date_tool_node = DateToolNode()
    search_tool_node = SearchToolNode()
    llm_node = LLMNode()

    # Connect nodes using PocketFlow's conditional transitions
    # Decision node routes to different tools or directly to LLM
    decision_node - "date" >> date_tool_node
    decision_node - "search" >> search_tool_node
    decision_node - "llm" >> llm_node
    decision_node - "both" >> date_tool_node  # For queries needing both tools

    # Tools route to LLM after processing - with conditional routing for "both"
    date_tool_node - "llm" >> llm_node
    date_tool_node - "search" >> search_tool_node  # Added this transition for "both" case
    search_tool_node - "llm" >> llm_node

    # Create and configure the flow
    flow = Flow(start=decision_node)

    # Initialize conversation state
    shared_state = {
        'messages': [],
        'name': 'KO',
        'tool_results': []
    }

    print("🚀 Chatbot initialized with PocketFlow! Type 'quit', 'exit', or 'q' to end the conversation.")

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        # Add user message to shared state
        current_messages = shared_state.get('messages', [])
        current_messages.append({"role": "user", "content": user_input})
        shared_state['messages'] = current_messages

        # Clear previous tool results
        shared_state['tool_results'] = []

        # Run the flow
        try:
            flow.run(shared_state)

            # Display the response
            if 'last_response' in shared_state:
                print(f"Assistant: {shared_state['last_response']}")
            else:
                print("Assistant: I apologize, but I encountered an error processing your request.")

        except Exception as e:
            print(f"❌ Flow execution error: {str(e)}")
            # Try to provide a helpful fallback
            if "today" in user_input.lower():
                current_date = datetime.now().strftime("%Y-%m-%d")
                print(f"Assistant: Today's date is {current_date}.")


if __name__ == "__main__":
    main()