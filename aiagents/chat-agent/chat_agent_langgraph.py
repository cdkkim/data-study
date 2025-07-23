'''
https://www.langchain.com/langgraph
'''
import re
from datetime import datetime, timedelta
from typing import Annotated

from dateutil.relativedelta import relativedelta
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dateutil.parser import parse as parse_date
from typing_extensions import TypedDict


@tool(
    "current_date",
    description=(
            "Return the date that matches the query in YYYY-MM-DD format. "
            "Understands: today, tomorrow, yesterday, ISO/US dates, and "
            "relative expressions like '1 year ago', 'in 3 weeks', '2 months ago'."
    ),
)
def date_tool(query: str) -> str:
    """Convert common date queries into an ISO string (YYYY-MM-DD)."""
    q = query.strip().lower()
    now = datetime.now()

    # Simple keywords ---------------------------------------------------------
    if q in {"today", "current date", "date"}:
        return now.strftime("%Y-%m-%d")
    if q == "tomorrow":
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if q == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Relative “in 2 weeks” / “3 months ago” ----------------------------------
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

    # Fallback to free-form parsing -------------------------------------------
    try:
        dt = parse_date(query, default=now)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Could not parse date from query: {query!r}") from e


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str


graph_builder = StateGraph(State)

llm = init_chat_model("openai:gpt-4.1")
search_tool = TavilySearch(max_results=2)

tools = [search_tool, date_tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node('tools', tool_node)

graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,
)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {'configurable': {'thread_id': '1'}}

    def stream_graph_updates(user_input: str):
        events = graph.stream(
            {"name": "KO", "messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode='values',
        )

        for event in events:
            # print("Assistant:", event["messages"][-1].content)
            event['messages'][-1].pretty_print()


    while True:
        user_input = input("User: ")
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
