import yaml
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tools import ddg_search
from settings import secret_path
from utils import graph_to_png

memory = MemorySaver()

tools = [ddg_search]
tool_node = ToolNode(tools)

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tool"


def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tool", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
workflow.add_edge("tool", "agent")
app = workflow.compile(checkpointer=memory)
# graph_to_png(app, "09_graph.png")

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": 1}}

input_message = HumanMessage(content="안녕 내이름은 김선진이야~")
for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
print("*" * 50)

input_message = HumanMessage(content="내이름이 뭐라고?")
for event in app.stream({"messages": [input_message]}, config=config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
print("*" * 50)

messages = app.get_state(config).values["messages"]
for message in messages:
    message.pretty_print()
print("*" * 50)

from langchain_core.messages import RemoveMessage

app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})

messages = app.get_state(config).values["messages"]
for message in messages:
    message.pretty_print()
print("*" * 50)

app.update_state(config, {"messages": RemoveMessage(id=messages[1].id)})

messages = app.get_state(config).values["messages"]
for message in messages:
    message.pretty_print()
