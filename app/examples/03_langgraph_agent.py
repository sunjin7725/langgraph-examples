import yaml
import json

from typing import Annotated, TypedDict
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph, START, END

from settings import secret_path
from langchain_tools import ddg_search
from utils import graph_to_png

tools = [ddg_search]


class State(TypedDict):
    messages: Annotated[list, add_messages]


with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    answer = llm_with_tools.invoke(state["messages"])
    return {"messages": [answer]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)


class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_list = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in input")

        outputs = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_list[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=tools)

graph_builder.add_node("tools", tool_node)


def route_tools(state: State):
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


graph_builder.add_conditional_edges(source="chatbot", path=route_tools, path_map={"tools": "tools", END: END})

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# graph_to_png(graph=graph, output_file_path="03_graph.png")
question = "오케스트로가 뭐하는 회산지 설명좀 해봐봐"

for event in graph.stream({"messages": [("user", question)]}):
    for key, value in event.items():
        print(f"\n==============\nSTEP: {key}\n==============\n")
        print(value["messages"][-1])
