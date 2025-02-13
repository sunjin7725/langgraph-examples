import yaml

from typing import Annotated, TypedDict
from langchain_tools import ddg_search

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from settings import secret_path
from utils import graph_to_png


class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool


class HumanRequest(BaseModel):
    """
    Forward the conversation to an expert. Use when you can't assist directly or the user needs assistance that exceeds your authority.
    To use this function, pass the user's 'request' so that an expert can provide appropriate guidance.
    """

    request: str


tools = [ddg_search, HumanRequest]

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])

    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == HumanRequest.__name__:
        ask_human = True

    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[ddg_search]))


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(content=response, tool_call_id=ai_message.tool_calls[0]["id"])


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(create_response("No response from humna.", state["messages"][-1]))

    return {"messages": new_messages, "ask_human": False}


graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


graph_builder.add_conditional_edges("chatbot", select_next_node, {"human": "human", "tools": "tools", END: END})

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])

graph_to_png(graph, "08_graph.png")

user_input = "이 AI 에이전트를 구축하기 위해 전문가의 도움이 필요합니다. 도움을 요청할 수 있나요?"
config = {"configurable": {"thread_id": 1}}

events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot.next)

ai_message = snapshot.values["messages"][-1]

human_response = (
    "전문가들이 도와드리겠습니다! 에이전트 구축을 위해 LangGraph를 확인해 보시기를 적극 추천드립니다. "
    "단순한 자율 에이전트보다 훨씬 더 안정적이고 확장성이 뛰어납니다. "
    "https://wikidocs.net/233785 에서 더 많은 정보를 확인할 수 있습니다."
)

tool_message = create_response(human_response, ai_message)
graph.update_state(config, {"messages": [tool_message]})

# print(graph.get_state(config).values["messages"])

events = graph.stream(None, config=config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

state = graph.get_state(config)

# 단계별 메시지 출력
for message in state.values["messages"]:
    message.pretty_print()
