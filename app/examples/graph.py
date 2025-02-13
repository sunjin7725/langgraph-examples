import yaml
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from langchain_tools import ddg_search_results, ddg_search

from settings import secret_path
from utils import graph_to_png


#### 1.상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]


#### 2. 도구 정의 및 바인딩
tools = [ddg_search]

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)


#### 3. 노드 추가
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)

#### 4. 엣지 추가
# tools -> chatbot
graph_builder.add_edge("tools", "chatbot")

# START -> chatbot
graph_builder.add_edge(START, "chatbot")

# chatbot -> END
graph_builder.add_edge("chatbot", END)

#### 5. 메모리 추가
memory = MemorySaver()

#### 6. 그래프 컴파일
graph = graph_builder.compile(checkpointer=memory)
# graph_to_png(graph, "graph.png")
