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


memory = MemorySaver()


#### 1.상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]
    dummy_data: Annotated[str, "dummy"]


#### 2. 도구 정의 및 바인딩
tools = [ddg_search]

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

llm_with_tools = llm.bind_tools(tools)


#### 3. 노드 추가
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])], "dummy_data": "[chatbot] 호출, dummy data"}


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

#### 5. 그래프 컴파일
graph = graph_builder.compile(checkpointer=memory)
# graph_to_png(graph, "05_graph.png")

print(list(graph.channels.keys()))
# #### 6. RunnableConfig 설정
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "1"}, tags=["my-rag"])

question = "2024년 노벨 문학상 관련 뉴스를 알려주세요."

input = State(dummy_data="테스트 문자열", messages=[("user", question)])

# for event in graph.stream(input=input, config=config):
#     for key, value in event.items():
#         print(f"\n[ {key} ]\n")
#         if "messages" in value:
#             messages = value["messages"]
#             messages[-1].pretty_print()

for event in graph.stream(input=input, config=config, stream_mode="values"):
    # for key, value in event.items():
    #     print(f"\n[ {key} ]\n")
    # print(value["messages"][-1].content)
    # print(value.keys())
    # if "dummy_data" in value:
    #     print(value["dummy_data"])
    # if "messages" in value:
    #     messages = value["messages"]
    #     messages[-1].pretty_print()

    for key, value in event.items():
        # key 는 state 의 key 값
        print(f"\n[ {key} ]\n")
        if key == "messages":
            print(f"메시지 개수: {len(value)}")
            # print(value)
    print("===" * 10, " 단계 ", "===" * 10)
