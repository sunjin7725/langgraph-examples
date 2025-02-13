import yaml
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tools import ddg_search, python_repl, datetime_tool
from settings import secret_path
from utils import graph_to_png


tools = [ddg_search, python_repl, datetime_tool]
tool_node = ToolNode(tools)

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chat(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(MessagesState)

workflow.add_node("chat", chat)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "chat")
workflow.add_conditional_edges("chat", tools_condition)
workflow.add_edge("tools", "chat")
workflow.add_edge("chat", END)
app = workflow.compile()

# graph_to_png(app, "10_graph.png")
# question = "처음 5개의 소수를 출력하는 python 코드 작성해줘"
# question = "오케스트로가 뭐하는 곳이야"
# question = "오늘날짜에 맞는 주요뉴스 5개 알려줘"
question = "안녕?"
for event in app.stream({"messages": [("user", question)]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
