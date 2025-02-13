import yaml

from typing import Annotated, Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.graph.message import add_messages

from settings import secret_path
from utils import graph_to_png

memory = MemorySaver()


class State(MessagesState):
    messages: Annotated[list, add_messages]
    summary: str


with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


def ask_llm(state: State):
    summary = state.get("summary", "")

    if summary:
        system_message = f"Summary of conversation before: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    return {"messages": llm.invoke(messages)}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]

    if len(messages) > 6:
        return "summarize_conversation"
    return END


def summarize_conversation(state: State):
    summary = state.get("summary")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date : {summary}\n\n",
            "Extend the summary by taking into account the new messages above in Korean:",
        )
    else:
        summary_message = "Create a summary of the conversation above in Korean:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


workflow = StateGraph(State)
workflow.add_node("chat", ask_llm)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "chat")
workflow.add_conditional_edges(
    "chat",
    should_continue,
)
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=memory)
# graph_to_png(app, "11_graph.png")

config = {"configurable": {"thread_id": 1}}

input_message = HumanMessage(content="안녕하세요? 저는 김선진 입니다.")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
    if "summary" in event:
        print("대화요약: \n\n", event["summary"])

# 두 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 이름이 뭔지 기억하세요?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
    if "summary" in event:
        print("\n대화요약: \n\n", event["summary"])

# 세 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 직업은 AI 연구원이에요")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
    if "summary" in event:
        print("\n대화요약: \n\n", event["summary"])

input_message = HumanMessage(content="최근 LLM 에 대해 좀 더 알아보고 있어요. LLM 에 대한 최근 논문을 읽고 있습니다.")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
    if "summary" in event:
        print("\n대화요약: \n\n", event["summary"])


input_message = HumanMessage(content="혹시 제이름이랑 직업 기억하나요?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
    if "summary" in event:
        print("\n대화요약: \n\n", event["summary"])

messages = app.get_state(config).values["messages"]
for message in messages:
    message.pretty_print()
