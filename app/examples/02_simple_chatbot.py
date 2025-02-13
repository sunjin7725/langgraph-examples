import yaml
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from settings import secret_path
from utils import graph_to_png


class State(TypedDict):
    messages: Annotated[list, add_messages]


with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
api_key = secret["openai"]["api_key"]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    question = "서울의 유명한 맛집 TOP 10 추천해줘"

    for event in graph.stream({"messages": [("user", question)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
