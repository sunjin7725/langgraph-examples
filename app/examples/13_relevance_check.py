import os
import uuid
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

from langchain_tools import ddg_search
from rag.pdf import PDFRetrievalChain
from rag.pgvector.vectorstore import PostgresVectorstore
from rag.utils import format_docs, question_rewrite
from settings import data_dir
from utils import messages_to_history, graph_to_png, stream_graph
from relevance import GroundednessChecker

vectorstore = PostgresVectorstore().create_chain()

retriever = vectorstore.retriever
chain = vectorstore.chain

# question = "국세 기본법의 목적에 대해서 설명해주세요"
# search_result = retriever.invoke(question)
# print(search_result)
# print("*" * 80)

# answer = chain.invoke({"question": question, "context": search_result, "chat_history": []})
# print(answer)


class GraphState(TypedDict):
    question: Annotated[str, add_messages]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    relevance: Annotated[str, "Relevance"]


def retrieve_document(state: GraphState) -> GraphState:
    lastest_question = state["question"][-1].content
    retrieved_docs = retriever.invoke(lastest_question)
    retrieved_docs = format_docs(retrieved_docs)
    return GraphState(context=retrieved_docs)


def llm_answer(state: GraphState) -> GraphState:
    lastest_question = state["question"][-1].content

    context = state["context"]
    response = chain.invoke(
        {"question": lastest_question, "context": context, "chat_history": messages_to_history(state["messages"])}
    )

    return GraphState(answer=response, messages=[("user", lastest_question), ("assistant", response)])


def relevance_check(state: GraphState) -> GraphState:
    question_answer_relevant = GroundednessChecker(llm=vectorstore.model, target="question-retrieval").create()

    response = question_answer_relevant.invoke({"question": state["question"], "context": state["context"]})
    print("==== [RELEVANCE CHECK] ====")
    print(response.score)
    return GraphState(relevance=response.score)


def is_relevant(state: GraphState):
    return state["relevance"]


def web_search(state: GraphState) -> GraphState:
    search_query = state["question"][-1].content
    search_result = ddg_search.invoke(search_query)
    return GraphState(context=search_result)


def query_rewrite(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    question_rewritten = question_rewrite(latest_question)
    return GraphState(question=question_rewritten)


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("web_search", web_search)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("query_rewrite", query_rewrite)

workflow.add_edge(START, "query_rewrite")
workflow.add_edge("query_rewrite", "retrieve")
workflow.add_edge("retrieve", "relevance_check")
workflow.add_conditional_edges("relevance_check", is_relevant, {"yes": "llm_answer", "no": "web_search"})
workflow.add_edge("web_search", "llm_answer")
workflow.add_edge("llm_answer", END)


memory = MemorySaver()

app = workflow.compile(checkpointer=memory)
# graph_to_png(app, "13_graph.png")

config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})
inputs = GraphState(question="국세와 지방세의 범위 차이가 뭐야")

try:
    stream_graph(app, inputs, config, ["query_rewrite", "web_search", "relevance_check", "llm_answer"])
except GraphRecursionError as recursion_error:
    print(f"GraphRecursionError: {recursion_error}")
