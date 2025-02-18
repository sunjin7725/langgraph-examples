import os
import uuid
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from rag.pdf import PDFRetrievalChain
from rag.pgvector.vectorstore import Postgresql
from rag.utils import format_docs
from settings import data_dir
from utils import messages_to_history, graph_to_png, stream_graph

vectorstore = Postgresql().create_chain()

retriever = vectorstore.retriever
chain = vectorstore.chain

# question = "국세 기본법의 목적에 대해서 설명해주세요"
# search_result = retriever.invoke(question)
# print(search_result)
# print("*" * 80)

# answer = chain.invoke({"question": question, "context": search_result, "chat_history": []})
# print(answer)


class GraphState(TypedDict):
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]


def retrieve_document(state: GraphState) -> GraphState:
    lastest_question = state["question"]
    retrieved_docs = retriever.invoke(lastest_question)
    retrieved_docs = format_docs(retrieved_docs)
    return GraphState(context=retrieved_docs)


def llm_answer(state: GraphState) -> GraphState:
    lastest_question = state["question"]

    context = state["context"]
    response = chain.invoke(
        {"question": lastest_question, "context": context, "chat_history": messages_to_history(state["messages"])}
    )

    return GraphState(answer=response, messages=[("user", lastest_question), ("assistant", response)])


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)


memory = MemorySaver()

app = workflow.compile(checkpointer=memory)
# graph_to_png(app, "12_graph.png")

config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})
inputs = GraphState(question="조세 탈루 혐의가 인정되면 어떤 벌을 받아?")

stream_graph(app, inputs, config, ["llm_answer"])

outputs = app.get_state(config).values
print(outputs)

print(f'Question: {outputs["question"]}')
print("===" * 20)
print(f'Answer:\n{outputs["answer"]}')
