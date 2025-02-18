import os
import yaml

from typing import Annotated, Sequence, TypedDict, Literal

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from pydantic import BaseModel, Field

from rag.pgvector.vectorstore import PostgresVectorstore
from langchain_tools import ddg_search, datetime_tool, who_are_you_tool, wikipidia, get_remote_ip_tool
from utils import load_chat_model, graph_to_png, stream_graph

vectorstore = PostgresVectorstore().create_chain()
retriever = vectorstore.retriever
chain = vectorstore.chain

retriever_tool = create_retriever_tool(
    retriever,
    name="tax_law_retriever",
    description="""
    A tool for searching and returning information about Korean tax law. 
    Users can input tax-related questions, and this tool will find relevant laws, regulations, and cases. 
    It includes the latest tax law information.
    """,
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
    ),
)

tools = [retriever_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class grade(BaseModel):
    """A binary score for relevance checks."""

    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the question or 'no' if it is not"
    )


def grade_documents(state) -> Literal["generate", "rewrite"]:
    model = load_chat_model(model="gpt-4o-mini", temperature=0, stream=True)
    llm_with_tools = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tools
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    retrieved_docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": retrieved_docs})
    score = scored_result.binary_score

    if score == "yes":
        print("==== [DECISION: DOCS RELEVANT] ====")
        return "generate"
    else:
        print("==== [DECISION: DOCS NOT RELEVANT] ====")
        print(score)
        return "rewrite"


def agent(state):
    messages = state["messages"]
    model = load_chat_model(model="gpt-4o-mini", temperature=0, stream=True)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}


def rewrite(state):
    print("==== [QUERY REWRITE] ====")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    model = load_chat_model(model="gpt-4o-mini", temperature=0, stream=True)
    response = model.invoke(msg)
    return {"messages": response}


def generate(state):
    messages = state["messages"]
    question = messages[0].content

    docs = messages[-1].content

    prompt = hub.pull("teddynote/rag-prompt")
    model = load_chat_model(model="gpt-4o-mini", temperature=0, stream=True)

    rag_chain = prompt | model | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile()
# graph_to_png(graph, "16_graph.png")

config = RunnableConfig(recursion_limit=10, configurable={"thread_id": 1})

inputs = {"messages": [("user", "테디노트의 LangChain 튜토리얼 주소는?")]}

stream_graph(graph, inputs, config, ["agent", "rewrite", "generate", "retrieve"])
