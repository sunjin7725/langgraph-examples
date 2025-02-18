from typing import Literal, TypedDict, Annotated, List
from pydantic import BaseModel, Field

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_tools import ddg_search
from rag.pgvector.vectorstore import PostgresVectorstore
from utils import load_chat_model, graph_to_png, stream_graph
from rag.utils import format_docs

vectorstroe = PostgresVectorstore().create_chain()
chain = vectorstroe.chain
retriever = vectorstroe.retriever


class RouteQuery(BaseModel):
    """Route a user query to the mose relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., description="Given a user question choose to route it to web search or vectorstore"
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to evaluate the appropriateness of answer to questions."""

    binary_score: str = Field(description="Indicate 'yes' or 'no' whether the answer solves the question.")


class GraphState(TypedDict):
    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]


model = load_chat_model(model="gpt-4o-mini", temperature=0)
structed_model_router = model.with_structured_output(RouteQuery)
structed_model_grader = model.with_structured_output(GradeDocuments)
structed_model_hallucination = model.with_structured_output(GradeHallucinations)
structed_model_answer_check = model.with_structured_output(GradeAnswer)

router_system_prompt = """You are an expert at routing a user question to a vectorstore or web search
The vectorstroe contains documents related to information about Korean tax law
Use the vectorstore for questions on these topics. Otherwise, use web-search"""
route_prompt = ChatPromptTemplate.from_messages([("system", router_system_prompt), ("human", "{question}")])
router_chain = route_prompt | structed_model_router

grader_system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        (
            "user",
            """
        Retrieved documents: 
            {document}
        
        User question:
            {question}
     """,
        ),
    ]
)
grade_chain = grade_prompt | structed_model_grader

hallucination_system_prompt = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system_prompt),
        (
            "user",
            """
        Set of facts: 
            {documents}
     
        LLM generation:
            {generation}
    """,
        ),
    ]
)
hallucination_chain = hallucination_prompt | structed_model_hallucination

answer_check_system_prompt = """
    You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. Yes means that the answer resolves the question.
"""
answer_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_check_system_prompt),
        (
            "user",
            """
        User question:
            {question}
     
        LLM generation:
            {generation}
    """,
        ),
    ]
)
answer_check_chain = answer_check_prompt | structed_model_answer_check

rewrite_system_prompt = """
    You are a question re-writer that converts an input equestion to abetter vesrsion that is optimized for vectorstore retrieval.
    Look at the input and try to reason about the underlying semantic intent / meaning.
"""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system_prompt),
        (
            "user",
            """
        Here is the initial question:
            {question}
        Formulate an imporved question.
     """,
        ),
    ]
)
rewrite_chain = rewrite_prompt | model | StrOutputParser()

rag_prompt = hub.pull("teddynote/rag-prompt")
rag_chain = rag_prompt | model | StrOutputParser()


#### 노드 정의 ###########################################################################
def retrieve(state):
    print("==== [RETRIEVE] ====")
    question = state["question"]
    return {"quesiton": question, "documents": retriever.invoke(question)}


def generate(state):
    print("==== [GENERATE] ====")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = grade_chain.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score

        if grade == "yes":
            print("---GRADE DOCUMENTS: ADD RELEVANT DOCS")
            filtered_docs.append(d)
        else:
            print("---GRADE DOCUMENTS: NOT RELEVANT DOCS PASSED")
    return {"documents": documents, "question": question}


def transform_query(state):
    print("==== [TRANSFORM QUERY] ====")
    question = state["question"]
    documents = state["documents"]

    better_question = rewrite_chain.invoke({"question": question})
    return {"question": better_question, "documents": documents}


def web_search(state):
    print("==== [WEB SEARCH] ====")
    question = state["question"]
    return {"question": question, "documents": ddg_search.invoke(question)}


###########################################################################################


#### 엣지 정의 #############################################################################
def route_question(state):
    print("==== [ROUTE QUESTION] ====")

    question = state["question"]
    source = router_chain.invoke({"question": question})
    if source.datasource == "vectorstore":
        print("==== [ROUTE TO VECTORSTORE] ====")
        return "vectorstore"
    else:
        print("==== [ROUTE TO WEB_SEARCH] ====")
        return "web_search"


def decide_to_generate(state):
    print("==== [DECISION TO GENERATE] ==== ")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVAT TO QUESTION -> TRANSFORM QUERY")
        return "transform_query"
    else:
        print("---DECISION: GENERATE")
        return "generate"


def hallucination_check(state):
    print("==== [CHECK HALLUCINATION] ====")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---CHECK HALLUCINATION: IT`S GROUNDED IN -> CHECK RELEVANT ANSWER")
        score = answer_check_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score

        if grade == "yes":
            print("---RELEVANT ANSWER CHECK: OK!")
            return "relevant"
        else:
            print("---RELEVANT ANSWER CHECK: NOT OKAY")
            return "not_relevant"
    else:
        print("---CHECK HALLUCINATION: IT`S NOT GROUNDED IN, RE-TRY")
        return "hallucination"


###########################################################################################

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"}
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    hallucination_check,
    {
        "relevant": END,
        "not_relevant": "transform_query",
        "hallucination": "generate",
    },
)
memory = MemorySaver()
app = workflow.compile(memory)
graph_to_png(app, "17_graph.png")

config = RunnableConfig(recursion_limit=10, configurable={"thread_id": 1})

# inputs = {"question": "국세 기본법이 뭐야"}
inputs = {"question": "2024년 노벨 문학상 수상자는 누구야?"}

stream_graph(app, inputs, config, ["agent", "rewrite", "generate"])
