import os
import yaml

from typing import Union, Callable, List

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import MermaidDrawMethod, NodeStyles
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from settings import secret_path


def graph_to_png(graph: CompiledStateGraph, output_file_path: str = "graph.png", xray: Union[int, bool] = False):
    """
    CompiledStateGraph 를 png 파일로 생성합니다.

    이 함수는 주어진 그래프 객체가 CompiledStateGraph 인스턴스인 경우
    해당 그래프를 Mermaid 형식의 PNG 이미지로 생성합니다.

    Args:
        graph: 시각화할 그래프 객체. CompiledStateGraph 인스턴스여야 합니다.
        ouput_file_path: png 이미지가 생성될 경로.
        xray: 표시할 서브그래프.

    Returns:
        output_file_path에 png 파일이 생성됩니다.

    Raises:
        Exception: 그래프 시각화 과정에서 오류가 발생한 경우 예외를 출력합니다.
    """
    try:
        if isinstance(graph, CompiledStateGraph):
            graph.get_graph(xray=xray).draw_mermaid_png(
                background_color="white",
                node_colors=NodeStyles(),
                output_file_path=output_file_path,
            )
    except Exception as e:
        print(f"[ERROR] Visualize Graph Error: {e}")


def get_role_from_messages(msg):
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages):
    return "\n".join([f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages])


def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
        if not node_names or curr_node in node_names:
            # 콜백 함수가 있는 경우 실행
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            # 콜백이 없는 경우 기본 출력
            else:
                # 노드가 변경된 경우에만 구분선 출력
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node


def load_chat_model(model="gpt-4o-mini", temperature=None, stream=True):
    with open(secret_path) as f:
        secret = yaml.safe_load(f)
    model = ChatOpenAI(api_key=secret["openai"]["api_key"], model=model, temperature=temperature, streaming=stream)
    return model
