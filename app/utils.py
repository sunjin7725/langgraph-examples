from typing import Union

from langchain_core.runnables.graph import MermaidDrawMethod, NodeStyles

from langgraph.graph.state import CompiledStateGraph


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
