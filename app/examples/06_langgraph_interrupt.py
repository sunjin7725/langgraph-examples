from examples.graph import graph, State
from langchain_core.runnables import RunnableConfig

question = "AI 관련 최신 뉴스를 알려주세요."

input = State(messages=[("user", question)])

config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "1"}, tags=["my-rag"])

for event in graph.stream(input=input, config=config, stream_mode="values", interrupt_before=["tools"]):
    for key, value in event.items():
        print(f"\n[ {key} ]\n")
        print(value[-1].content)
        if "messages" in value:
            print("메시지 개수:", len(value["messages"]))

snapshot = graph.get_state(config)

print(snapshot.next)

events = graph.stream(None, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

to_reply = None

for state in graph.get_state_history(config):
    print("메시지 수: ", len(state.values["messages"]), "다음 노드: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 3:
        to_reply = state
