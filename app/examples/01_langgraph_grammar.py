from typing import Dict, TypedDict, Annotated, List
from pydantic import Field, BaseModel, ValidationError
from langgraph.graph import add_messages
from langchain_core.messages import AIMessage, HumanMessage


class Person(TypedDict):
    name: Annotated[str, "이름"]
    age: Annotated[int, "나이"]
    job: Annotated[str, "직업"]


class Employee(BaseModel):
    id: Annotated[int, Field(..., description="직원 ID")]
    name: Annotated[str, Field(..., min_length=3, max_length=50, description="이름")]
    age: Annotated[int, Field(gt=18, lt=65, description="나이 (19-64세)")]
    salary: Annotated[int, Field(gt=0, lt=10000, description="연봉 (단위: 만원, 최대 10억)")]
    skills: Annotated[List[str], Field(min_items=1, max_items=10, description="보유 기술 (1-10개)")]


class MyData(TypedDict):
    messages: Annotated[list, add_messages]


if __name__ == "__main__":
    msg1 = [HumanMessage(content="안녕하세요?", id=1)]
    msg2 = [AIMessage(content="반갑습니다~", id=2)]

    msg1_1 = [HumanMessage(content="할루할루~", id=1)]

    result = add_messages(msg1, msg2)
    result = add_messages(result, msg1_1)
    print(result)
