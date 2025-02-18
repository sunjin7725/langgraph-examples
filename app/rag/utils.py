import yaml

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from settings import secret_path
from utils import load_chat_model


def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    return "\n".join(
        [f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>" for doc in docs]
    )


def format_task(tasks):
    # 결과를 저장할 빈 리스트 생성
    task_time_pairs = []

    # 리스트를 순회하면서 각 항목을 처리
    for item in tasks:
        # 콜론(:) 기준으로 문자열을 분리
        task, time_str = item.rsplit(":", 1)
        # '시간' 문자열을 제거하고 정수로 변환
        time = int(time_str.replace("시간", "").strip())
        # 할 일과 시간을 튜플로 만들어 리스트에 추가
        task_time_pairs.append((task, time))

    # 결과 출력
    return task_time_pairs


def question_rewrite(query: str) -> str:
    re_write_prompt = PromptTemplate(
        template="""Reformulate the given question to enhance its effectiveness for vectorstore retrieval.

    - Analyze the initial question to identify areas for improvement such as specificity, clarity, and relevance.
    - Consider the context and potential keywords that would optimize retrieval.
    - Maintain the intent of the original question while enhancing its structure and vocabulary.

    # Steps

    1. **Understand the Original Question**: Identify the core intent and any keywords.
    2. **Enhance Clarity**: Simplify language and ensure the question is direct and to the point.
    3. **Optimize for Retrieval**: Add or rearrange keywords for better alignment with vectorstore indexing.
    4. **Review**: Ensure the improved question accurately reflects the original intent and is free of ambiguity.

    # Output Format

    - Provide a single, improved question.
    - Do not include any introductory or explanatory text; only the reformulated question.

    # Examples

    **Input**: 
    "What are the benefits of using renewable energy sources over fossil fuels?"

    **Output**: 
    "How do renewable energy sources compare to fossil fuels in terms of benefits?"

    **Input**: 
    "How does climate change impact polar bear populations?"

    **Output**: 
    "What effects does climate change have on polar bear populations?"

    # Notes

    - Ensure the improved question is concise and contextually relevant.
    - Avoid altering the fundamental intent or meaning of the original question.


    [REMEMBER] Re-written question should be in the same language as the original question.

    # Here is the original question that needs to be rewritten:
    {question}
    """,
        input_variables=["generation", "question"],
    )

    model = load_chat_model(model="gpt-4o-mini", temperature=0)
    question_rewriter = re_write_prompt | model | StrOutputParser()
    return question_rewriter.invoke({"question": query})
