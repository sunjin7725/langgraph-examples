from datetime import datetime
from langchain.tools import Tool

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools import PythonAstREPLTool

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

search_wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=10, time="y")
ddg_search_results = DuckDuckGoSearchResults(api_wrapper=search_wrapper, output_format="json")
ddg_search = DuckDuckGoSearchRun(api_wrapper=search_wrapper)

wikipidia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

python_repl = PythonAstREPLTool()

datetime_tool = Tool(name="now_datetime", func=lambda x: datetime.now(), description="Returns the current datetime")
who_are_you_tool = Tool(
    name="who_are_you",
    func=lambda x: """
        You have to answer the question of who you are,
        You made by Okestro AI service team, and you are a AI chatbot service.
        And, You made by "김선진" in company name of "오케스트로" in korea.
        But, Okestro is not made by "김선진", if the user want to know company, 
        you have to search and respond by user`s question.
    """,
    description="Return what is service purpose and who make this service.",
)


def get_remote_ip() -> str:
    """Get remote client ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip


get_remote_ip_tool = Tool(name="Get remote IP", func=lambda x: get_remote_ip(), description="Get remote client ip.")
