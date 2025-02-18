import os
import yaml

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Union
from operator import itemgetter

from settings import secret_path


class PostgresVectorstore:
    def __init__(self):
        with open(secret_path, "r", encoding="utf-8") as f:
            __secret = yaml.safe_load(f)
        __host = __secret["postgresql"]["host"]
        __port = __secret["postgresql"]["port"]
        __database = __secret["postgresql"]["database"]
        __username = __secret["postgresql"]["username"]
        __password = __secret["postgresql"]["password"]
        __open_ai_api_key = __secret["openai"]["api_key"]

        connection_string = f"postgresql+psycopg://{__username}:{__password}@{__host}:{__port}/{__database}"

        self.model = ChatOpenAI(api_key=__open_ai_api_key, model_name="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(api_key=__open_ai_api_key, model="text-embedding-3-small")
        collection_name = "langgraph_examples"

        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )

    def create_tables(self):
        self.vectorstore.create_tables_if_not_exists()

    def drop_tables(self):
        self.vectorstore.drop_tables()

    def load_documents(self, source_uris: Union[List[str], str]):
        if isinstance(source_uris, list):
            docs = []
            for source_uri in source_uris:
                loader = PDFPlumberLoader(source_uri)
                docs.extend(loader.load())
        if isinstance(source_uris, str):
            docs = PDFPlumberLoader(source_uris).load()
        for doc in docs:
            doc.metadata["source"] = self._change_source_path(doc.metadata["source"])
        return docs

    def create_text_splitter(self, chunk_size=300, chunk_overlap=50):
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def create_retriever(self, k=10):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return dense_retriever

    def create_prompt(self):
        return hub.pull("teddynote/rag-prompt-chat-history")

    def create_chain(self):
        prompt = self.create_prompt()
        self.retriever = self.create_retriever()
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | self.model
            | StrOutputParser()
        )
        return self

    def insert_pdf(self, source_uris: Union[List[str], str]):
        text_splitter = self.create_text_splitter()
        docs = self.load_documents(source_uris)
        split_docs = text_splitter.split_documents(docs)
        self.vectorstore.add_documents(split_docs)

    def _change_source_path(self, path: str) -> str:
        return os.path.split(path)[1]
