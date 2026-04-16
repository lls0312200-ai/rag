import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(r"C:\RAG\data\.env")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\RAG\data")
PDF_PATH = "data/2024_KB_부동산_보고서_최종.pdf"
FAISS_DIR = BASE_DIR / "faiss_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
SESSION_STORE = {}


def validate_environment():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")


def get_session_history(session_id: str):
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def process_pdf():
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)


@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    if (FAISS_DIR / "index.faiss").exists() and (FAISS_DIR / "index.pkl").exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(
        documents=process_pdf(),
        embedding=embeddings,
    )
    vectorstore.save_local(str(FAISS_DIR))
    return vectorstore
    # FAISS_DIR.mkdir(parents=True, exist_ok=True)
    # vectorstore = FAISS.from_documents(
    #     documents=process_pdf(),
    #     embedding=embeddings,
    # )
    # vectorstore.save_local(str(FAISS_DIR))
    # return vectorstore
    # if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
    #     return Chroma(
    #         persist_directory=str(CHROMA_DIR),
    #         embedding_function=embeddings,
    #     )
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # return Chroma.from_documents(
    #     documents=process_pdf(),
    #     embedding=embeddings,
    #     persist_directory=str(CHROMA_DIR),
    # )


@st.cache_resource
def initialize_chain():
    retriever = initialize_vectorstore().as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB부동산 보고서 전문가입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하고 간결하게 답변하세요.
컨텍스트에서 확인되지 않은 내용은 추측하지 말고 모른다고 답하세요.

컨텍스트:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history", n_messages=4),
            ("human", "{question}"),
        ]
    )
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", template),
    #         ("placeholder", "{chat_history}"),
    #         ("human", "{question}"),
    #     ]
    # )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


@st.cache_resource
def get_public_url():
    if not NGROK_AUTHTOKEN:
        return None

    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnels = ngrok.get_tunnels()
    if tunnels:
        return tunnels[0].public_url
    return ngrok.connect(8501).public_url


def main():
    validate_environment()
    st.set_page_config(page_title="KB부동산 보고서 챗봇", page_icon=":house:")
    st.title(":house: KB부동산 보고서 AI 어드바이저")
    st.caption("2024 KB부동산 보고서 기반 질의응답 시스템")

    public_url = get_public_url()
    if public_url:
        st.info(f"외부 접속 URL: {public_url}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("부동산 관련 질문을 입력하세요.")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = initialize_chain()
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = chain.invoke(
                {"question": prompt},
                {"configurable": {"session_id": "streamlit_session"}},
            )
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
