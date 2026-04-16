import os
from pathlib import Path

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "2024_KB_부동산_보고서_최종.pdf"
FAISS_DIR = BASE_DIR / "faiss_index"

SESSION_STORE = {}


def get_openai_api_key() -> str:
    # Streamlit Cloud에서는 Secrets 우선
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # 로컬 실행용 fallback
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")


def validate_environment() -> None:
    if not PDF_PATH.exists():
        st.error(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    _ = get_openai_api_key()


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]


def format_docs(docs) -> str:
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
    api_key = get_openai_api_key()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    index_file = FAISS_DIR / "index.faiss"
    pkl_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and pkl_file.exists():
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


@st.cache_resource
def initialize_chain():
    api_key = get_openai_api_key()

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
            MessagesPlaceholder(variable_name="chat_history", n_messages=4),
            ("human", "{question}"),
        ]
    )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
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

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def main():
    st.set_page_config(page_title="KB부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB부동산 보고서 AI 어드바이저")
    st.caption("2024 KB부동산 보고서 기반 질의응답 시스템")

    validate_environment()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("부동산 관련 질문을 입력하세요.")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    chain = initialize_chain()

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = chain.invoke(
                {"question": user_prompt},
                config={"configurable": {"session_id": "streamlit_session"}},
            )
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


if __name__ == "__main__":
    main()
