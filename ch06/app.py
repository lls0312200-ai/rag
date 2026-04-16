from pyngrok import ngrok

ngrok.set_auth_token("3CLRKdEWRSmsb0qZYC5TK2BXgIN_2AyfBq5ryo4Xx2mAbuah8")

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv("C:\\RAG\\data\\.env")
api_key = os.getenv("OPENAI_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_index"


@st.cache_resource
def process_pdf(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if (FAISS_DIR / "index.faiss").exists() and (FAISS_DIR / "index.pkl").exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    chunks = process_pdf(r"C:\RAG\data\2024_KB_부동산_보고서_최종.pdf")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_DIR))
    return vectorstore
    # return Chroma.from_documents(chunks, embeddings)


@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB부동산 보고서 전문가입니다. 다음 정보를 바탕으로 질문에 답변해 주세요.

    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def main():
    st.set_page_config(page_title="KB부동산 보고서 챗봇", page_icon=":house:")
    st.title(":house: KB부동산 보고서 AI 어드바이저")
    st.caption("2024 KB부동산 보고서 기반 질의응답 시스템")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
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

from pyngrok import ngrok

public_url = ngrok.connect(8501)
print(f"Streamlit 앱이 다음 URL에서 실행 중입니다: {public_url}")
