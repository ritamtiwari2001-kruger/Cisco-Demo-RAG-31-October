import hashlib
import html
import json
import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "data" / "cisco_nexus_9000_troubleshooting_guide.pdf"
CHROMA_PATH = PROJECT_DIR / "chroma_db"
INDEX_METADATA_PATH = CHROMA_PATH / "source.json"
COLLECTION_NAME = "cisco_nexus_troubleshooting"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "openai/gpt-oss-120b"
RETRIEVAL_K = 5

SYSTEM_PROMPT = """You are a technical assistant specializing in the Cisco Nexus 9000
Series Troubleshooting Guide.

Answer the user's question using only the retrieved document context as your
primary and authoritative source.

Rules:
1. Treat the retrieved passages as the only factual source. Do not fill gaps
   with general networking knowledge.
2. Do not invent commands, examples, root causes, troubleshooting steps, or
   conclusions that are not supported by those passages.
3. If the context does not support part of the answer, say so instead of
   guessing.
4. For conceptual questions, synthesize only the ideas explicitly present in
   the retrieved passages; label any limited inference as a synthesis.
5. Use only the [Page N] labels shown in the context for page references.
   Do not cite page numbers found in passage text or invent page numbers.
6. Keep the answer concise and technically clear. Use numbered steps when
   useful.

Retrieved document context:
{context}"""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )


def format_context(documents):
    context_parts = []
    for document in documents:
        page = document.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page not available"
        context_parts.append(f"[{page_label}]\n{document.page_content}")
    return "\n\n".join(context_parts)


def source_page_numbers(documents):
    pages = {
        document.metadata["page"] + 1
        for document in documents
        if isinstance(document.metadata.get("page"), int)
    }
    return sorted(pages)


def answer_question(question, retriever=None, llm=None):
    if not question or not question.strip():
        raise ValueError("Enter a question before submitting.")

    active_retriever = retriever or app_retriever
    active_llm = llm or app_llm
    documents = active_retriever.invoke(question.strip())
    if not documents:
        return "The supplied document context does not provide enough information to answer this question."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "User question:\n{question}\n\nAnswer:"),
        ]
    )
    response = active_llm.invoke(
        prompt.format_messages(
            context=format_context(documents),
            question=question.strip(),
        )
    )
    answer = response.content.strip()
    pages = source_page_numbers(documents)
    if pages:
        page_list = ", ".join(str(page) for page in pages)
        answer += (
            f"\n\nSource: Cisco Nexus 9000 Troubleshooting Guide, "
            f"pages {page_list}."
        )
    return answer


def render_history(history):
    if not history:
        return "_Your questions and answers will appear here._"

    entries = []
    for number, item in enumerate(history, start=1):
        question = html.escape(item["question"])
        answer = html.escape(item["answer"])
        entries.append(f"**{number}. You asked**\n\n{question}\n\n**Answer**\n\n{answer}")
    return "\n\n---\n\n".join(entries)


def submit_question(question, chat_messages, conversation_history):
    chat_messages = chat_messages or []
    conversation_history = conversation_history or []
    if not question or not question.strip():
        return question, chat_messages, render_history(conversation_history), conversation_history, "Enter a question before submitting."

    try:
        answer = answer_question(question)
    except Exception:
        logger.exception("Question answering failed.")
        return (
            question,
            chat_messages,
            render_history(conversation_history),
            conversation_history,
            "I couldn't complete the answer. Check the application terminal for details and try again.",
        )

    updated_chat = chat_messages + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer},
    ]
    updated_history = conversation_history + [
        {"question": question.strip(), "answer": answer}
    ]
    return "", updated_chat, render_history(updated_history), updated_history, ""


def clear_history():
    return [], "_Your questions and answers will appear here._", [], ""


def load_or_create_vectorstore(embeddings, pdf_path=PDF_PATH):
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Troubleshooting PDF not found at {pdf_path}. "
            "Place the Cisco Nexus 9000 troubleshooting guide in the project's data folder."
        )

    pdf_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    stored_digest = None
    if INDEX_METADATA_PATH.is_file():
        stored_digest = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8")).get(
            "pdf_sha256"
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
    )
    stored_documents = vectorstore.get(limit=1).get("ids", [])
    if stored_digest == pdf_digest and stored_documents:
        logger.info("Reusing the existing ChromaDB index.")
        return vectorstore

    if stored_documents:
        vectorstore.delete_collection()
        vectorstore = None

    logger.info("Loading PDF and creating the local ChromaDB index.")
    pages = PyPDFLoader(str(pdf_path)).load()
    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "The PDF contains no extractable text. Scanned PDFs requiring OCR are not supported."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=60,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )
    INDEX_METADATA_PATH.write_text(
        json.dumps({"pdf_sha256": pdf_digest}, indent=2),
        encoding="utf-8",
    )
    logger.info("Indexed %s text chunks.", len(chunks))
    return vectorstore


def build_interface():
    with gr.Blocks(title="Cisco Nexus 9000 Troubleshooting Assistant") as demo:
        gr.Markdown("# Cisco Nexus 9000 Troubleshooting Assistant")
        gr.Markdown(
            "Ask questions about the supplied Cisco Nexus 9000 troubleshooting guide. "
            "Answers are grounded in retrieved passages from the PDF."
        )

        conversation_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Conversation History")
                history_panel = gr.Markdown(
                    "_Your questions and answers will appear here._"
                )
                clear_button = gr.Button("Clear History")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=500)
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="Ask a question about the troubleshooting guide...",
                    lines=2,
                )
                with gr.Row():
                    submit_button = gr.Button("Ask", variant="primary")
                    status = gr.Markdown()

        outputs = [question_box, chatbot, history_panel, conversation_state, status]
        inputs = [question_box, chatbot, conversation_state]
        question_box.submit(submit_question, inputs=inputs, outputs=outputs)
        submit_button.click(submit_question, inputs=inputs, outputs=outputs)
        clear_button.click(
            clear_history,
            inputs=[],
            outputs=[chatbot, history_panel, conversation_state, status],
        )
    return demo


def main():
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the project's .env file."
        )
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Troubleshooting PDF not found at {PDF_PATH}. "
            "Place it in the project's data folder."
        )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = load_or_create_vectorstore(embeddings)
        global app_retriever, app_llm
        app_retriever = build_retriever(vectorstore)
        app_llm = ChatGroq(model=MODEL_NAME, temperature=0.1)
    except Exception as error:
        logger.exception("Application initialization failed.")
        raise RuntimeError(
            "Application initialization failed. Check the terminal output for details."
        ) from error

    build_interface().launch(server_name="127.0.0.1", server_port=7860, share=False)


app_retriever = None
app_llm = None


if __name__ == "__main__":
    main()
