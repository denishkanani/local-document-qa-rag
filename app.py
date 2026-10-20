
import os
import tempfile

import streamlit as st

from src.document_loader import load_pdf, chunk_documents
from src.vector_store import VectorStore
from src.rag_pipeline import answer_question


# -----------------------------
# 1. Configure the application
# -----------------------------

st.set_page_config(
    page_title="Local Document Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Local Document Q&A with RAG")

st.write(
    "Upload PDFs and ask questions about their content. "
    "Answers are generated using retrieved document passages."
)


# -----------------------------
# 2. Initialize session state
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_names" not in st.session_state:
    st.session_state.document_names = []


# -----------------------------
# 3. Upload PDF documents
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)


# -----------------------------
# 4. Process the documents
# -----------------------------

if st.button("Process documents", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        all_chunks = []
        document_names = [
            file.name for file in uploaded_files
        ]

        try:
            with st.spinner(
                "Extracting text from PDFs..."
            ):
                for uploaded_file in uploaded_files:

                    # Save the uploaded PDF temporarily.
                    temporary_path = None

                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf",
                            delete=False,
                        ) as temporary_file:

                            temporary_file.write(
                                uploaded_file.getvalue()
                            )

                            temporary_path = (
                                temporary_file.name
                            )

                        # Extract the page text.
                        pages = load_pdf(temporary_path)

                        # Use the original filename for citations.
                        for page in pages:
                            page["source"] = uploaded_file.name

                        # Split the pages into chunks.
                        chunks = chunk_documents(pages)

                        all_chunks.extend(chunks)

                    finally:
                        if temporary_path and os.path.exists(
                            temporary_path
                        ):
                            os.unlink(temporary_path)

            if not all_chunks:
                st.error(
                    "No extractable text was found. "
                    "The PDF may be scanned and need OCR."
                )

            else:
                with st.spinner(
                    "Creating embeddings and building "
                    "the vector index..."
                ):
                    vector_store = VectorStore()
                    vector_store.build(all_chunks)

                # Update session state only after successful indexing.
                st.session_state.vector_store = vector_store
                st.session_state.document_names = document_names

                st.success(
                    f"Successfully processed "
                    f"{len(document_names)} PDF(s) "
                    f"into {len(all_chunks)} chunks."
                )

        except Exception as error:
            st.error(
                f"Document processing failed: {error}"
            )


# -----------------------------
# 5. Show indexed documents
# -----------------------------

if st.session_state.document_names:
    with st.expander("Processed documents"):
        for name in st.session_state.document_names:
            st.write(f"- {name}")


# -----------------------------
# 6. Ask a question
# -----------------------------

st.divider()

question = st.text_input(
    "Ask a question about your documents",
    placeholder="What is supervised learning?",
)

top_k = st.slider(
    "Number of passages to retrieve",
    min_value=1,
    max_value=8,
    value=4,
)


# -----------------------------
# 7. Retrieve and generate
# -----------------------------

if st.button("Ask question"):

    vector_store = st.session_state.vector_store

    if vector_store is None:
        st.warning(
            "Please upload and process your documents first."
        )

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                results = vector_store.search(
                    question,
                    top_k=top_k,
                )

                answer = answer_question(
                    question,
                    results,
                )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Supporting sources")

            if not results:
                st.info(
                    "No document passages were retrieved."
                )

            for number, result in enumerate(
                results,
                start=1,
            ):
                with st.expander(
                    f"Source {number}: "
                    f"{result['source']} | "
                    f"Page {result['page']} | "
                    f"Similarity {result['score']:.3f}"
                ):
                    st.write(result["text"])

        except Exception as error:
            st.error(
                f"Could not answer the question: {error}"
            )

            st.info(
                "Check that Ollama is running, "
                "the model is installed, and sufficient "
                "memory is available."
            )
