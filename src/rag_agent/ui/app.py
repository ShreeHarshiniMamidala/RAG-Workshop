"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """
    Return the compiled LangGraph agent.

    Cached so the graph and its underlying LLM resources are not
    recreated on every Streamlit rerun.
    """
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """

    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).

    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """

    st.sidebar.header("📂 Corpus Ingestion")

    # IMPLEMENTED: Streamlit document ingestion
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        try:
            all_chunks = []

            with st.spinner(
                "Processing and ingesting documents..."
            ):
                # UploadedFile objects are temporary, so save them
                # before passing their paths to DocumentChunker.
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    for uploaded_file in uploaded_files:
                        file_path = (
                            temp_path / uploaded_file.name
                        )

                        file_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        # Use chunk_file() individually because that
                        # is the implemented single-file interface.
                        chunks = chunker.chunk_file(
                            file_path
                        )

                        all_chunks.extend(chunks)

                    result = store.ingest(all_chunks)

            st.session_state[
                "last_ingestion_result"
            ] = result

            if result.ingested > 0:
                st.sidebar.success(
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped."
                )
            elif result.skipped > 0 and not result.errors:
                st.sidebar.warning(
                    "No new chunks were added. "
                    f"{result.skipped} duplicates skipped."
                )

            if result.errors:
                st.sidebar.error(
                    f"{len(result.errors)} chunk(s) "
                    "failed during ingestion."
                )

                with st.sidebar.expander(
                    "View ingestion errors"
                ):
                    for error in result.errors:
                        st.write(error)

            # Refresh document list only when the backend
            # inspection method is available.
            try:
                st.session_state[
                    "ingested_documents"
                ] = store.list_documents()
            except NotImplementedError:
                # list_documents() belongs to a different TODO
                # scope, so do not break ingestion if it has
                # not yet been implemented.
                pass

        except NotImplementedError as exc:
            st.sidebar.error(
                "A required document-processing method "
                "has not been implemented yet."
            )
            st.sidebar.caption(str(exc))

        except Exception as exc:
            st.sidebar.error(
                f"Document ingestion failed: {exc}"
            )

    # Render ingested document list if available.
    documents = st.session_state.get(
        "ingested_documents",
        [],
    )

    if documents:
        st.sidebar.divider()
        st.sidebar.subheader("Ingested Documents")

        for document in documents:
            source = document.get(
                "source",
                "Unknown",
            )
            topic = document.get(
                "topic",
                "Unknown",
            )
            chunk_count = document.get(
                "chunk_count",
                0,
            )

            st.sidebar.write(
                f"**{source}**"
            )
            st.sidebar.caption(
                f"{topic} · {chunk_count} chunks"
            )

    else:
        st.sidebar.info(
            "Upload .pdf or .md files to populate the corpus."
        )


def render_corpus_stats(
    store: VectorStoreManager,
) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """

    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    pass


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(
    store: VectorStoreManager,
) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """

    st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox(
    #     "Select document",
    #     options=[doc["source"] for doc in docs]
    # )
    #    Store selection in
    #    st.session_state["selected_document"]
    #
    # 3. On selection change:
    #    store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container
    #    (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge:
    #      topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was
    #      used in last response
    #
    # 5. Display chunk count and coverage summary below viewer

    st.info(
        "Ingest documents using the sidebar "
        "to view content here."
    )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.

    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """

    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)

    with col_topic:
        topic_options = [
            "All",
            "ANN",
            "CNN",
            "RNN",
            "LSTM",
            "Seq2Seq",
            "Autoencoder",
            "SOM",
            "BoltzmannMachine",
            "GAN",
        ]

        selected_topic = st.selectbox(
            "Topic",
            options=topic_options,
            index=0,
        )

        st.session_state.topic_filter = (
            None
            if selected_topic == "All"
            else selected_topic
        )

    with col_diff:
        difficulty_options = [
            "All",
            "beginner",
            "intermediate",
            "advanced",
        ]

        selected_difficulty = st.selectbox(
            "Difficulty",
            options=difficulty_options,
            index=0,
        )

        st.session_state.difficulty_filter = (
            None
            if selected_difficulty == "All"
            else selected_difficulty
        )

    # Chat history display
    chat_container = st.container(
        height=400
    )

    with chat_container:
        for message in (
            st.session_state.chat_history
        ):
            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

                if message.get("sources"):
                    with st.expander(
                        "📎 Sources"
                    ):
                        for source in (
                            message["sources"]
                        ):
                            st.caption(source)

                if message.get(
                    "no_context_found"
                ):
                    st.warning(
                        "⚠️ No relevant content "
                        "found in corpus."
                    )

    # IMPLEMENTED: RAG question-and-answer chat
    query = st.chat_input(
        "Ask about a deep learning topic..."
    )

    if query:
        # Add user message to Streamlit history.
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query,
            }
        )

        # Display immediately.
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        try:
            graph_input = {
                "messages": [
                    HumanMessage(
                        content=query
                    )
                ],
                "original_query": query,
                "topic_filter": (
                    st.session_state.topic_filter
                ),
                "difficulty_filter": (
                    st.session_state.difficulty_filter
                ),
            }

            config = {
                "configurable": {
                    "thread_id": (
                        st.session_state.thread_id
                    )
                }
            }

            with st.spinner(
                "Searching the corpus..."
            ):
                result = graph.invoke(
                    graph_input,
                    config=config,
                )

            response = result.get(
                "final_response"
            )

            if response is None:
                # The retrieval branch may terminate before
                # generation when no context is found.
                no_context = result.get(
                    "no_context_found",
                    True,
                )

                if no_context:
                    answer = (
                        "I couldn't find relevant "
                        "information in the corpus "
                        "for that question."
                    )
                else:
                    answer = (
                        "The agent did not return "
                        "a response."
                    )

                sources = []

            else:
                answer = response.answer
                sources = response.sources
                no_context = (
                    response.no_context_found
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "no_context_found": no_context,
            }

            st.session_state.chat_history.append(
                assistant_message
            )

            # Display assistant response immediately.
            with chat_container:
                with st.chat_message(
                    "assistant"
                ):
                    st.markdown(answer)

                    if sources:
                        with st.expander(
                            "📎 Sources"
                        ):
                            for source in sources:
                                st.caption(source)

                    if no_context:
                        st.warning(
                            "⚠️ No relevant content "
                            "found in corpus."
                        )

        except Exception as exc:
            error_message = (
                f"Unable to generate response: {exc}"
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "no_context_found": False,
                }
            )

            with chat_container:
                with st.chat_message(
                    "assistant"
                ):
                    st.error(error_message)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with:
        uv run streamlit run src/rag_agent/ui/app.py
    """

    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(
        f"🧠 {settings.app_title}"
    )

    st.caption(
        "RAG-powered interview preparation — "
        "built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(
        store,
        chunker,
    )
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()