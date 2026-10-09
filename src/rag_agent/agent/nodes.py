
"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import (
    AgentResponse,
    AgentState,
    RetrievedChunk,
)
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Backend Resources
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_cached_llm():
    """
    Return a cached LLM instance.

    Avoids recreating the LLM client on every graph execution.
    """
    settings = get_settings()
    return LLMFactory(settings).create()


@lru_cache(maxsize=1)
def get_cached_vector_store():
    """
    Return a cached VectorStoreManager instance.

    Avoids repeatedly initializing ChromaDB and embeddings.
    """
    return VectorStoreManager()


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """

    # Extract latest HumanMessage from the state dictionary.
    messages = state.get("messages", [])

    original_query = next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        "",
    )

    if not original_query:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    rewrite_prompt = (
        "You are a search query optimization assistant.\n"
        "Rewrite the following question into a concise, "
        "technical search query suitable for retrieving "
        "deep learning study materials.\n"
        "Preserve the original meaning.\n"
        "Return only the rewritten query.\n\n"
        f"Question: {original_query}"
    )

    try:
        llm = get_cached_llm()

        result = llm.invoke(
            [HumanMessage(content=rewrite_prompt)]
        )

        rewritten_query = str(result.content).strip()

        if not rewritten_query:
            rewritten_query = original_query

    except Exception:
        # If rewriting fails, continue with the original query.
        rewritten_query = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """

    manager = get_cached_vector_store()

    query_text = (
        state.get("rewritten_query")
        or state.get("original_query", "")
    )

    if not query_text:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    chunks = manager.query(
        query_text=query_text,
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter"),
    )

    return {
        "retrieved_chunks": chunks,
        "no_context_found": len(chunks) == 0,
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """

    settings = get_settings()

    # ---- Hallucination Guard -----------------------------------------------

    if state.get("no_context_found", False):
        no_context_message = (
            "I was unable to find relevant information in the corpus "
            "for your query. This may mean the topic is not yet covered "
            "in the study material, or your query may need to be "
            "rephrased. Please try a more specific deep learning topic "
            "such as 'LSTM forget gate' or 'CNN pooling layers'."
        )

        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )

        return {
            "final_response": response,
            "messages": [
                AIMessage(content=no_context_message)
            ],
        }

    # ---- Build Context from Retrieved Chunks -------------------------------

    retrieved_chunks = state.get("retrieved_chunks", [])

    context_parts = []
    sources = []
    scores = []

    for chunk in retrieved_chunks:
        citation = (
            f"{chunk.metadata.topic} | "
            f"{chunk.metadata.source}"
        )

        context_parts.append(
            f"[SOURCE: {citation}]\n"
            f"{chunk.chunk_text}\n"
        )

        if citation not in sources:
            sources.append(citation)

        scores.append(chunk.score)

    context = "\n".join(context_parts)

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # ---- Trim Conversation History -----------------------------------------

    messages_in_state = state.get("messages", [])

    # Exclude latest user message because it is added separately.
    history = list(messages_in_state[:-1])

    def approximate_token_count(messages):
        """
        Estimate token count from message content.

        Approximately four characters per token.
        """
        total_characters = sum(
            len(str(message.content))
            for message in messages
        )

        return max(1, total_characters // 4)

    max_context_tokens = getattr(
        settings,
        "max_context_tokens",
        4096,
    )

    if history:
        history = trim_messages(
            history,
            max_tokens=max_context_tokens,
            strategy="last",
            token_counter=approximate_token_count,
            allow_partial=False,
        )

    # ---- Build Generation Prompt -------------------------------------------

    original_query = state.get("original_query", "")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "Use only the retrieved study material below "
                "to answer the user's question.\n"
                "Explain concepts clearly and accurately.\n"
                "Cite the relevant source documents.\n"
                "If the context does not contain the answer, "
                "say that the information is not available.\n\n"
                f"RETRIEVED CONTEXT:\n{context}"
            )
        ),
        *history,
        HumanMessage(content=original_query),
    ]

    # ---- Generate Answer ---------------------------------------------------

    llm = get_cached_llm()

    result = llm.invoke(messages)

    answer = (
        result.content
        if isinstance(result.content, str)
        else str(result.content)
    )

    # ---- Construct Structured Response -------------------------------------

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", ""),
    )

    return {
        "final_response": response,
        "messages": [
            AIMessage(content=answer)
        ],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation.

    Notes
    -----
    This implementation always routes to generation_node so the
    hallucination guard can return a proper response even when
    retrieval finds no relevant context.
    """

    return "generate"
