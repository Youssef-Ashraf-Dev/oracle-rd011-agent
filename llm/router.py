"""
RD.011 Agent — Provider-agnostic LLM router.

Every node calls ``get_client(TaskType)`` to obtain the appropriate
LangChain chat model.  Provider SDKs are imported **only** inside this
module.  No other file in the project imports a provider SDK directly.
"""

from __future__ import annotations

import logging

from config import (
    CAPABILITY_MAP,
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    MISTRAL_API_KEY,
    OPENROUTER_API_KEY,
    TaskType,
)

logger = logging.getLogger(__name__)


def _build_google_client(model: str, task_type=None):
    """Build a ChatGoogleGenerativeAI client."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")

    # LARGE_CONTEXT tasks: apply thinking_level="medium" for balance in reasoning and inference speed.
    kwargs = {
        "model": model,
        "google_api_key": GOOGLE_API_KEY,
        "temperature": 0.3,
        "convert_system_message_to_human": True,
    }
    if task_type == TaskType.LARGE_CONTEXT:
        kwargs["thinking_level"] = "medium"

    return ChatGoogleGenerativeAI(**kwargs)


def _build_groq_client(model: str):
    """Build a ChatGroq client for the Groq inference platform."""
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment.")
    return ChatGroq(
        model=model,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
    )





def _build_openrouter_client(model: str, task_type=None):
    """Build a ChatOpenAI client pointing at OpenRouter API."""
    from langchain_openai import ChatOpenAI

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment.")

    # LARGE_CONTEXT extraction outputs ~7K tokens of JSON — use a higher cap.
    # Confirmed from telemetry: truncation occurs at char ~27,000 (~6,750 tokens).
    # Other tasks (planning ~500t, generation ~2K t) are fine with 8,192.
    max_tokens = 16_000 if task_type == TaskType.LARGE_CONTEXT else 8_192

    return ChatOpenAI(
        model=model,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=max_tokens,
    )


_BUILDERS = {
    "google": lambda cfg, task_type=None: _build_google_client(cfg["model"], task_type),
    "groq": lambda cfg, task_type=None: _build_groq_client(cfg["model"]),
    "openrouter": lambda cfg, task_type=None: _build_openrouter_client(cfg["model"], task_type),
}


def get_client(provider: str, model_name: str, task_type: TaskType = None):
    """
    Return the appropriate LangChain chat model for the given provider and model.

    Parameters
    ----------
    provider
        Provider name: 'google', 'groq', 'mistral', 'openrouter'
    model_name
        Model identifier for that provider
    task_type
        Optional TaskType for special handling (e.g. thinking_level for Google LARGE_CONTEXT)

    Returns
    -------
    A LangChain BaseChatModel instance.

    Raises
    ------
    ValueError
        If provider is unknown, model is unavailable, or API key is missing.
    """
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {', '.join(_BUILDERS.keys())}"
        )

    cfg = {"model": model_name}
    try:
        client = builder(cfg, task_type)
        logger.info(
            "Initialised %s client (model=%s, task=%s)",
            provider,
            model_name,
            task_type.value if task_type else "none",
        )
        return client
    except Exception as exc:
        logger.error(
            "Failed to initialise %s client for model %s: %s",
            provider,
            model_name,
            exc,
        )
        raise
