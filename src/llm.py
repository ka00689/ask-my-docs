"""Which language model writes the answers, and what to do when one is unavailable.

Locally there is Ollama, which is free and private but only reachable from your
own machine. A deployed copy has to call a hosted model instead, so this module
picks whichever is available and falls back to the next one when a provider is
down or has hit its daily limit.

Order is set by ASK_PROVIDERS (default "google,groq,ollama"). A provider is
skipped when its key is missing, so nothing has to change between your laptop
and the deployed version.

Keys are read from the environment, or from a .env file that is never committed:

    GOOGLE_API_KEY=...
    GROQ_API_KEY=...
"""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Model names change; override them with environment variables rather than
# editing code when a provider retires one.
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

DEFAULT_ORDER = "google,groq,ollama"


def load_env_file():
    """Read .env without needing an extra library, so a key never lives in code."""
    env_file = PROJECT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def as_text(content):
    """Flatten a reply into plain text.

    Providers differ: some return a string, and newer models return a list of
    content blocks (dicts with a "text" field, or plain strings). The rest of
    the code only wants the text, so the difference is absorbed here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


class Provider:
    """One model, wrapped so the rest of the code does not care which it is."""

    def __init__(self, name, model, invoke):
        self.name = name
        self.model = model
        self._invoke = invoke

    def complete(self, system, user, schema=None):
        return self._invoke(system, user, schema)

    def __repr__(self):
        return f"{self.name} ({self.model})"


def google_provider():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("  google: skipped, GOOGLE_API_KEY is not set")
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    client = ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0)

    def invoke(system, user, schema=None):
        return as_text(client.invoke([("system", system), ("human", user)]).content)

    return Provider("google", GOOGLE_MODEL, invoke)


def groq_provider():
    if not os.environ.get("GROQ_API_KEY"):
        print("  groq: skipped, GROQ_API_KEY is not set")
        return None
    from langchain_groq import ChatGroq

    client = ChatGroq(model=GROQ_MODEL, temperature=0)

    def invoke(system, user, schema=None):
        return as_text(client.invoke([("system", system), ("human", user)]).content)

    return Provider("groq", GROQ_MODEL, invoke)


def ollama_provider(model=None):
    from langchain_ollama import ChatOllama

    name = model or OLLAMA_MODEL

    def invoke(system, user, schema=None):
        # Ollama can enforce a JSON schema while generating, which the hosted
        # providers do differently; when it is unavailable, plain JSON mode.
        try:
            client = ChatOllama(model=name, temperature=0,
                                format=schema if schema else "json")
            return as_text(client.invoke([("system", system), ("human", user)]).content)
        except Exception:
            client = ChatOllama(model=name, temperature=0, format="json")
            return as_text(client.invoke([("system", system), ("human", user)]).content)

    return Provider("ollama", name, invoke)


BUILDERS = {"google": google_provider, "groq": groq_provider, "ollama": ollama_provider}


class ModelChain:
    """Tries each available provider in turn, so one outage is not an outage."""

    def __init__(self, order=None, ollama_model=None):
        names = (order or os.environ.get("ASK_PROVIDERS", DEFAULT_ORDER)).split(",")
        self.providers = []
        for name in [n.strip() for n in names if n.strip()]:
            builder = BUILDERS.get(name)
            if not builder:
                continue
            try:
                provider = (builder(ollama_model) if name == "ollama" else builder())
            except Exception as error:      # a missing library should not be fatal
                print(f"  {name}: unavailable - {type(error).__name__}: {error}")
                continue
            if provider:
                self.providers.append(provider)

        if not self.providers:
            raise SystemExit(
                "No model provider available. Either run Ollama locally, or set "
                "GOOGLE_API_KEY or GROQ_API_KEY in a .env file."
            )

    @property
    def primary(self):
        return self.providers[0]

    def complete(self, system, user, schema=None):
        """Returns (text, provider_name). Raises only if every provider fails."""
        errors = []
        for provider in self.providers:
            try:
                return provider.complete(system, user, schema), provider.name
            except Exception as error:
                # The full message matters while setting up: a rejected key and
                # an unreachable service look identical without it.
                print(f"  {provider.name} failed: {type(error).__name__}: {error}")
                errors.append(f"{provider.name}: {type(error).__name__}")
                continue
        raise RuntimeError("Every model provider failed - " + "; ".join(errors))


def describe():
    """Used at startup so the logs say which model is actually answering."""
    chain = ModelChain()
    return ", ".join(str(p) for p in chain.providers)
