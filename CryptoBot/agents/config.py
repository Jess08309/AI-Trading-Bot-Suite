"""Agent framework configuration — reads from .env."""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cryptotrades", ".env"))

# ── LLM Keys ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ── Model Selection ───────────────────────────────────────
# Which model each agent uses.  Swap these as you add keys.
TECHNICAL_MODEL = os.getenv("AGENT_TECHNICAL_MODEL", "gpt-4o")
SENTIMENT_MODEL = os.getenv("AGENT_SENTIMENT_MODEL", "gpt-4o")
RISK_MODEL = os.getenv("AGENT_RISK_MODEL", "gpt-4o")
ORCHESTRATOR_MODEL = os.getenv("AGENT_ORCHESTRATOR_MODEL", "gpt-4o")

# ── Behaviour ─────────────────────────────────────────────
# "advisor" = log-only, "active" = can modify signals
AGENT_MODE = os.getenv("AGENT_MODE", "advisor")

# Temperature (lower = more deterministic / conservative)
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.2"))

# Max tokens per agent response
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "512"))

# Timeout per agent call (seconds)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "30"))

# Enable/disable the advisor entirely (hot toggle via .env)
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() in ("true", "1", "yes")

# Log file
AGENT_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "agent_advisor.log"
)
