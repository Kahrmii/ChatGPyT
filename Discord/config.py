import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

N8N_BUTTONS: list[dict[str, str]] = [
    {"label": "test", "url": os.getenv("N8N_WEBHOOK_TEST", "")},
    {"label": "rm unwanted mods/devs", "url": os.getenv("N8N_WEBHOOK_RM_UNWANTED", "")},
]

DEFAULT_MODEL: str = "claude-sonnet-4-5-20250929"
AVAILABLE_MODELS: list[str] = [
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20250514",
    "claude-haiku-3-5-20241022",
]
DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant in a Discord server. "
    "Keep responses concise and well-formatted for Discord. "
    "Use markdown formatting where appropriate."
)
MAX_TOKENS: int = 1024
MAX_HISTORY_LENGTH: int = 20
DISCORD_MAX_MESSAGE_LENGTH: int = 2000
STREAM_EDIT_INTERVAL: float = 1.5
