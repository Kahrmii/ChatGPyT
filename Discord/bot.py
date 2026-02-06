import logging
import time
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from anthropic import (
    AsyncAnthropic,
    APIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    BadRequestError,
)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("claude-bot")

conversation_history: dict[int, list[dict[str, str]]] = {}
user_system_prompts: dict[int, str] = {}
user_models: dict[int, str] = {}

anthropic_client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


def get_history(user_id: int) -> list[dict[str, str]]:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return conversation_history[user_id]

def add_to_history(user_id: int, role: str, content: str) -> None:
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    max_messages = config.MAX_HISTORY_LENGTH * 2
    if len(history) > max_messages:
        conversation_history[user_id] = history[-max_messages:]

def get_system_prompt(user_id: int) -> str:
    return user_system_prompts.get(user_id, config.DEFAULT_SYSTEM_PROMPT)

def get_model(user_id: int) -> str:
    return user_models.get(user_id, config.DEFAULT_MODEL)

def truncate_for_discord(text: str) -> str:
    if len(text) <= config.DISCORD_MAX_MESSAGE_LENGTH:
        return text
    truncated = text[: config.DISCORD_MAX_MESSAGE_LENGTH - 20]
    return truncated + "\n\n*[truncated...]*"

def handle_api_error(error: Exception) -> str:
    if isinstance(error, AuthenticationError):
        logger.error("Anthropic API authentication failed")
        return "Configuration error: Invalid API key. Please contact the bot administrator."
    elif isinstance(error, RateLimitError):
        logger.warning("Anthropic API rate limit hit")
        return "Claude is receiving too many requests right now. Please try again in a moment."
    elif isinstance(error, BadRequestError):
        logger.error(f"Bad request to Anthropic API: {error}")
        return "Something went wrong with the request. Try resetting your conversation with `/reset`."
    elif isinstance(error, APIConnectionError):
        logger.error(f"Connection error to Anthropic API: {error}")
        return "Unable to connect to Claude's API. Please try again later."
    elif isinstance(error, APIError):
        logger.error(f"Anthropic API error: {error}")
        return "An API error occurred. Please try again later."
    else:
        logger.error(f"Unexpected error: {error}", exc_info=True)
        return "An unexpected error occurred. Please try again later."


class ClaudeBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Slash commands synced globally.")

    async def on_ready(self) -> None:
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

bot = ClaudeBot()


async def stream_claude_response(interaction: discord.Interaction, user_message: str) -> None:
    user_id = interaction.user.id
    model = get_model(user_id)
    system_prompt = get_system_prompt(user_id)
    add_to_history(user_id, "user", user_message)
    message = await interaction.followup.send("*Thinking...*", wait=True)
    accumulated_text = ""
    last_edit_time = 0.0
    try:
        async with anthropic_client.messages.stream(
            model=model,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=get_history(user_id),
        ) as stream:
            async for text in stream.text_stream:
                accumulated_text += text
                current_time = time.monotonic()
                if current_time - last_edit_time >= config.STREAM_EDIT_INTERVAL:
                    display_text = truncate_for_discord(accumulated_text)
                    await message.edit(content=display_text)
                    last_edit_time = current_time
        final_text = truncate_for_discord(accumulated_text)
        await message.edit(content=final_text)
        add_to_history(user_id, "assistant", accumulated_text)
    except Exception as e:
        error_msg = handle_api_error(e)
        await message.edit(content=error_msg)
        history = get_history(user_id)
        if history and history[-1]["role"] == "user":
            history.pop()


@bot.tree.command(name="ask", description="Ask Claude a question")
@app_commands.describe(message="Your message to Claude")
async def ask(interaction: discord.Interaction, message: str) -> None:
    await interaction.response.defer()
    await stream_claude_response(interaction, message)

@bot.tree.command(name="reset", description="Reset your conversation history with Claude")
async def reset(interaction: discord.Interaction) -> None:
    user_id = interaction.user.id
    cleared = user_id in conversation_history and len(conversation_history.get(user_id, [])) > 0
    conversation_history.pop(user_id, None)
    if cleared:
        await interaction.response.send_message(
            "Your conversation history has been cleared.", ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "You don't have any conversation history to clear.", ephemeral=True
        )

@bot.tree.command(name="system", description="Set a custom system prompt for Claude")
@app_commands.describe(prompt="The system prompt to use (leave empty to reset to default)")
async def system(interaction: discord.Interaction, prompt: Optional[str] = None) -> None:
    user_id = interaction.user.id
    if prompt is None:
        user_system_prompts.pop(user_id, None)
        await interaction.response.send_message(
            "System prompt reset to default.", ephemeral=True
        )
    else:
        user_system_prompts[user_id] = prompt
        display = prompt[:100] + "..." if len(prompt) > 100 else prompt
        await interaction.response.send_message(
            f"System prompt updated to:\n> {display}", ephemeral=True
        )

@bot.tree.command(name="model", description="Switch the Claude model")
@app_commands.describe(model_name="The Claude model to use")
@app_commands.choices(model_name=[
    app_commands.Choice(name="Claude Sonnet 4.5", value="claude-sonnet-4-5-20250929"),
    app_commands.Choice(name="Claude Opus 4.5", value="claude-opus-4-5-20250514"),
    app_commands.Choice(name="Claude Haiku 3.5", value="claude-haiku-3-5-20241022"),
])
async def model(interaction: discord.Interaction, model_name: app_commands.Choice[str]) -> None:
    user_id = interaction.user.id
    user_models[user_id] = model_name.value
    await interaction.response.send_message(
        f"Model switched to **{model_name.name}**.", ephemeral=True
    )

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    logger.error(f"Command error: {error}", exc_info=True)
    msg = "An error occurred while processing your command."
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)

if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN not set. Copy .env.example to .env and fill in your token.")
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in your key.")
    bot.run(config.DISCORD_TOKEN)
