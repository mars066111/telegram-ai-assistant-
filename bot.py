from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import deque
from dataclasses import dataclass

from openai import AsyncOpenAI
from telegram import BusinessConnection, Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    BusinessConnectionHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

HISTORY_LIMIT = 30
REPLY_DELAY_MIN_SECONDS = 2
REPLY_DELAY_MAX_SECONDS = 3

SYSTEM_PROMPT = """
ты помогаешь вести естественную переписку с девушками.

СТИЛЬ:
- всегда пиши только маленькими буквами.
- пиши естественно, как обычный человек в telegram.
- будь уверенным, непринуждённым и живым.
- используй юмор, лёгкие подколы и флирт.
- не пиши сухо, официально или как искусственный интеллект.
- не используй книжные выражения и шаблонные фразы.

ИНИЦИАТИВА:
- не ограничивайся короткой реакцией на сообщение.
- развивай разговор и предлагай интересные темы.
- задавай уместные вопросы, если это помогает продолжить общение.
- не превращай разговор в допрос.
- делай искренние комплименты, когда для них есть повод.
- если девушка делает комплимент, можешь ответить взаимностью.
- если девушка шутит, поддержи её шутку.
- если девушка флиртует, отвечай лёгким флиртом.
- не повторяй уже заданные вопросы.
- не навязывайся, если девушка не заинтересована.

КОНТЕКСТ:
- учитывай всю историю переписки.
- сообщения девушки и владельца аккаунта должны различаться.
- не выдумывай факты о девушке.
- не выдавай предположения за известные факты.

ФОРМАТ:
- один готовый ответ.
- без заголовков, кавычек и пояснений.
- не начинай каждое сообщение с приветствия.
- только строчные буквы.
"""

logger = logging.getLogger(__name__)

histories: dict[tuple[str, str], deque] = {}
locks: dict[tuple[str, str], asyncio.Lock] = {}
pending: dict[tuple[str, str], asyncio.Task] = {}


def get_history(key):
    if key not in histories:
        histories[key] = deque(maxlen=HISTORY_LIMIT)
    return histories[key]


def get_lock(key):
    if key not in locks:
        locks[key] = asyncio.Lock()
    return locks[key]


def lowercase(text: str) -> str:
    return text.strip().lower()


def get_key(connection_id: str, chat_id: int):
    return ("business", f"{connection_id}:{chat_id}")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value.strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text(
            "привет, скидывай сообщение — помогу придумать ответ"
        )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user

    if user is None or update.message is None:
        return

    key = ("private", str(user.id))
    histories.pop(key, None)

    await update.message.reply_text("историю очистил")


async def handle_private(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message

    if message is None or not message.text:
        return

    user = update.effective_user

    if user is None:
        return

    key = ("private", str(user.id))

    async with get_lock(key):
        history = get_history(key)
        history.append({
            "role": "user",
            "content": message.text,
        })

        try:
            client = context.application.bot_data["groq_client"]

            result = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *list(history),
                ],
                temperature=0.9,
                max_tokens=256,
            )

            answer = result.choices[0].message.content

            if not answer:
                await message.reply_text("не получилось придумать ответ")
                return

            answer = lowercase(answer)

            history.append({
                "role": "assistant",
                "content": answer,
            })

            await message.reply_text(answer)

        except Exception as error:
            logger.error("private response error: %s", type(error).__name__)
            await message.reply_text("не получилось получить ответ")


async def handle_business_connection(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    connection = update.business_connection

    if connection is None:
        return

    connections = context.application.bot_data.setdefault(
        "business_connections", {}
    )

    if connection.is_enabled:
        connections[connection.id] = connection
    else:
        connections.pop(connection.id, None)


async def get_connection(connection_id, context):
    connections = context.application.bot_data.setdefault(
        "business_connections", {}
    )

    connection = connections.get(connection_id)

    if connection is not None:
        return connection if connection.is_enabled else None

    try:
        connection = await context.bot.get_business_connection(
            connection_id
        )
    except Exception as error:
        logger.warning("connection error: %s", type(error).__name__)
        return None

    if connection.is_enabled:
        connections[connection.id] = connection
        return connection

    return None


@dataclass
class Batch:
    key: tuple[str, str]
    connection_id: str
    messages: list[Message]
    task: asyncio.Task | None = None


async def handle_business(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.business_message or update.edited_business_message

    if (
        message is None
        or not message.text
        or not message.business_connection_id
        or message.sender_business_bot is not None
    ):
        return

    connection_id = message.business_connection_id

    connection = await get_connection(connection_id, context)

    if connection is None:
        return

    key = get_key(connection_id, message.chat.id)

    # Сохраняем реальные сообщения владельца аккаунта.
    # На собственные сообщения бот не отвечает.
    if (
        message.from_user is not None
        and message.from_user.id == connection.user.id
    ):
        async with get_lock(key):
            get_history(key).append({
                "role": "assistant",
                "content": message.text,
            })
        return

    batches = context.application.bot_data.setdefault(
        "pending_batches", {}
    )

    batch = batches.get(key)

    if batch is None:
        batch = Batch(
            key=key,
            connection_id=connection_id,
            messages=[],
        )
        batches[key] = batch

    batch.messages.append(message)

    if batch.task is not None:
        batch.task.cancel()

    batch.task = context.application.create_task(
        process_batch(batch, context)
    )


async def process_batch(
    batch: Batch,
    context: ContextTypes.DEFAULT_TYPE,
):
    try:
        await asyncio.sleep(
            random.uniform(
                REPLY_DELAY_MIN_SECONDS,
                REPLY_DELAY_MAX_SECONDS,
            )
        )

        batches = context.application.bot_data.get(
            "pending_batches", {}
        )

        if batches.get(batch.key) is not batch:
            return

        batches.pop(batch.key, None)

        connection = await get_connection(
            batch.connection_id,
            context,
        )

        if connection is None:
            return

        incoming = [
            message for message in batch.messages
            if message.text
            and message.sender_business_bot is None
            and (
                message.from_user is None
                or message.from_user.id != connection.user.id
            )
        ]

        if not incoming:
            return

        combined_text = "\n".join(
            message.text for message in incoming
        )

        await generate_business_reply(
            batch,
            combined_text,
            context,
        )

    except asyncio.CancelledError:
        return

    except Exception as error:
        logger.error("batch error: %s", type(error).__name__)


async def generate_business_reply(
    batch: Batch,
    incoming_text: str,
    context: ContextTypes.DEFAULT_TYPE,
):
    key = batch.key

    async with get_lock(key):
        history = get_history(key)

        history.append({
            "role": "user",
            "content": incoming_text,
        })

        try:
            connection = await get_connection(
                batch.connection_id,
                context,
            )

            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]

            try:
                await context.bot.send_chat_action(
                    chat_id=batch.messages[-1].chat.id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch.connection_id,
                )
            except Exception:
                pass

            result = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *list(history),
                ],
                temperature=0.9,
                max_tokens=256,
            )

            answer = result.choices[0].message.content

            if not answer or not answer.strip():
                history.pop()
                return

            answer = lowercase(answer)

            # Отправка от имени подключённого Business-аккаунта.
            await context.bot.send_message(
                chat_id=batch.messages[-1].chat.id,
                text=answer,
                business_connection_id=batch.connection_id,
            )

            history.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:
            history.pop()
            logger.error(
                "business reply error: %s",
                type(error).__name__,
            )


async def handle_error(update, context):
    error = context.error

    logger.error(
        "telegram error: %s",
        type(error).__name__ if error else "unknown",
    )


def main():
    telegram_token = required_env("TELEGRAM_BOT_TOKEN")
    groq_api_key = required_env("GROQ_API_KEY")

    async def initialize(application):
        application.bot_data["groq_client"] = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=GROQ_BASE_URL,
            timeout=45,
            max_retries=1,
        )

    async def shutdown(application):
        client = application.bot_data.get("groq_client")

        if client:
            await client.close()

    application = (
        Application.builder()
        .token(telegram_token)
        .post_init(initialize)
        .post_shutdown(shutdown)
        .build()
    )

    private_filter = (
        filters.UpdateType.MESSAGE
        & filters.ChatType.PRIVATE
    )

    application.add_handler(
        BusinessConnectionHandler(handle_business_connection)
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            "clear",
            clear,
            filters=private_filter,
        )
    )

    application.add_handler(
        MessageHandler(
            private_filter
            & filters.TEXT
            & ~filters.COMMAND,
            handle_private,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.UpdateType.BUSINESS_MESSAGES
            & filters.ChatType.PRIVATE
            & filters.TEXT
            & ~filters.COMMAND,
            handle_business,
        )
    )

    application.add_error_handler(handle_error)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    logger.info("starting telegram assistant")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()