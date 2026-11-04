
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from openai import AsyncOpenAI
from telegram import Message, Update
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
BATCH_WAIT_SECONDS = 1.0
BASE_REPLY_DELAY_SECONDS = 4

logger = logging.getLogger(__name__)


# -----------------------------
# HTTP SERVER FOR RENDER
# -----------------------------

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "10000"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    logger.info("health server started on port %s", port)


# -----------------------------
# AI PERSONALITY
# -----------------------------

SYSTEM_PROMPT = """
ТВОЯ ЛИЧНОСТЬ:

Тебя зовут Марсель.
Тебе 15 лет.
Ты живёшь в Люберцах.
Ты парень и говоришь о себе в мужском роде.

Если спрашивают имя — отвечай, что тебя зовут Марсель.
Если спрашивают возраст — говори, что тебе 15.
Если спрашивают город — можешь сказать, что ты из Люберец.

Не называй себя Алексом, Алексеем, Лёшей или Иваном.
Не говори, что тебе 20 или 25 лет.
Не придумывай себе профессию, работу или биографию.
Не перенимай личность из старых сообщений.

ТВОЙ ХАРАКТЕР:

Ты общительный, живой, с чувством юмора.
Ты расслабленный, уверенный, иногда немного дерзкий и ироничный.
Ты не пытаешься понравиться любой ценой.
Ты умеешь поддерживать разговор и проявлять инициативу.
Ты можешь шутить, удивляться и высказывать собственное мнение.

СТИЛЬ:

- всегда пиши маленькими буквами.
- пиши естественно, как в обычном Telegram.
- используй полноценные слова.
- не будь официальным и чрезмерно вежливым.
- не злоупотребляй эмодзи.
- не ставь точку в конце каждого сообщения.
- не используй шаблонные фразы.
- не повторяй одинаковые реакции.
- не пиши как бот поддержки.
- обычно отвечай 1–4 предложениями.
- не растягивай короткие ответы искусственно.

ЖИВОЕ ОБЩЕНИЕ:

- реагируй на конкретные детали сообщения.
- поддерживай шутки собеседницы.
- если она рассказывает историю, покажи, что внимательно прочитал её.
- можешь доброжелательно подшутить.
- иногда делай искренние комплименты, если это уместно.
- задавай интересные вопросы по теме разговора.
- не задавай вопрос в каждом сообщении.
- не превращай разговор в интервью.
- не заканчивай каждое сообщение вопросом.
- иногда сам предлагай интересную тему.
- не повторяй вопросы, на которые уже ответили.
- учитывай настроение собеседницы.
- уважай личные границы и отказ.
- не навязывайся.

КОНТЕКСТ:

- учитывай историю переписки.
- не путай сообщения разных собеседников.
- не выдумывай общие воспоминания, встречи и события.
- не выдавай предположения за факты.
- не перенимай личность из старых ответов.
- если не знаешь ответа о себе, не придумывай его.

ЯЗЫК:

- отвечай на русском языке.
- не переходи на китайский, японский или другие языки без просьбы.
- не вставляй случайные иностранные символы.
- если не понял сообщение, попроси уточнить по-русски.

ПРАВДИВОСТЬ:

- не выдумывай факты о себе или собеседнице.
- не вводи человека в заблуждение, если он прямо спрашивает об автоматизации.
- не манипулируй чувствами собеседницы.

ФОРМАТ:

- выдавай только готовый ответ.
- без заголовков и пояснений.
- без кавычек.
- используй строчные буквы.
- не добавляй подпись в конце.

ГЛАВНОЕ:

Будь живым, интересным собеседником.
Не отвечай сухо и одинаково.
Подстраивайся под контекст конкретного разговора.
"""


# -----------------------------
# MEMORY
# -----------------------------

histories: dict[tuple[str, str], deque] = {}
locks: dict[tuple[str, str], asyncio.Lock] = {}


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


def get_reply_delay(answer: str) -> int:
    extra_delay = len(answer) // 120
    return min(10, BASE_REPLY_DELAY_SECONDS + extra_delay)


# -----------------------------
# PRIVATE COMMANDS
# -----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text(
            "привет, скидывай сообщение"
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
                temperature=0.85,
                max_tokens=320,
            )

            answer = result.choices[0].message.content

            if not answer or not answer.strip():
                history.pop()
                await message.reply_text(
                    "не получилось придумать ответ"
                )
                return

            answer = lowercase(answer)

            await asyncio.sleep(get_reply_delay(answer))

            history.append({
                "role": "assistant",
                "content": answer,
            })

            await message.reply_text(answer)

        except Exception:
            if history and history[-1].get("role") == "user":
                history.pop()

            logger.exception("private response error")

            await message.reply_text(
                "не получилось получить ответ"
            )


# -----------------------------
# BUSINESS CONNECTION
# -----------------------------

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
    except Exception:
        logger.exception("connection error")
        return None

    if connection.is_enabled:
        connections[connection.id] = connection
        return connection

    return None


# -----------------------------
# MESSAGE BATCHING
# -----------------------------

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

    connection = await get_connection(
        connection_id,
        context,
    )

    if connection is None:
        return

    key = get_key(connection_id, message.chat.id)

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

    try:
        await context.bot.read_business_message(
            business_connection_id=connection_id,
            chat_id=message.chat.id,
            message_id=message.message_id,
        )
    except Exception:
        logger.warning("could not mark message as read")

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
        await asyncio.sleep(BATCH_WAIT_SECONDS)

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

    except Exception:
        logger.exception("batch error")


# -----------------------------
# GENERATE BUSINESS REPLY
# -----------------------------

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
                max_tokens=320,
            )

            answer = result.choices[0].message.content

            if not answer or not answer.strip():
                history.pop()
                return

            answer = lowercase(answer)

            await asyncio.sleep(get_reply_delay(answer))

            await context.bot.send_message(
                chat_id=batch.messages[-1].chat.id,
                text=answer,
                business_connection_id=batch.connection_id,
            )

            history.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception:
            if history and history[-1].get("role") == "user":
                history.pop()

            logger.exception("business reply error")


# -----------------------------
# ERROR HANDLER
# -----------------------------

async def handle_error(update, context):
    logger.error(
        "telegram error",
        exc_info=context.error,
    )


# -----------------------------
# MAIN
# -----------------------------

def main():
    start_health_server()

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
