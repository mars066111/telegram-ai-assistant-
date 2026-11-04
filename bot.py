
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
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

DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite3")

logger = logging.getLogger(__name__)

# -----------------------------
# DATABASE
# -----------------------------

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row


def init_db():
    db.execute("""
        CREATE TABLE IF NOT EXISTS owner_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT DEFAULT '',
            age TEXT DEFAULT '',
            city TEXT DEFAULT '',
            about TEXT DEFAULT '',
            interests TEXT DEFAULT ''
        )
    """)

    db.execute("""
        INSERT OR IGNORE INTO owner_profile (id)
        VALUES (1)
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS people (
            chat_key TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            age TEXT DEFAULT '',
            city TEXT DEFAULT '',
            interests TEXT DEFAULT '',
            about TEXT DEFAULT ''
        )
    """)

    db.commit()


def get_profile():
    row = db.execute(
        "SELECT * FROM owner_profile WHERE id = 1"
    ).fetchone()

    return dict(row) if row else {}


def set_profile_field(field, value):
    allowed = {
        "name", "age", "city", "about", "interests"
    }

    if field not in allowed:
        return

    db.execute(
        f"UPDATE owner_profile SET {field} = ? WHERE id = 1",
        (value[:1000],),
    )
    db.commit()


def get_person(chat_key):
    row = db.execute(
        "SELECT * FROM people WHERE chat_key = ?",
        (chat_key,),
    ).fetchone()

    if row:
        return dict(row)

    db.execute(
        "INSERT OR IGNORE INTO people (chat_key) VALUES (?)",
        (chat_key,),
    )
    db.commit()

    return {
        "name": "",
        "age": "",
        "city": "",
        "interests": "",
        "about": "",
    }


def update_person(chat_key, facts):
    allowed = {"name", "age", "city", "interests", "about"}

    for field, value in facts.items():
        if field not in allowed:
            continue

        if not isinstance(value, str):
            continue

        value = value.strip()[:500]

        if not value:
            continue

        db.execute(
            f"UPDATE people SET {field} = ? WHERE chat_key = ?",
            (value, chat_key),
        )

    db.commit()


init_db()


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

    Thread(
        target=server.serve_forever,
        daemon=True,
    ).start()

    logger.info("health server started on port %s", port)


# -----------------------------
# AI PROMPT
# -----------------------------

SYSTEM_PROMPT = """
Ты помогаешь владельцу аккаунта составлять естественные ответы
для переписки в Telegram.

Используй только достоверные сведения из анкеты владельца.
Не придумывай ему имя, возраст, город, профессию, события или биографию.
Если информации нет, не выдумывай её.

О владельце:
{owner_profile}

О собеседнике:
{person_profile}

ПРАВИЛА ОБЩЕНИЯ:

- Отвечай на русском языке.
- Пиши естественно, как в обычном Telegram.
- Обычно используй 1–4 предложения.
- Не будь сухим и шаблонным.
- Реагируй на конкретные детали сообщения.
- Иногда задавай уместные вопросы, чтобы поддержать разговор.
- Не превращай беседу в допрос.
- Не заканчивай каждое сообщение вопросом.
- Делай искренние комплименты, когда они действительно уместны.
- Не повторяй одинаковые реакции и фразы.
- Не злоупотребляй эмодзи.
- Не используй случайные китайские или другие иностранные символы.
- Не выдумывай факты о собеседнике.
- Не раскрывай всю анкету владельца, если спросили только об одном факте.
- Если спросили имя, отвечай только именем.
- Если спросили город, отвечай только городом.
- Если нужного факта нет, честно скажи, что не знаешь.
- Учитывай историю разговора.
- Уважай личные границы собеседника.
- Не вводи собеседника в заблуждение, если он прямо спрашивает об автоматизации.

Выдавай только готовый ответ, без пояснений и кавычек.
"""


def format_profile(profile):
    fields = {
        "name": "Имя",
        "age": "Возраст",
        "city": "Город",
        "about": "О себе",
        "interests": "Интересы",
    }

    return "\n".join(
        f"{label}: {profile.get(key) or 'не указано'}"
        for key, label in fields.items()
    )


def make_prompt(chat_key):
    owner = get_profile()
    person = get_person(chat_key)

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(owner),
        person_profile=format_profile(person),
    )


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
    return min(
        10,
        BASE_REPLY_DELAY_SECONDS + len(answer) // 120,
    )


# -----------------------------
# AUTOMATIC FACT EXTRACTION
# -----------------------------

async def remember_person(chat_key, message, client):
    """
    Извлекает только сведения, которые собеседник
    явно сообщил о себе.
    """

    extraction_prompt = """
Извлеки из сообщения сведения, которые человек явно сообщил О СЕБЕ.

Верни только JSON следующего формата:
{
  "name": "",
  "age": "",
  "city": "",
  "interests": "",
  "about": ""
}

Правила:
- Не делай предположений.
- Не записывай сведения о других людях.
- Не записывай пароли, адреса, телефоны, контакты,
  медицинские и другие чувствительные данные.
- Не заполняй поле, если факт неизвестен.
- Если поле не упоминается, оставь пустую строку.
- Не считай вопрос или предположение фактом.
"""

    try:
        result = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": extraction_prompt,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            temperature=0,
            max_tokens=180,
        )

        content = result.choices[0].message.content

        if not content:
            return

        content = content.strip()

        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
        )

        facts = json.loads(content)

        if isinstance(facts, dict):
            update_person(chat_key, facts)

    except Exception:
        logger.exception("memory extraction error")


# -----------------------------
# PRIVATE COMMANDS
# -----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text(
            "привет! я готов помогать с перепиской\n\n"
            "команды для анкеты владельца:\n"
            "/profile — посмотреть анкету\n"
            "/setname — изменить имя\n"
            "/setage — изменить возраст\n"
            "/setcity — изменить город\n"
            "/setabout — рассказать о себе\n"
            "/setinterests — указать интересы"
        )


async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    data = get_profile()

    await update.message.reply_text(
        "анкета владельца:\n\n"
        f"имя: {data.get('name') or 'не указано'}\n"
        f"возраст: {data.get('age') or 'не указано'}\n"
        f"город: {data.get('city') or 'не указано'}\n"
        f"о себе: {data.get('about') or 'не указано'}\n"
        f"интересы: {data.get('interests') or 'не указано'}"
    )


async def set_field(update, context, field, label):
    if not update.message:
        return

    value = " ".join(context.args).strip()

    if not value:
        await update.message.reply_text(
            f"напиши так: /{field} значение"
        )
        return

    set_profile_field(field, value)

    await update.message.reply_text(
        f"{label} сохранено"
    )


async def setname(update, context):
    await set_field(update, context, "name", "имя")


async def setage(update, context):
    await set_field(update, context, "age", "возраст")


async def setcity(update, context):
    await set_field(update, context, "city", "город")


async def setabout(update, context):
    await set_field(update, context, "about", "информация о себе")


async def setinterests(update, context):
    await set_field(update, context, "interests", "интересы")


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
    chat_key = f"private:{user.id}"

    async with get_lock(key):
        history = get_history(key)

        history.append({
            "role": "user",
            "content": message.text,
        })

        try:
            client = context.application.bot_data["groq_client"]

            await remember_person(
                chat_key,
                message.text,
                client,
            )

            result = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": make_prompt(chat_key),
                    },
                    *list(history),
                ],
                temperature=0.9,
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
    chat_key = f"{batch.connection_id}:{batch.messages[-1].chat.id}"

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

            # Сохраняем факты отдельно для каждого чата.
            await remember_person(
                chat_key,
                incoming_text,
                client,
            )

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
                    {
                        "role": "system",
                        "content": make_prompt(chat_key),
                    },
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

        db.close()

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
        CommandHandler("start", start, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("clear", clear, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("profile", profile, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("setname", setname, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("setage", setage, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("setcity", setcity, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("setabout", setabout, filters=private_filter)
    )

    application.add_handler(
        CommandHandler(
            "setinterests",
            setinterests,
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
