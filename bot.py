
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    BusinessConnectionHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

HISTORY_LIMIT = 40
BATCH_WAIT_SECONDS = 1.0
DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite3")

logger = logging.getLogger(__name__)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row


# =====================================
# DATABASE
# =====================================

def init_db():
    with db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS owner_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT DEFAULT '',
                age TEXT DEFAULT '',
                city TEXT DEFAULT '',
                about TEXT DEFAULT '',
                interests TEXT DEFAULT '',
                letter_case TEXT DEFAULT 'lower'
            )
        """)

        db.execute("""
            INSERT OR IGNORE INTO owner_profile (id)
            VALUES (1)
        """)

        columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(owner_profile)"
            )
        }

        if "letter_case" not in columns:
            db.execute("""
                ALTER TABLE owner_profile
                ADD COLUMN letter_case TEXT DEFAULT 'lower'
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

        db.execute("""
            CREATE TABLE IF NOT EXISTS message_memory (
                chat_key TEXT PRIMARY KEY,
                facts TEXT DEFAULT '{}'
            )
        """)


def get_profile():
    row = db.execute(
        "SELECT * FROM owner_profile WHERE id = 1"
    ).fetchone()

    return dict(row) if row else {}


def set_profile_field(field, value):
    allowed = {
        "name", "age", "city", "about",
        "interests", "letter_case"
    }

    if field not in allowed:
        return

    with db:
        db.execute(
            f"UPDATE owner_profile SET {field} = ? WHERE id = 1",
            (value[:1000],),
        )


def get_person(chat_key):
    row = db.execute(
        "SELECT * FROM people WHERE chat_key = ?",
        (chat_key,),
    ).fetchone()

    if row:
        return dict(row)

    with db:
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)",
            (chat_key,),
        )

    return {
        "name": "",
        "age": "",
        "city": "",
        "interests": "",
        "about": "",
    }


def update_person(chat_key, facts):
    allowed = {
        "name", "age", "city", "interests", "about"
    }

    with db:
        for field, value in facts.items():
            if field not in allowed:
                continue

            if not isinstance(value, str):
                continue

            value = value.strip()[:500]

            if value:
                db.execute(
                    f"UPDATE people SET {field} = ? WHERE chat_key = ?",
                    (value, chat_key),
                )


def get_extra_memory(chat_key):
    row = db.execute(
        "SELECT facts FROM message_memory WHERE chat_key = ?",
        (chat_key,),
    ).fetchone()

    if not row:
        return {}

    try:
        return json.loads(row["facts"])
    except Exception:
        return {}


def save_extra_memory(chat_key, facts):
    existing = get_extra_memory(chat_key)
    existing.update(facts)

    with db:
        db.execute("""
            INSERT INTO message_memory (chat_key, facts)
            VALUES (?, ?)
            ON CONFLICT(chat_key)
            DO UPDATE SET facts = excluded.facts
        """, (
            chat_key,
            json.dumps(existing, ensure_ascii=False),
        ))


init_db()


# =====================================
# RENDER HEALTH SERVER
# =====================================

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

    logger.info("Health server started on port %s", port)


# =====================================
# REPLY DELAY
# =====================================

def get_reply_delay(answer):
    words = len(answer.split())

    if words <= 5:
        return random.uniform(2, 5)

    if words <= 15:
        return random.uniform(5, 9)

    return random.uniform(10, 15)


# =====================================
# PROMPT
# =====================================

SYSTEM_PROMPT = """
Ты помогаешь владельцу Telegram Business-аккаунта вести переписку.

АНКЕТА ВЛАДЕЛЬЦА:
{owner_profile}

ИНФОРМАЦИЯ О СОБЕСЕДНИЦЕ:
{person_profile}

ДОПОЛНИТЕЛЬНАЯ ПАМЯТЬ:
{extra_memory}

РЕГИСТР:
{letter_case_instruction}

========================
ЯЗЫК
========================

- Если собеседница пишет по-русски, отвечай исключительно по-русски.
- Не вставляй случайные английские слова.
- Не смешивай русский и английский.
- Не используй случайные иностранные фразы.
- Пиши естественно, без машинных формулировок.
- Не используй случайные символы и иероглифы.

========================
ПОЛ СОБЕСЕДНИЦЫ
========================

- Собеседница всегда девушка.
- Всегда обращайся к ней в женском роде.
- Используй формы «ты сказала», «ты хотела», «ты делала», «ты сама».
- Никогда не используй мужской род по отношению к ней.
- Не называй её парнем.
- Следи за родом в прошедшем времени и прилагательных.
- Не выдумывай пол владельца, если он неизвестен.

========================
СТИЛЬ ОБЩЕНИЯ
========================

- Общайся естественно и непринуждённо.
- Не будь сухим, холодным или слишком официальным.
- Не переигрывай с эмоциями.
- Обычно отвечай 1–3 предложениями.
- Не растягивай очевидные мысли.
- Иногда можно ответить подробнее, если тема интересная.
- Используй лёгкий юмор и естественные реакции.
- Не начинай каждое сообщение с «ахах» или «ого».
- Не повторяй одинаковые фразы.

========================
ПРИВЕТСТВИЯ
========================

- Если собеседница пишет просто «привет», просто поздоровайся.
- Не спрашивай автоматически «как дела?».
- Не спрашивай сразу «что делаешь?» или «как настроение?».
- Не превращай приветствие в анкету.
- Если она сама начала тему, подхвати её.
- Если тема пока не появилась, ответь коротко и дружелюбно.

========================
ВОПРОСЫ И ИНИЦИАТИВА
========================

- Проявляй настоящий интерес к собеседнице.
- Регулярно задавай уместные встречные вопросы.
- Не задавай вопросы механически после каждой реплики.
- Один хороший вопрос лучше нескольких случайных.
- Если собеседница рассказывает историю, сначала отреагируй на неё.
- Иногда возвращайся к темам, которые она упоминала раньше.
- Не спрашивай то, что уже известно.
- Не превращай общение в интервью.
- Иногда сам делись подходящей мыслью или шуткой.

========================
КОМПЛИМЕНТЫ
========================

- Делай искренние и уместные комплименты.
- Замечай чувство юмора, характер, сообразительность, интересы и вкус.
- Обращай внимание на интересные детали в её сообщениях.
- Не ограничивайся одинаковыми фразами вроде «ты милая».
- Иногда делай комплимент неожиданно, без отдельного повода.
- Не делай комплименты в каждом сообщении.
- Не выдумывай качества, которых не заметил.
- Не используй чрезмерно сладкие и пафосные фразы.
- Комплимент должен звучать конкретно и естественно.

========================
ЮМОР И ИГРИВОСТЬ
========================

- Иногда шути и слегка подкалывай собеседницу.
- Используй лёгкую иронию, если она подходит к ситуации.
- Можешь отвечать с небольшим озорством.
- Не шути над комплексами и болезненными темами.
- Не навязывай флирт.
- Не превращай каждую реплику в шутку.
- Чувствуй настроение разговора.

========================
ПАМЯТЬ
========================

- Используй только известные факты.
- Не путай людей из разных чатов.
- Не выдумывай биографию собеседницы.
- Не спрашивай повторно то, на что она уже ответила.
- Используй её интересы в подходящем контексте.
- Не перечисляй сохранённые сведения без необходимости.

========================
АНКЕТА ВЛАДЕЛЬЦА
========================

- Анкета владельца — главный источник информации о нём.
- Если имя, возраст или город заполнены, используй их.
- Если спрашивают только возраст — отвечай возрастом.
- Если спрашивают только имя — отвечай именем.
- Не перечисляй всю анкету без просьбы.
- Не выдумывай факты о владельце.

========================
ОБЩИЕ ПРАВИЛА
========================

- Не будь навязчивым.
- Не будь слишком официальным.
- Не используй заезженные фразы.
- Не задавай вопросы просто ради продолжения диалога.
- Не отправляй несколько сообщений подряд без необходимости.
- Выдавай только готовый текст ответа.
- Не добавляй заголовки и пояснения.
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
    profile = get_profile()
    letter_case = profile.get("letter_case", "lower")

    if letter_case == "upper":
        case_instruction = (
            "Начинай обычные предложения с большой буквы. "
            "Имена собственные также пиши с заглавной буквы."
        )
    else:
        case_instruction = (
            "Начинай обычные предложения с маленькой буквы. "
            "Имена собственные всегда пиши с заглавной буквы."
        )

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(profile),
        person_profile=format_profile(get_person(chat_key)),
        extra_memory=json.dumps(
            get_extra_memory(chat_key),
            ensure_ascii=False,
        ),
        letter_case_instruction=case_instruction,
    )


# =====================================
# MEMORY EXTRACTION
# =====================================

async def remember_person(chat_key, message, client):
    extraction_prompt = """
Извлеки факты, которые собеседница явно сообщила о себе.

Верни строго JSON:
{
  "name": "",
  "age": "",
  "city": "",
  "interests": "",
  "about": "",
  "other_facts": []
}

Правила:
- Не делай предположений.
- Не записывай информацию о других людях как информацию о собеседнице.
- Не записывай адреса, телефоны и пароли.
- Неизвестные поля оставляй пустыми.
- other_facts — полезные явно сообщённые факты.
- Не добавляй пояснений вне JSON.
"""

    try:
        result = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=250,
        )

        content = result.choices[0].message.content

        if not content:
            return

        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content.strip(),
        )

        facts = json.loads(content)

        if not isinstance(facts, dict):
            return

        update_person(chat_key, facts)

        other = facts.get("other_facts", [])

        if isinstance(other, list):
            previous = get_extra_memory(chat_key)
            old_facts = previous.get("other_facts", [])

            if not isinstance(old_facts, list):
                old_facts = []

            for fact in other:
                if (
                    isinstance(fact, str)
                    and fact.strip()
                    and fact.strip() not in old_facts
                ):
                    old_facts.append(fact.strip()[:300])

            save_extra_memory(
                chat_key,
                {"other_facts": old_facts[-30:]},
            )

    except Exception:
        logger.exception("Memory extraction error")


# =====================================
# PROFILE COMMANDS
# =====================================

async def start(update, context):
    if update.message:
        await update.message.reply_text(
            "привет! я готов помогать с перепиской\n\n"
            "/profile — посмотреть анкету\n"
            "/name — изменить имя\n"
            "/age — изменить возраст\n"
            "/city — изменить город\n"
            "/about — информация о себе\n"
            "/interests — интересы\n"
            "/caps lower — маленькие буквы\n"
            "/caps upper — обычный регистр\n"
            "/memory ID — посмотреть память\n"
            "/clear — очистить историю"
        )


async def profile(update, context):
    if not update.message:
        return

    data = get_profile()
    letter_case = data.get("letter_case", "lower")

    case_text = (
        "маленькие буквы"
        if letter_case == "lower"
        else "обычный регистр"
    )

    await update.message.reply_text(
        "анкета владельца:\n\n"
        f"имя: {data.get('name') or 'не указано'}\n"
        f"возраст: {data.get('age') or 'не указано'}\n"
        f"город: {data.get('city') or 'не указано'}\n"
        f"о себе: {data.get('about') or 'не указано'}\n"
        f"интересы: {data.get('interests') or 'не указано'}\n"
        f"регистр: {case_text}"
    )


async def set_field(update, context, field, label, command):
    if not update.message:
        return

    value = " ".join(context.args).strip()

    if not value:
        await update.message.reply_text(
            f"использование: /{command} значение"
        )
        return

    set_profile_field(field, value)

    await update.message.reply_text(
        f"{label} сохранено: {value}"
    )


async def setname(update, context):
    await set_field(update, context, "name", "имя", "name")


async def setage(update, context):
    await set_field(update, context, "age", "возраст", "age")


async def setcity(update, context):
    await set_field(update, context, "city", "город", "city")


async def setabout(update, context):
    await set_field(update, context, "about", "информация о себе", "about")


async def setinterests(update, context):
    await set_field(
        update, context, "interests", "интересы", "interests"
    )


async def setcaps(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "выбери режим:\n"
            "/caps lower — маленькие буквы\n"
            "/caps upper — обычный регистр"
        )
        return

    choice = context.args[0].lower()

    if choice not in ("lower", "upper"):
        await update.message.reply_text(
            "используй lower или upper"
        )
        return

    set_profile_field("letter_case", choice)

    answer = (
        "теперь буду писать с маленькой буквы"
        if choice == "lower"
        else "теперь буду писать с заглавной буквы"
    )

    await update.message.reply_text(answer)


async def memory_command(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "использование: /memory ID_чата"
        )
        return

    chat_key = "private:" + context.args[0]
    person = get_person(chat_key)
    extra = get_extra_memory(chat_key)

    await update.message.reply_text(
        "память собеседницы:\n\n"
        f"{format_profile(person)}\n\n"
        f"другие факты: "
        f"{json.dumps(extra, ensure_ascii=False)}"
    )


async def clear(update, context):
    user = update.effective_user

    if user is None or update.message is None:
        return

    key = ("private", str(user.id))
    histories.pop(key, None)

    await update.message.reply_text("историю очистил")


# =====================================
# HISTORY AND RESPONSE FORMATTING
# =====================================

histories = {}
locks = {}


def get_history(key):
    if key not in histories:
        histories[key] = deque(maxlen=HISTORY_LIMIT)

    return histories[key]


def get_lock(key):
    if key not in locks:
        locks[key] = asyncio.Lock()

    return locks[key]


def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"^```.*?\n", "", answer)
    answer = re.sub(r"\n```$", "", answer)

    return answer.strip().strip('"')


def apply_letter_case(answer, mode):
    if not answer or mode == "upper":
        return answer

    # Сохраняем заглавные буквы внутри предложений,
    # чтобы не портить имена и названия.
    result = []
    sentence_start = True

    for char in answer:
        if sentence_start and char.isalpha():
            result.append(char.lower())
            sentence_start = False
        else:
            result.append(char)

        if char in ".!?":
            sentence_start = True

    return "".join(result)


# =====================================
# AI RESPONSE
# =====================================

async def generate_answer(chat_key, history, client):
    result = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": make_prompt(chat_key),
            },
            *list(history),
        ],
        temperature=0.8,
        max_tokens=300,
    )

    answer = result.choices[0].message.content

    if not answer or not answer.strip():
        return None

    answer = clean_answer(answer)

    mode = get_profile().get("letter_case", "lower")

    return apply_letter_case(answer, mode)


# =====================================
# PRIVATE CHAT
# =====================================

async def handle_private(update, context):
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
                chat_key, message.text, client
            )

            answer = await generate_answer(
                chat_key, history, client
            )

            if not answer:
                history.pop()
                await message.reply_text(
                    "не получилось придумать ответ"
                )
                return

            await asyncio.sleep(get_reply_delay(answer))

            history.append({
                "role": "assistant",
                "content": answer,
            })

            await message.reply_text(answer)

        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()

            logger.exception("Private response error")

            await message.reply_text(
                "не получилось получить ответ"
            )


# =====================================
# BUSINESS CONNECTIONS
# =====================================

async def handle_business_connection(update, context):
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
        logger.exception("Connection error")
        return None

    if connection.is_enabled:
        connections[connection.id] = connection
        return connection

    return None


# =====================================
# BUSINESS MESSAGE HANDLER
# =====================================

async def handle_business(update, context):
    message = (
        update.business_message
        or update.edited_business_message
    )

    if (
        message is None
        or not message.text
        or not message.business_connection_id
        or message.sender_business_bot is not None
    ):
        return

    connection_id = message.business_connection_id

    connection = await get_connection(
        connection_id, context
    )

    if connection is None:
        return

    key = ("business", f"{connection_id}:{message.chat.id}")

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
        logger.warning("Could not mark message as read")

    batches = context.application.bot_data.setdefault(
        "pending_batches", {}
    )

    batch = batches.get(key)

    if batch is None:
        batch = {
            "messages": [],
            "task": None,
            "connection_id": connection_id,
            "key": key,
        }

        batches[key] = batch

    batch["messages"].append(message)

    if batch["task"] is not None:
        batch["task"].cancel()

    batch["task"] = context.application.create_task(
        process_batch(batch, context)
    )


# =====================================
# PROCESS MESSAGE BATCH
# =====================================

async def process_batch(batch, context):
    try:
        await asyncio.sleep(BATCH_WAIT_SECONDS)

        batches = context.application.bot_data.get(
            "pending_batches", {}
        )

        if batches.get(batch["key"]) is not batch:
            return

        batches.pop(batch["key"], None)

        connection = await get_connection(
            batch["connection_id"], context
        )

        if connection is None:
            return

        incoming = [
            message
            for message in batch["messages"]
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
            batch, combined_text, context
        )

    except asyncio.CancelledError:
        return

    except Exception:
        logger.exception("Batch error")


# =====================================
# BUSINESS REPLY
# =====================================

async def generate_business_reply(batch, incoming_text, context):
    key = batch["key"]
    chat_id = batch["messages"][-1].chat.id

    chat_key = f"{batch['connection_id']}:{chat_id}"

    async with get_lock(key):
        history = get_history(key)

        history.append({
            "role": "user",
            "content": incoming_text,
        })

        try:
            connection = await get_connection(
                batch["connection_id"], context
            )

            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]

            await remember_person(
                chat_key, incoming_text, client
            )

            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch["connection_id"],
                )
            except Exception:
                pass

            answer = await generate_answer(
                chat_key, history, client
            )

            if not answer:
                history.pop()
                return

            await asyncio.sleep(get_reply_delay(answer))

            await context.bot.send_message(
                chat_id=chat_id,
                text=answer,
                business_connection_id=batch["connection_id"],
            )

            history.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()

            logger.exception("Business reply error")


# =====================================
# ERROR HANDLER
# =====================================

async def handle_error(update, context):
    logger.error(
        "Telegram error",
        exc_info=context.error,
    )


# =====================================
# MAIN
# =====================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    start_health_server()

    telegram_token = os.environ["TELEGRAM_BOT_TOKEN"]
    groq_api_key = os.environ["GROQ_API_KEY"]

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
        CommandHandler("profile", profile, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("clear", clear, filters=private_filter)
    )

    application.add_handler(
        CommandHandler(
            ["name", "setname"],
            setname,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            ["age", "setage"],
            setage,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            ["city", "setcity"],
            setcity,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            ["about", "setabout"],
            setabout,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            ["interests", "setinterests"],
            setinterests,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            "caps",
            setcaps,
            filters=private_filter,
        )
    )

    application.add_handler(
        CommandHandler(
            "memory",
            memory_command,
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

    logger.info("Starting Telegram assistant")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
