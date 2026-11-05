import asyncio
import json
import logging
import os
import random
import re
import sqlite3
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock, Thread

from openai import AsyncOpenAI
from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    BusinessConnectionHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

# =========================================================
# CONFIG
# =========================================================

MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite3")

HISTORY_LIMIT = 40
BATCH_WAIT_SECONDS = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db_lock = RLock()

# In-memory conversation state
histories = {}
locks = {}


# =========================================================
# DATABASE
# =========================================================

def init_db():
    with db_lock, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS owner_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT DEFAULT '',
                age TEXT DEFAULT '',
                city TEXT DEFAULT '',
                about TEXT DEFAULT '',
                interests TEXT DEFAULT '',
                letter_case TEXT DEFAULT 'lower',
                gender TEXT DEFAULT 'male'
            )
        """)
        db.execute("INSERT OR IGNORE INTO owner_profile (id) VALUES (1)")

        owner_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(owner_profile)")
        }
        if "letter_case" not in owner_columns:
            db.execute(
                "ALTER TABLE owner_profile ADD COLUMN letter_case TEXT DEFAULT 'lower'"
            )
        if "gender" not in owner_columns:
            db.execute(
                "ALTER TABLE owner_profile ADD COLUMN gender TEXT DEFAULT 'male'"
            )

        db.execute("""
            CREATE TABLE IF NOT EXISTS people (
                chat_key TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                age TEXT DEFAULT '',
                city TEXT DEFAULT '',
                interests TEXT DEFAULT '',
                about TEXT DEFAULT '',
                gender TEXT DEFAULT ''
            )
        """)
        person_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(people)")
        }
        if "gender" not in person_columns:
            db.execute("ALTER TABLE people ADD COLUMN gender TEXT DEFAULT ''")

        db.execute("""
            CREATE TABLE IF NOT EXISTS message_memory (
                chat_key TEXT PRIMARY KEY,
                facts TEXT DEFAULT '{}'
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting TEXT PRIMARY KEY,
                value TEXT DEFAULT ''
            )
        """)

        # One-time reset of old interlocutor data. Owner profile is preserved.
        marker = db.execute(
            "SELECT value FROM app_settings WHERE setting = ?",
            ("people_memory_reset_v1",),
        ).fetchone()
        if marker is None:
            db.execute("DELETE FROM people")
            db.execute("DELETE FROM message_memory")
            db.execute(
                "INSERT INTO app_settings (setting, value) VALUES (?, ?)",
                ("people_memory_reset_v1", "done"),
            )
            logger.info("Old people memory was reset")


init_db()


def get_profile():
    with db_lock:
        row = db.execute(
            "SELECT * FROM owner_profile WHERE id = 1"
        ).fetchone()
    return dict(row) if row else {}


def set_profile_field(field, value):
    allowed = {
        "name", "age", "city", "about", "interests", "letter_case", "gender"
    }
    if field not in allowed:
        return
    with db_lock, db:
        db.execute(
            f"UPDATE owner_profile SET {field} = ? WHERE id = 1",
            (str(value)[:1000],),
        )


def get_person(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT * FROM people WHERE chat_key = ?", (chat_key,)
        ).fetchone()
        if row:
            return dict(row)

        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        db.commit()
        row = db.execute(
            "SELECT * FROM people WHERE chat_key = ?", (chat_key,)
        ).fetchone()
        return dict(row) if row else {
            "chat_key": chat_key,
            "name": "",
            "age": "",
            "city": "",
            "interests": "",
            "about": "",
            "gender": "",
        }


def update_person(chat_key, facts):
    allowed = {"name", "age", "city", "interests", "about", "gender"}
    with db_lock, db:
        # Important: create the row before updating it. Otherwise commands
        # for a person who has not messaged yet would silently do nothing.
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        for field, value in facts.items():
            if field not in allowed or not isinstance(value, str):
                continue
            value = value.strip()[:500]
            if value:
                db.execute(
                    f"UPDATE people SET {field} = ? WHERE chat_key = ?",
                    (value, chat_key),
                )


def get_extra_memory(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT facts FROM message_memory WHERE chat_key = ?", (chat_key,)
        ).fetchone()
    if not row:
        return {}
    try:
        data = json.loads(row["facts"])
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def save_extra_memory(chat_key, facts):
    existing = get_extra_memory(chat_key)
    existing.update(facts)
    with db_lock, db:
        db.execute("""
            INSERT INTO message_memory (chat_key, facts)
            VALUES (?, ?)
            ON CONFLICT(chat_key) DO UPDATE SET facts = excluded.facts
        """, (chat_key, json.dumps(existing, ensure_ascii=False)))


def find_person_key(identifier):
    """Find private ID or a unique Business chat ID."""
    identifier = str(identifier).strip()
    with db_lock:
        exact = db.execute(
            "SELECT chat_key FROM people WHERE chat_key = ?",
            (f"private:{identifier}",),
        ).fetchone()
        if exact:
            return exact["chat_key"]
        rows = db.execute("SELECT chat_key FROM people").fetchall()

    matches = [
        row["chat_key"] for row in rows
        if row["chat_key"].endswith(f":{identifier}")
    ]
    return matches[0] if len(matches) == 1 else None


def delete_person(chat_key):
    with db_lock, db:
        db.execute("DELETE FROM people WHERE chat_key = ?", (chat_key,))
        db.execute("DELETE FROM message_memory WHERE chat_key = ?", (chat_key,))


def clear_all_people():
    with db_lock, db:
        db.execute("DELETE FROM people")
        db.execute("DELETE FROM message_memory")


def list_people():
    with db_lock:
        return db.execute(
            "SELECT chat_key, name, gender FROM people ORDER BY chat_key"
        ).fetchall()


# =========================================================
# RENDER HEALTH CHECK
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, fmt, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health server started on port %s", port)


# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
Ты — ИИ-помощник для переписки в Telegram Business.
Не выдавай себя за человека. Если прямо спрашивают, отвечай честно,
что ты ИИ-помощник, работающий автоматически.

АНКЕТА ВЛАДЕЛЬЦА:
{owner_profile}

ТВОЙ ГРАММАТИЧЕСКИЙ РОД:
{bot_gender_instruction}

ИНФОРМАЦИЯ О СОБЕСЕДНИКЕ:
{person_profile}

ГРАММАТИЧЕСКИЙ РОД СОБЕСЕДНИКА:
{person_gender_instruction}

ДОПОЛНИТЕЛЬНАЯ ПАМЯТЬ:
{extra_memory}

РЕГИСТР:
{letter_case_instruction}

СТИЛЬ — NATURAL, ИНИЦИАТИВНОСТЬ — HIGH:
- Общайся естественно, тепло и непринуждённо, без канцелярита.
- Отвечай на конкретное содержание сообщения, замечай детали.
- Развивай интересные темы: иногда высказывай мнение, предлагай тему,
  задавай уместные вопросы и возвращайся к ранее обсуждавшимся планам.
- Не задавай вопрос после каждого сообщения и не превращай чат в интервью.
- Не используй шаблонные «как дела?», «что делаешь?», «что нового?».
- Не отвечай постоянно односложно: «ну», «да», «понятно», «ясно»,
  «бывает», «прикольно», «ахах».
- Не переигрывай с эмоциями, юмором или комплиментами.
- Обычно 1–4 предложения. Простые вопросы — коротко.

ПАМЯТЬ И ФАКТЫ:
- Используй только сведения из профиля, памяти и текущей переписки.
- Не придумывай имя, возраст, город, интересы, события или личный опыт.
- Если имя неизвестно, не используй имя, прозвище или случайное обращение.
- Не переноси факты между разными чатами.
- При противоречиях не делай вид, что уверен; уточни.
- Не перечисляй сохранённые данные без необходимости.

ПОЛ СОБЕСЕДНИКА:
- Используй указанный род. Если он неизвестен, формулируй нейтрально.
- Не угадывай пол по имени или фотографии.

ПОЛ БОТА:
- Соблюдай свой выбранный грамматический род, не путай его с родом собеседника.
- Мужской: «я понял», «я рад», «я заметил».
- Женский: «я поняла», «я рада», «я заметила».

ЯЗЫК И РЕГИСТР:
- Отвечай на языке собеседника; для русского используй естественный русский.
- Не вставляй случайные английские или китайские слова.
- Если включён нижний регистр, начинай обычные предложения со строчной буквы.
  Имена собственные всегда пиши с заглавной буквы.

Выдавай только текст ответа.
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
        f"{label}: {profile.get(key) or 'неизвестно'}"
        for key, label in fields.items()
    )


def make_prompt(chat_key):
    owner = get_profile()
    person = get_person(chat_key)

    if owner.get("gender") == "female":
        bot_gender = "Женский род: говори «я поняла», «я рада», «я заметила»."
    else:
        bot_gender = "Мужской род: говори «я понял», «я рад», «я заметил»."

    gender = person.get("gender", "")
    if gender == "female":
        person_gender = "Женский род собеседника."
    elif gender == "male":
        person_gender = "Мужской род собеседника."
    else:
        person_gender = "Пол собеседника неизвестен; используй нейтральные фразы."

    if owner.get("letter_case") == "upper":
        case_instruction = "Обычный регистр, заглавные буквы в начале предложений."
    else:
        case_instruction = (
            "Начинай обычные предложения со строчной буквы; "
            "имена собственные пиши с заглавной."
        )

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(owner),
        bot_gender_instruction=bot_gender,
        person_profile=format_profile(person),
        person_gender_instruction=person_gender,
        extra_memory=json.dumps(get_extra_memory(chat_key), ensure_ascii=False),
        letter_case_instruction=case_instruction,
    )


# =========================================================
# MEMORY EXTRACTION
# =========================================================

async def remember_person(chat_key, message, client):
    extraction_prompt = """
Извлеки только факты, которые автор сообщения явно сообщил о себе.
Верни строго JSON с полями:
{"name":"","age":"","city":"","interests":"","about":"","other_facts":[]}

Не угадывай имя или пол. Не записывай сведения о других людях как
сведения об авторе. Если факт не подтверждён явно, оставь поле пустым.
other_facts — список важных сведений о себе. Не добавляй текст вне JSON.
"""
    try:
        result = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=300,
        )
        content = result.choices[0].message.content
        if not content:
            return

        content = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content.strip()
        )
        facts = json.loads(content)
        if not isinstance(facts, dict):
            return

        facts.pop("gender", None)
        other = facts.pop("other_facts", [])
        update_person(chat_key, facts)

        if isinstance(other, list):
            previous = get_extra_memory(chat_key)
            old_facts = previous.get("other_facts", [])
            if not isinstance(old_facts, list):
                old_facts = []
            for fact in other:
                if isinstance(fact, str) and fact.strip():
                    fact = fact.strip()[:300]
                    if fact not in old_facts:
                        old_facts.append(fact)
            save_extra_memory(chat_key, {"other_facts": old_facts[-30:]})
    except Exception:
        logger.exception("Memory extraction error")


# =========================================================
# COMMANDS
# =========================================================

async def start(update, context):
    if update.message:
        await update.message.reply_text(
            "привет! я готов помогать с перепиской.\n\n"
            "АНКЕТА:\n"
            "/profile — посмотреть анкету\n"
            "/name значение — имя владельца\n"
            "/age значение — возраст владельца\n"
            "/city значение — город владельца\n"
            "/about значение — информация о себе\n"
            "/interests значение — интересы\n\n"
            "НАСТРОЙКИ:\n"
            "/gender male|female — род бота\n"
            "/caps lower|upper — регистр\n\n"
            "СОБЕСЕДНИКИ:\n"
            "/people — список\n"
            "/person ID — информация\n"
            "/person_name ID имя — задать имя\n"
            "/person_gender ID male|female — задать пол\n"
            "/forget ID — забыть собеседника\n"
            "/forget_all — очистить память всех\n"
            "/clear — очистить историю этого личного чата"
        )


async def profile(update, context):
    if not update.message:
        return
    data = get_profile()
    gender_text = "женский" if data.get("gender") == "female" else "мужской"
    case_text = (
        "маленькие буквы"
        if data.get("letter_case") == "lower"
        else "обычный регистр"
    )
    await update.message.reply_text(
        "анкета владельца:\n\n"
        f"имя: {data.get('name') or 'не указано'}\n"
        f"возраст: {data.get('age') or 'не указано'}\n"
        f"город: {data.get('city') or 'не указано'}\n"
        f"о себе: {data.get('about') or 'не указано'}\n"
        f"интересы: {data.get('interests') or 'не указано'}\n\n"
        f"пол бота: {gender_text}\nрегистр: {case_text}"
    )


async def set_field(update, context, field, label, command):
    if not update.message:
        return
    value = " ".join(context.args).strip()
    if not value:
        await update.message.reply_text(f"использование: /{command} значение")
        return
    set_profile_field(field, value)
    await update.message.reply_text(f"{label} сохранено: {value}")


async def setname(update, context):
    await set_field(update, context, "name", "имя", "name")


async def setage(update, context):
    await set_field(update, context, "age", "возраст", "age")


async def setcity(update, context):
    await set_field(update, context, "city", "город", "city")


async def setabout(update, context):
    await set_field(update, context, "about", "информация", "about")


async def setinterests(update, context):
    await set_field(update, context, "interests", "интересы", "interests")


async def setcaps(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text(
            "/caps lower — маленькие буквы\n/caps upper — обычный регистр"
        )
        return
    choice = context.args[0].lower()
    if choice not in ("lower", "upper"):
        await update.message.reply_text("используй lower или upper")
        return
    set_profile_field("letter_case", choice)
    await update.message.reply_text("настройка регистра сохранена")


async def setgender(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text(
            "/gender male — мужской\n/gender female — женский"
        )
        return
    choice = context.args[0].lower()
    if choice in ("male", "мужской", "м"):
        gender, answer = "male", "выбран мужской род"
    elif choice in ("female", "женский", "ж"):
        gender, answer = "female", "выбран женский род"
    else:
        await update.message.reply_text("используй /gender male или /gender female")
        return
    set_profile_field("gender", gender)
    await update.message.reply_text(answer)


async def people_command(update, context):
    if not update.message:
        return
    rows = list_people()
    if not rows:
        await update.message.reply_text("сохранённых собеседников пока нет")
        return
    lines = ["сохранённые собеседники:"]
    for row in rows[:50]:
        lines.append(f"{row['chat_key']} — {row['name'] or 'имя неизвестно'}")
    await update.message.reply_text("\n".join(lines))


async def person_command(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("использование: /person ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await update.message.reply_text(
            "собеседник с таким ID не найден. Сначала дождись его сообщения."
        )
        return
    person = get_person(key)
    extra = get_extra_memory(key)
    gender_text = {
        "male": "мужской", "female": "женский"
    }.get(person.get("gender"), "не выбран")
    await update.message.reply_text(
        f"ID: {key}\n\n"
        f"имя: {person.get('name') or 'неизвестно'}\n"
        f"возраст: {person.get('age') or 'неизвестно'}\n"
        f"город: {person.get('city') or 'неизвестно'}\n"
        f"интересы: {person.get('interests') or 'неизвестно'}\n"
        f"о себе: {person.get('about') or 'неизвестно'}\n"
        f"пол: {gender_text}\n\n"
        f"дополнительные факты:\n{json.dumps(extra, ensure_ascii=False)}"
    )


async def person_name_command(update, context):
    if not update.message:
        return
    if len(context.args) < 2:
        await update.message.reply_text("использование: /person_name ID имя")
        return
    identifier = context.args[0]
    name = " ".join(context.args[1:]).strip()
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"name": name})
    await update.message.reply_text(f"имя собеседника сохранено: {name}")


async def person_gender_command(update, context):
    if not update.message:
        return
    if len(context.args) < 2:
        await update.message.reply_text(
            "использование: /person_gender ID male\n"
            "или /person_gender ID female"
        )
        return
    identifier, choice = context.args[0], context.args[1].lower()
    if choice in ("male", "мужской", "м"):
        gender, answer = "male", "мужской пол сохранён"
    elif choice in ("female", "женский", "ж"):
        gender, answer = "female", "женский пол сохранён"
    else:
        await update.message.reply_text("выбери male или female")
        return
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"gender": gender})
    await update.message.reply_text(answer)


async def forget_command(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("использование: /forget ID")
        return
    identifier = context.args[0]
    key = find_person_key(identifier)
    if not key:
        await update.message.reply_text("собеседник с таким ID не найден")
        return
    delete_person(key)
    histories.pop(("business", key), None)
    histories.pop(("private", identifier), None)
    await update.message.reply_text("данные этого собеседника удалены")


async def forget_all_command(update, context):
    if not update.message:
        return
    clear_all_people()
    histories.clear()
    await update.message.reply_text(
        "память всех собеседников очищена. "
        "анкета владельца и настройки бота сохранены."
    )


async def clear(update, context):
    """Clear only the in-memory history for the current private user."""
    user = update.effective_user
    if user is None or update.effective_message is None:
        return
    histories.pop(("private", str(user.id)), None)
    await update.effective_message.reply_text("история этого чата очищена")


# =========================================================
# HISTORY, DELAYS, AI
# =========================================================

def get_history(key):
    if key not in histories:
        histories[key] = deque(maxlen=HISTORY_LIMIT)
    return histories[key]


def get_lock(key):
    if key not in locks:
        locks[key] = asyncio.Lock()
    return locks[key]


def get_reply_delay(answer):
    words = len(answer.split())
    if words <= 5:
        return random.uniform(2, 5)
    if words <= 15:
        return random.uniform(5, 9)
    return random.uniform(10, 15)


def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"^```.*?\n", "", answer)
    answer = re.sub(r"\n```$", "", answer)
    return answer.strip().strip('"')


def apply_letter_case(answer, mode):
    if not answer or mode == "upper":
        return answer
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


async def generate_answer(chat_key, history, client):
    result = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": make_prompt(chat_key)},
            *list(history),
        ],
        temperature=0.9,
        max_tokens=450,
    )
    answer = result.choices[0].message.content
    if not answer or not answer.strip():
        return None
    answer = clean_answer(answer)
    return apply_letter_case(
        answer, get_profile().get("letter_case", "lower")
    )


# =========================================================
# PRIVATE CHATS
# =========================================================

async def handle_private(update, context):
    message = update.effective_message
    user = update.effective_user
    if message is None or not message.text or user is None:
        return

    history_key = ("private", str(user.id))
    chat_key = f"private:{user.id}"

    async with get_lock(history_key):
        history = get_history(history_key)
        history.append({"role": "user", "content": message.text})
        try:
            client = context.application.bot_data["groq_client"]
            await remember_person(chat_key, message.text, client)
            answer = await generate_answer(chat_key, history, client)
            if not answer:
                history.pop()
                await message.reply_text("не получилось сформировать ответ")
                return

            await asyncio.sleep(get_reply_delay(answer))
            history.append({"role": "assistant", "content": answer})
            await message.reply_text(answer)
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Private response error")
            await message.reply_text("не получилось получить ответ")


# =========================================================
# TELEGRAM BUSINESS
# =========================================================

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
        connection = await context.bot.get_business_connection(connection_id)
    except Exception:
        logger.exception("Business connection lookup failed")
        return None
    if connection.is_enabled:
        connections[connection.id] = connection
        return connection
    return None


async def handle_business(update, context):
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
        logger.warning("Business connection is unavailable or disabled")
        return

    chat_key = f"{connection_id}:{message.chat.id}"
    history_key = ("business", chat_key)

    if (
        message.from_user is not None
        and message.from_user.id == connection.user.id
    ):
        async with get_lock(history_key):
            get_history(history_key).append(
                {"role": "assistant", "content": message.text}
            )
        return

    try:
        await context.bot.read_business_message(
            business_connection_id=connection_id,
            chat_id=message.chat.id,
            message_id=message.message_id,
        )
    except Exception:
        logger.debug("Could not mark Business message as read", exc_info=True)

    batches = context.application.bot_data.setdefault("pending_batches", {})
    batch = batches.get(history_key)
    if batch is None:
        batch = {
            "messages": [],
            "task": None,
            "connection_id": connection_id,
            "key": history_key,
            "chat_key": chat_key,
        }
        batches[history_key] = batch

    batch["messages"].append(message)
    if batch["task"] is not None:
        batch["task"].cancel()
    batch["task"] = context.application.create_task(
        process_batch(batch, context)
    )


async def process_batch(batch, context):
    try:
        await asyncio.sleep(BATCH_WAIT_SECONDS)
        batches = context.application.bot_data.get("pending_batches", {})
        if batches.get(batch["key"]) is not batch:
            return
        batches.pop(batch["key"], None)

        connection = await get_connection(batch["connection_id"], context)
        if connection is None:
            return

        incoming = [
            msg for msg in batch["messages"]
            if msg.text
            and msg.sender_business_bot is None
            and (
                msg.from_user is None
                or msg.from_user.id != connection.user.id
            )
        ]
        if not incoming:
            return

        await generate_business_reply(
            batch,
            "\n".join(msg.text for msg in incoming),
            context,
        )
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Business batch processing error")


async def generate_business_reply(batch, incoming_text, context):
    history_key = batch["key"]
    chat_id = batch["messages"][-1].chat.id
    chat_key = batch["chat_key"]

    async with get_lock(history_key):
        history = get_history(history_key)
        history.append({"role": "user", "content": incoming_text})
        try:
            connection = await get_connection(batch["connection_id"], context)
            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]
            await remember_person(chat_key, incoming_text, client)

            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch["connection_id"],
                )
            except Exception:
                pass

            answer = await generate_answer(chat_key, history, client)
            if not answer:
                history.pop()
                return

            await asyncio.sleep(get_reply_delay(answer))
            await context.bot.send_message(
                chat_id=chat_id,
                text=answer,
                business_connection_id=batch["connection_id"],
            )
            history.append({"role": "assistant", "content": answer})
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Business reply error")


# =========================================================
# ERRORS AND STARTUP
# =========================================================

async def handle_error(update, context):
    logger.error("Unhandled Telegram error", exc_info=context.error)


def main():
    start_health_server()

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not telegram_token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is missing")
    if not groq_api_key:
        raise RuntimeError("Environment variable GROQ_API_KEY is missing")

    async def initialize(application):
        application.bot_data["groq_client"] = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=GROQ_BASE_URL,
            timeout=45,
            max_retries=1,
        )
        commands = [
            BotCommand("start", "показать команды"),
            BotCommand("profile", "анкета владельца"),
            BotCommand("name", "изменить имя владельца"),
            BotCommand("age", "изменить возраст"),
            BotCommand("city", "изменить город"),
            BotCommand("about", "информация о владельце"),
            BotCommand("interests", "интересы владельца"),
            BotCommand("gender", "пол бота"),
            BotCommand("caps", "регистр текста"),
            BotCommand("people", "список собеседников"),
            BotCommand("person", "информация по ID"),
            BotCommand("person_name", "имя собеседника по ID"),
            BotCommand("person_gender", "пол собеседника по ID"),
            BotCommand("forget", "забыть собеседника"),
            BotCommand("forget_all", "очистить память"),
            BotCommand("clear", "очистить историю чата"),
        ]
        await application.bot.set_my_commands(commands)
        logger.info("Telegram commands registered")

    async def shutdown(application):
        client = application.bot_data.get("groq_client")
        if client:
            await client.close()
        with db_lock:
            db.close()

    application = (
        Application.builder()
        .token(telegram_token)
        .post_init(initialize)
        .post_shutdown(shutdown)
        .build()
    )

    # Private-chat commands
    private_filter = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE
    application.add_handler(BusinessConnectionHandler(handle_business_connection))

    application.add_handler(CommandHandler("start", start, filters=private_filter))
    application.add_handler(CommandHandler("profile", profile, filters=private_filter))
    application.add_handler(CommandHandler("clear", clear, filters=private_filter))
    application.add_handler(CommandHandler("name", setname, filters=private_filter))
    application.add_handler(CommandHandler("age", setage, filters=private_filter))
    application.add_handler(CommandHandler("city", setcity, filters=private_filter))
    application.add_handler(CommandHandler("about", setabout, filters=private_filter))
    application.add_handler(
        CommandHandler("interests", setinterests, filters=private_filter)
    )
    application.add_handler(CommandHandler("caps", setcaps, filters=private_filter))
    application.add_handler(CommandHandler("gender", setgender, filters=private_filter))
    application.add_handler(
        CommandHandler("people", people_command, filters=private_filter)
    )
    application.add_handler(
        CommandHandler("person", person_command, filters=private_filter)
    )
    application.add_handler(
        CommandHandler("person_name", person_name_command, filters=private_filter)
    )
    application.add_handler(
        CommandHandler("person_gender", person_gender_command, filters=private_filter)
    )
    application.add_handler(
        CommandHandler("forget", forget_command, filters=private_filter)
    )
    application.add_handler(
        CommandHandler("forget_all", forget_all_command, filters=private_filter)
    )

    application.add_handler(
        MessageHandler(
            private_filter & filters.TEXT & ~filters.COMMAND,
            handle_private,
        )
    )

    # PTB uses singular BUSINESS_MESSAGE / EDITED_BUSINESS_MESSAGE update types.
    business_filter = (
        (filters.UpdateType.BUSINESS_MESSAGE
         | filters.UpdateType.EDITED_BUSINESS_MESSAGE)
        & filters.ChatType.PRIVATE
        & filters.TEXT
        & ~filters.COMMAND
    )
    application.add_handler(MessageHandler(business_filter, handle_business))

    application.add_error_handler(handle_error)
    logger.info("Starting Telegram assistant")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
