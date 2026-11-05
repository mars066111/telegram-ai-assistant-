
import asyncio
import json
import logging
import os
import random
import re
import sqlite3
from collections import deque
from threading import RLock, Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import AsyncOpenAI
from telegram import Update, BotCommand
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    BusinessConnectionHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

# ==========================================
# CONFIG
# ==========================================

MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite3")

HISTORY_LIMIT = 40
BATCH_WAIT_SECONDS = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db_lock = RLock()


# ==========================================
# DATABASE
# ==========================================

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

        if "gender" not in columns:
            db.execute("""
                ALTER TABLE owner_profile
                ADD COLUMN gender TEXT DEFAULT 'male'
            """)

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
            row["name"]
            for row in db.execute("PRAGMA table_info(people)")
        }

        if "gender" not in person_columns:
            db.execute("""
                ALTER TABLE people
                ADD COLUMN gender TEXT DEFAULT ''
            """)

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

        # Однократный сброс старой памяти собеседников.
        # Анкета владельца и настройки бота не удаляются.
        marker = db.execute("""
            SELECT value FROM app_settings
            WHERE setting = 'people_memory_reset_v1'
        """).fetchone()

        if marker is None:
            db.execute("DELETE FROM people")
            db.execute("DELETE FROM message_memory")

            db.execute("""
                INSERT INTO app_settings (setting, value)
                VALUES ('people_memory_reset_v1', 'done')
            """)

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
        "name", "age", "city", "about",
        "interests", "letter_case", "gender"
    }

    if field not in allowed:
        return

    with db_lock, db:
        db.execute(
            f"UPDATE owner_profile SET {field} = ? WHERE id = 1",
            (value[:1000],)
        )


def get_person(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT * FROM people WHERE chat_key = ?",
            (chat_key,)
        ).fetchone()

        if row:
            return dict(row)

        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)",
            (chat_key,)
        )
        db.commit()

    return {
        "name": "",
        "age": "",
        "city": "",
        "interests": "",
        "about": "",
        "gender": ""
    }


def update_person(chat_key, facts):
    allowed = {
        "name", "age", "city", "interests",
        "about", "gender"
    }

    with db_lock, db:
        for field, value in facts.items():
            if field not in allowed:
                continue

            if not isinstance(value, str):
                continue

            value = value.strip()[:500]

            if value:
                db.execute(
                    f"UPDATE people SET {field} = ? WHERE chat_key = ?",
                    (value, chat_key)
                )


def get_extra_memory(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT facts FROM message_memory WHERE chat_key = ?",
            (chat_key,)
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

    with db_lock, db:
        db.execute("""
            INSERT INTO message_memory (chat_key, facts)
            VALUES (?, ?)
            ON CONFLICT(chat_key)
            DO UPDATE SET facts = excluded.facts
        """, (
            chat_key,
            json.dumps(existing, ensure_ascii=False)
        ))


def find_person_key(identifier):
    """Находит private ID или ID бизнес-чата."""

    identifier = str(identifier).strip()

    with db_lock:
        row = db.execute(
            "SELECT chat_key FROM people WHERE chat_key = ?",
            (f"private:{identifier}",)
        ).fetchone()

        if row:
            return row["chat_key"]

        rows = db.execute(
            "SELECT chat_key FROM people"
        ).fetchall()

    matches = [
        row["chat_key"]
        for row in rows
        if row["chat_key"].endswith(f":{identifier}")
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def delete_person(chat_key):
    with db_lock, db:
        db.execute(
            "DELETE FROM people WHERE chat_key = ?",
            (chat_key,)
        )
        db.execute(
            "DELETE FROM message_memory WHERE chat_key = ?",
            (chat_key,)
        )


def clear_all_people():
    with db_lock, db:
        db.execute("DELETE FROM people")
        db.execute("DELETE FROM message_memory")


def list_people():
    with db_lock:
        return db.execute("""
            SELECT chat_key, name, gender
            FROM people
            ORDER BY chat_key
        """).fetchall()


# ==========================================
# HEALTH SERVER FOR RENDER
# ==========================================

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
        HealthHandler
    )

    Thread(
        target=server.serve_forever,
        daemon=True
    ).start()

    logger.info("Health server started on port %s", port)


# ==========================================
# PROMPT
# ==========================================

SYSTEM_PROMPT = """
Ты — ИИ-помощник, который помогает вести переписку
в Telegram Business.

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

================================
СТИЛЬ ОБЩЕНИЯ
================================

Общайся естественно, как в обычной переписке.
Не используй шаблонные ответы и не звучь как
служба поддержки.

Будь дружелюбным, внимательным и живым.
Не переигрывай с эмоциями и не изображай
чрезмерный восторг от каждого сообщения.

Отвечай на конкретное содержание сообщения.
Не игнорируй детали, которые собеседница рассказала.

================================
ВЫСОКАЯ ИНИЦИАТИВНОСТЬ
================================

Не ограничивайся короткой реакцией на каждое сообщение.

Развивай интересные темы:
- высказывай своё мнение;
- замечай интересные подробности;
- иногда предлагай новую тему;
- задавай содержательные вопросы;
- возвращайся к ранее обсуждавшимся событиям;
- интересуйся результатами её планов, если это уместно.

Не задавай вопросы механически после каждого сообщения.
Не превращай разговор в интервью.

Избегай банальных вопросов:
«как дела?», «что делаешь?», «что нового?»,
«как настроение?».

Вместо них используй вопросы, связанные
с реальной темой разговора.

================================
ПАМЯТЬ И ИМЕНА
================================

Используй только информацию из профиля и памяти.

Никогда не придумывай имя, возраст, город,
интересы или другие факты.

Если имя неизвестно — не называй собеседницу по имени.
Не используй случайные имена и прозвища.

Если факт не сохранён и не был сообщён в переписке,
считай его неизвестным.

Не путай людей из разных чатов.
Не переноси факты одного человека другому.

Если информация противоречит старой записи,
не делай вид, что уверен. Уточни её.

Не сообщай всю сохранённую информацию,
если спросили только об одном факте.

================================
ПОЛ СОБЕСЕДНИКА
================================

Пол собеседника указан в отдельном блоке.
Используй соответствующие формы обращения.

Если пол неизвестен, избегай гендерно окрашенных
форм там, где можно сформулировать нейтрально.

Не угадывай пол по имени или фотографии.

================================
ПОЛ БОТА
================================

Твой собственный грамматический род указан отдельно.
Не путай его с полом собеседника.

Если выбран мужской род:
«я понял», «я рад», «я заметил».

Если выбран женский род:
«я поняла», «я рада», «я заметила».

Строго соблюдай выбранный род.

================================
ЮМОР И КОМПЛИМЕНТЫ
================================

Используй лёгкий юмор, если ситуация подходит.
Иногда можно слегка подшутить, но не обижай.

Комплименты должны быть конкретными и уместными.
Не повторяй одни и те же комплименты.
Не делай комплименты в каждом сообщении.

================================
ДЛИНА
================================

Обычно 1–4 предложения.
Простые вопросы — короткие ответы.
Интересные истории — более содержательные реакции.

Не растягивай ответ искусственно.

================================
ЯЗЫК
================================

Отвечай на языке собеседницы.
Если она пишет по-русски — используй естественный русский.
Не вставляй случайные английские слова.
Не используй случайные иностранные фразы.

================================
ЧЕСТНОСТЬ
================================

Не выдумывай личный опыт и события.
Не утверждай, что являешься человеком.
Если прямо спрашивают об автоматизации, отвечай честно.

Выдавай только текст ответа.
"""


def format_profile(profile):
    fields = {
        "name": "Имя",
        "age": "Возраст",
        "city": "Город",
        "about": "О себе",
        "interests": "Интересы"
    }

    return "\n".join(
        f"{label}: {profile.get(key) or 'неизвестно'}"
        for key, label in fields.items()
    )


def make_prompt(chat_key):
    owner = get_profile()
    person = get_person(chat_key)

    if owner.get("gender") == "female":
        bot_gender = (
            "Женский род. Говори о себе: «я поняла», "
            "«я рада», «я заметила»."
        )
    else:
        bot_gender = (
            "Мужской род. Говори о себе: «я понял», "
            "«я рад», «я заметил»."
        )

    person_gender = person.get("gender", "")

    if person_gender == "female":
        person_gender_instruction = (
            "Женский род. Обращайся: «ты сказала», "
            "«ты хотела», «ты сама»."
        )
    elif person_gender == "male":
        person_gender_instruction = (
            "Мужской род. Обращайся: «ты сказал», "
            "«ты хотел», «ты сам»."
        )
    else:
        person_gender_instruction = (
            "Пол неизвестен. Используй нейтральные формулировки."
        )

    if owner.get("letter_case") == "upper":
        case_instruction = (
            "Используй обычную заглавную букву в начале предложений."
        )
    else:
        case_instruction = (
            "Начинай обычные предложения с маленькой буквы. "
            "Имена собственные всегда пиши с заглавной буквы."
        )

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(owner),
        bot_gender_instruction=bot_gender,
        person_profile=format_profile(person),
        person_gender_instruction=person_gender_instruction,
        extra_memory=json.dumps(
            get_extra_memory(chat_key),
            ensure_ascii=False
        ),
        letter_case_instruction=case_instruction
    )


# ==========================================
# MEMORY EXTRACTION
# ==========================================

async def remember_person(chat_key, message, client):
    extraction_prompt = """
Извлеки только факты, которые человек явно сообщил о себе.

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
- Не угадывай имя.
- Не делай предположений по контексту.
- Не записывай сведения о других людях как сведения о собеседнике.
- Если человек не назвал своё имя, оставь name пустым.
- Если поле не подтверждено явно, оставь его пустым.
- Не записывай случайные обращения и прозвища как имя.
- other_facts — только важные сведения, сказанные о себе.
- Не добавляй пояснений вне JSON.
"""

    try:
        result = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0,
            max_tokens=300
        )

        content = result.choices[0].message.content

        if not content:
            return

        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content.strip()
        )

        facts = json.loads(content)

        if not isinstance(facts, dict):
            return

        # Не разрешаем автоматическому извлечению менять пол.
        facts.pop("gender", None)

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
                {"other_facts": old_facts[-30:]}
            )

    except Exception:
        logger.exception("Memory extraction error")


# ==========================================
# COMMANDS: OWNER PROFILE
# ==========================================

async def start(update, context):
    if update.message:
        await update.message.reply_text(
            "привет! я готов помогать с перепиской.\n\n"
            "АНКЕТА:\n"
            "/profile — посмотреть анкету\n"
            "/name значение — изменить имя\n"
            "/age значение — изменить возраст\n"
            "/city значение — изменить город\n"
            "/about значение — информация о себе\n"
            "/interests значение — интересы\n\n"
            "НАСТРОЙКИ БОТА:\n"
            "/gender male — мужской род\n"
            "/gender female — женский род\n"
            "/caps lower — маленькие буквы\n"
            "/caps upper — обычный регистр\n\n"
            "СОБЕСЕДНИКИ:\n"
            "/people — список собеседников\n"
            "/person ID — информация о собеседнике\n"
            "/person_name ID имя — изменить имя\n"
            "/person_gender ID male — мужской пол\n"
            "/person_gender ID female — женский пол\n"
            "/forget ID — забыть собеседника\n"
            "/forget_all — очистить всю память\n\n"
            "/clear — очистить историю текущего личного чата"
        )


async def profile(update, context):
    if not update.message:
        return

    data = get_profile()

    gender_text = (
        "женский"
        if data.get("gender") == "female"
        else "мужской"
    )

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
        f"пол бота: {gender_text}\n"
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
    await set_field(update, context, "about", "информация", "about")


async def setinterests(update, context):
    await set_field(
        update, context, "interests", "интересы", "interests"
    )


async def setcaps(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
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

    await update.message.reply_text("настройка регистра сохранена")


async def setgender(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "/gender male — мужской\n"
            "/gender female — женский"
        )
        return

    choice = context.args[0].lower()

    if choice in ("male", "мужской", "м"):
        gender = "male"
        answer = "выбран мужской род"

    elif choice in ("female", "женский", "ж"):
        gender = "female"
        answer = "выбран женский род"

    else:
        await update.message.reply_text(
            "используй /gender male или /gender female"
        )
        return

    set_profile_field("gender", gender)
    await update.message.reply_text(answer)


# ==========================================
# COMMANDS: PEOPLE
# ==========================================

async def people_command(update, context):
    if not update.message:
        return

    rows = list_people()

    if not rows:
        await update.message.reply_text(
            "сохранённых собеседников пока нет"
        )
        return

    lines = ["сохранённые собеседники:"]

    for row in rows[:50]:
        name = row["name"] or "имя неизвестно"
        lines.append(
            f"{row['chat_key']} — {name}"
        )

    await update.message.reply_text("\n".join(lines))


async def person_command(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "использование: /person ID"
        )
        return

    identifier = context.args[0]
    key = find_person_key(identifier)

    if not key:
        await update.message.reply_text(
            "собеседник с таким ID не найден"
        )
        return

    person = get_person(key)
    extra = get_extra_memory(key)

    gender_text = {
        "male": "мужской",
        "female": "женский"
    }.get(person.get("gender"), "не выбран")

    await update.message.reply_text(
        f"ID: {key}\n\n"
        f"имя: {person.get('name') or 'неизвестно'}\n"
        f"возраст: {person.get('age') or 'неизвестно'}\n"
        f"город: {person.get('city') or 'неизвестно'}\n"
        f"интересы: {person.get('interests') or 'неизвестно'}\n"
        f"о себе: {person.get('about') or 'неизвестно'}\n"
        f"пол: {gender_text}\n\n"
        f"дополнительные факты:\n"
        f"{json.dumps(extra, ensure_ascii=False)}"
    )


async def person_name_command(update, context):
    if not update.message:
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "использование: /person_name ID имя"
        )
        return

    identifier = context.args[0]
    name = " ".join(context.args[1:]).strip()

    key = find_person_key(identifier)

    if not key:
        # Создаём отдельную запись для private ID,
        # если собеседник ещё не попадал в базу.
        key = f"private:{identifier}"

    update_person(key, {"name": name})

    await update.message.reply_text(
        f"имя собеседника сохранено: {name}"
    )


async def person_gender_command(update, context):
    if not update.message:
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "использование:\n"
            "/person_gender ID male\n"
            "/person_gender ID female"
        )
        return

    identifier = context.args[0]
    choice = context.args[1].lower()

    if choice in ("male", "мужской", "м"):
        gender = "male"
        answer = "мужской пол сохранён"

    elif choice in ("female", "женский", "ж"):
        gender = "female"
        answer = "женский пол сохранён"

    else:
        await update.message.reply_text(
            "выбери male или female"
        )
        return

    key = find_person_key(identifier)

    if not key:
        key = f"private:{identifier}"

    update_person(key, {"gender": gender})

    await update.message.reply_text(answer)


async def forget_command(update, context):
    if not update.message:
        return

    if not context.args:
        await update.message.reply_text(
            "использование: /forget ID"
        )
        return

    identifier = context.args[0]
    key = find_person_key(identifier)

    if not key:
        await update.message.reply_text(
            "собеседник с таким ID не найден"
        )
        return

    delete_person(key)

    histories.pop(("business", key.replace(":", ":", 1)), None)
    histories.pop(("private", identifier), None)

    await update.message.reply_text(
        "данные этого собеседника удалены"
    )


async def forget_all_command(update, context):
    if not update.message:
        return

    clear_all_people()

    histories.clear()

    await update.message.reply_text(
        "память всех собеседников очищена. "
        "анкета владельца и настройки бота сохранены."
    )


# ==========================================
# CHAT HISTORY AND DELAYS
# ==========================================

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


# ==========================================
# AI RESPONSE
# ==========================================

async def generate_answer(chat_key, history, client):
    result = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": make_prompt(chat_key)
            },
            *list(history)
        ],
        temperature=0.9,
        max_tokens=450
    )

    answer = result.choices[0].message.content

    if not answer or not answer.strip():
        return None

    answer = clean_answer(answer)

    mode = get_profile().get("letter_case", "lower")

    return apply_letter_case(answer, mode)


# ==========================================
# PRIVATE CHAT
# ==========================================

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
            "content": message.text
        })

        try:
            client = context.application.bot_data["groq_client"]

            await remember_person(
                chat_key,
                message.text,
                client
            )

            answer = await generate_answer(
                chat_key,
                history,
                client
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
                "content": answer
            })

            await message.reply_text(answer)

        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()

            logger.exception("Private response error")

            await message.reply_text(
                "не получилось получить ответ"
            )


# ==========================================
# BUSINESS CONNECTIONS
# ==========================================

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


# ==========================================
# BUSINESS MESSAGE HANDLER
# ==========================================

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
        connection_id,
        context
    )

    if connection is None:
        return

    chat_key = f"{connection_id}:{message.chat.id}"
    key = ("business", chat_key)

    if (
        message.from_user is not None
        and message.from_user.id == connection.user.id
    ):
        async with get_lock(key):
            get_history(key).append({
                "role": "assistant",
                "content": message.text
            })
        return

    try:
        await context.bot.read_business_message(
            business_connection_id=connection_id,
            chat_id=message.chat.id,
            message_id=message.message_id
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
            "chat_key": chat_key
        }

        batches[key] = batch

    batch["messages"].append(message)

    if batch["task"] is not None:
        batch["task"].cancel()

    batch["task"] = context.application.create_task(
        process_batch(batch, context)
    )


# ==========================================
# PROCESS BATCH
# ==========================================

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
            batch["connection_id"],
            context
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
            batch,
            combined_text,
            context
        )

    except asyncio.CancelledError:
        return

    except Exception:
        logger.exception("Batch error")


# ==========================================
# BUSINESS REPLY
# ==========================================

async def generate_business_reply(batch, incoming_text, context):
    key = batch["key"]
    chat_id = batch["messages"][-1].chat.id
    chat_key = batch["chat_key"]

    async with get_lock(key):
        history = get_history(key)

        history.append({
            "role": "user",
            "content": incoming_text
        })

        try:
            connection = await get_connection(
                batch["connection_id"],
                context
            )

            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]

            await remember_person(
                chat_key,
                incoming_text,
                client
            )

            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch["connection_id"]
                )
            except Exception:
                pass

            answer = await generate_answer(
                chat_key,
                history,
                client
            )

            if not answer:
                history.pop()
                return

            await asyncio.sleep(get_reply_delay(answer))

            await context.bot.send_message(
                chat_id=chat_id,
                text=answer,
                business_connection_id=batch["connection_id"]
            )

            history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()

            logger.exception("Business reply error")


# ==========================================
# ERROR HANDLER
# ==========================================

async def handle_error(update, context):
    logger.error(
        "Telegram error",
        exc_info=context.error
    )


# ==========================================
# MAIN
# ==========================================

def main():
    start_health_server()

    telegram_token = os.environ["TELEGRAM_BOT_TOKEN"]
    groq_api_key = os.environ["GROQ_API_KEY"]

    async def initialize(application):
        application.bot_data["groq_client"] = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=GROQ_BASE_URL,
            timeout=45,
            max_retries=1
        )

        # Команды в меню Telegram.
        await application.bot.set_my_commands([
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
        ])

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
        CommandHandler("name", setname, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("age", setage, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("city", setcity, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("about", setabout, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("interests", setinterests, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("caps", setcaps, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("gender", setgender, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("people", people_command, filters=private_filter)
    )

    application.add_handler(
        CommandHandler("person", person_command, filters=private_filter)
    )

    application.add_handler(
        CommandHandler(
            "person_name",
            person_name_command,
            filters=private_filter
        )
    )

    application.add_handler(
        CommandHandler(
            "person_gender",
            person_gender_command,
            filters=private_filter
        )
    )

    application.add_handler(
        CommandHandler("forget", forget_command, filters=private_filter)
    )

    application.add_handler(
        CommandHandler(
            "forget_all",
            forget_all_command,
            filters=private_filter
        )
    )

    application.add_handler(
        MessageHandler(
            private_filter
            & filters.TEXT
            & ~filters.COMMAND,
            handle_private
        )
    )

    application.add_handler(
        MessageHandler(
            filters.UpdateType.BUSINESS_MESSAGES
            & filters.ChatType.PRIVATE
            & filters.TEXT
            & ~filters.COMMAND,
            handle_business
        )
    )

    application.add_error_handler(handle_error)

    logger.info("Starting Telegram assistant")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
