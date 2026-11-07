import asyncio
import json
import logging
import os
import re
import sqlite3
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    Application, BusinessConnectionHandler, CommandHandler,
    MessageHandler, ContextTypes, filters,
)

MODEL = "qwen/qwen3.8-27b"
BASE_URL = "https://api.groq.com/openai/v1"
DB_PATH = os.getenv("DB_PATH", "bot_data.sqlite3")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("telegram_ai_bot")

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db_lock = asyncio.Lock()
histories = defaultdict(lambda: deque(maxlen=30))
chat_locks = defaultdict(asyncio.Lock)
seen = deque(maxlen=5000)
seen_set = set()


def init_db():
    with db:
        db.execute("""CREATE TABLE IF NOT EXISTS owner_profile (
            id INTEGER PRIMARY KEY CHECK(id=1),
            name TEXT DEFAULT '', age TEXT DEFAULT '', city TEXT DEFAULT '',
            about TEXT DEFAULT '', interests TEXT DEFAULT '',
            gender TEXT DEFAULT 'female', letter_case TEXT DEFAULT 'lower'
        )""")
        db.execute("INSERT OR IGNORE INTO owner_profile(id) VALUES(1)")
        db.execute("""CREATE TABLE IF NOT EXISTS people (
            chat_key TEXT PRIMARY KEY, name TEXT DEFAULT '', age TEXT DEFAULT '',
            city TEXT DEFAULT '', about TEXT DEFAULT '', interests TEXT DEFAULT '',
            gender TEXT DEFAULT ''
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS memory (
            chat_key TEXT PRIMARY KEY, facts TEXT DEFAULT '{}'
        )""")


init_db()


def profile():
    return dict(db.execute("SELECT * FROM owner_profile WHERE id=1").fetchone())


def set_owner(field, value):
    allowed = {"name", "age", "city", "about", "interests", "gender", "letter_case"}
    if field not in allowed:
        return
    with db:
        db.execute(f"UPDATE owner_profile SET {field}=? WHERE id=1", (value[:1000],))


def person(chat_key):
    with db:
        db.execute("INSERT OR IGNORE INTO people(chat_key) VALUES(?)", (chat_key,))
    row = db.execute("SELECT * FROM people WHERE chat_key=?", (chat_key,)).fetchone()
    return dict(row)


def set_person(chat_key, field, value):
    if field not in {"name", "age", "city", "about", "interests", "gender"}:
        return
    with db:
        db.execute("INSERT OR IGNORE INTO people(chat_key) VALUES(?)", (chat_key,))
        db.execute(f"UPDATE people SET {field}=? WHERE chat_key=?", (value[:500], chat_key))


def get_memory(chat_key):
    row = db.execute("SELECT facts FROM memory WHERE chat_key=?", (chat_key,)).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row["facts"])
    except Exception:
        return {}


def save_memory(chat_key, facts):
    with db:
        db.execute("""INSERT INTO memory(chat_key,facts) VALUES(?,?)
            ON CONFLICT(chat_key) DO UPDATE SET facts=excluded.facts""",
            (chat_key, json.dumps(facts, ensure_ascii=False)))


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Bot is running".encode())

    def log_message(self, *_):
        pass


def start_health():
    port = int(os.getenv("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


def profile_text(p):
    labels = {"name": "Имя", "age": "Возраст", "city": "Город",
              "about": "О себе", "interests": "Интересы"}
    return "\n".join(f"{label}: {p.get(key) or 'не указано'}"
                     for key, label in labels.items())


def make_prompt(chat_key):
    p = profile()
    other = person(chat_key)
    memory = get_memory(chat_key)
    gender = "женский" if p.get("gender") == "female" else "мужской"
    other_gender = {"female": "женский", "male": "мужской"}.get(
        other.get("gender"), "неизвестен"
    )
    return f"""Ты — персонаж Telegram-аккаунта. Общайся естественно, как человек в обычной переписке.
Анкета персонажа:
{profile_text(p)}
Грамматический род персонажа: {gender}.
Данные собеседника:
{profile_text(other)}
Грамматический род собеседника: {other_gender}.
Дополнительная память об этом собеседнике: {json.dumps(memory, ensure_ascii=False)}

Правила:
- Имя и возраст из анкеты — это данные персонажа. Не заменяй их выдуманными именами и возрастом.
- Если имя в анкете — Аня, представляйся Аней; если возраст 15, не говори, что возраст неизвестен.
- Не начинай каждый ответ с приветствия и не задавай вопрос в каждом сообщении.
- Не отвечай сухими однословными фразами, если можно естественно продолжить мысль.
- Пиши живо, тепло, с уместным юмором и разговорными выражениями, но не переигрывай.
- Обычно 1–4 предложения. Не используй театральные фразы и шаблонные реплики.
- Не вставляй случайные иностранные слова, иероглифы или бессмысленные символы.
- Не выдумывай факты о себе, события и личный опыт, которых нет в анкете или переписке.
- Не переноси сведения между разными собеседниками.
- Если прямо спрашивают, является ли аккаунт ИИ-ботом, отвечай честно.
- Не имитируй романтические или сексуальные отношения.
- Отвечай на языке собеседника; в русском используй естественный современный русский.
- Регистр: {"обычный" if p.get("letter_case") == "upper" else "начинай обычные предложения со строчной буквы; имена собственные — с заглавной"}.
Возвращай только текст ответа."""


def clean(text):
    text = (text or "").strip()
    text = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", "", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def apply_case(text):
    if profile().get("letter_case") == "upper":
        return text
    out, start = [], True
    for ch in text:
        if start and ch.isalpha():
            out.append(ch.lower())
            start = False
        else:
            out.append(ch)
        if ch in ".!?":
            start = True
    return "".join(out)


async def answer(chat_key, text, app):
    hist = histories[chat_key]
    hist.append({"role": "user", "content": text})
    client = app.bot_data["client"]
    try:
        result = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": make_prompt(chat_key)}, *hist],
            temperature=0.9,
            max_tokens=350,
        )
        reply = apply_case(clean(result.choices[0].message.content))
        if not reply:
            hist.pop()
            return
        hist.append({"role": "assistant", "content": reply})
        return reply
    except Exception:
        if hist and hist[-1].get("role") == "user":
            hist.pop()
        raise


def is_duplicate(update):
    key = update.update_id
    if key in seen_set:
        return True
    if len(seen) == seen.maxlen:
        seen_set.discard(seen[0])
    seen.append(key)
    seen_set.add(key)
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Я на связи. Команды: /profile, /name, /age, /city, /about, "
        "/interests, /gender, /caps, /clear, /person_gender."
    )


async def profile_cmd(update, context):
    await update.effective_message.reply_text(profile_text(profile()))


async def set_field(update, context, field, usage):
    if not context.args:
        await update.effective_message.reply_text(f"Использование: /{field} {usage}")
        return
    set_owner(field, " ".join(context.args))
    await update.effective_message.reply_text("Сохранил.")


async def name_cmd(u, c): await set_field(u, c, "name", "имя")
async def age_cmd(u, c): await set_field(u, c, "age", "возраст")
async def city_cmd(u, c): await set_field(u, c, "city", "город")
async def about_cmd(u, c): await set_field(u, c, "about", "текст")
async def interests_cmd(u, c): await set_field(u, c, "interests", "интересы")


async def gender_cmd(update, context):
    if not context.args:
        await update.effective_message.reply_text("/gender female или /gender male")
        return
    value = context.args[0].lower()
    if value in ("female", "ж", "женский"):
        set_owner("gender", "female")
    elif value in ("male", "м", "мужской"):
        set_owner("gender", "male")
    else:
        await update.effective_message.reply_text("Укажи female или male.")
        return
    await update.effective_message.reply_text("Сохранил.")


async def caps_cmd(update, context):
    if not context.args or context.args[0] not in ("lower", "upper"):
        await update.effective_message.reply_text("/caps lower или /caps upper")
        return
    set_owner("letter_case", context.args[0])
    await update.effective_message.reply_text("Сохранил.")


async def clear_cmd(update, context):
    key = f"private:{update.effective_user.id}"
    histories.pop(key, None)
    await update.effective_message.reply_text("История очищена.")


async def person_gender_cmd(update, context):
    if len(context.args) < 2:
        await update.effective_message.reply_text("/person_gender ID male|female")
        return
    ident, value = context.args[0], context.args[1].lower()
    value = "female" if value in ("female", "ж", "женский") else (
        "male" if value in ("male", "м", "мужской") else ""
    )
    if not value:
        await update.effective_message.reply_text("Укажи male или female.")
        return
    set_person(f"private:{ident}", "gender", value)
    await update.effective_message.reply_text("Сохранил.")


async def handle_private(update, context):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not msg.text or not user or is_duplicate(update):
        return
    key = f"private:{user.id}"
    async with chat_locks[key]:
        try:
            reply = await answer(key, msg.text, context.application)
            if reply:
                await msg.reply_text(reply)
        except Exception:
            log.exception("Private chat error")
            await msg.reply_text("Не получилось получить ответ. Попробуй ещё раз позже.")


async def business_connection(update, context):
    conn = update.business_connection
    if conn:
        context.application.bot_data.setdefault("connections", {})[conn.id] = conn


async def handle_business(update, context):
    msg = update.business_message
    if not msg or not msg.text or not msg.business_connection_id or is_duplicate(update):
        return
    conn_id = msg.business_connection_id
    conn = context.application.bot_data.get("connections", {}).get(conn_id)
    if conn is None:
        try:
            conn = await context.bot.get_business_connection(conn_id)
        except Exception:
            log.exception("Cannot get business connection")
            return
    if not conn.is_enabled or (msg.from_user and msg.from_user.id == conn.user.id):
        return
    key = f"{conn_id}:{msg.chat.id}"
    async with chat_locks[key]:
        try:
            reply = await answer(key, msg.text, context.application)
            if reply:
                await context.bot.send_message(
                    chat_id=msg.chat.id, text=reply,
                    business_connection_id=conn_id,
                )
        except Exception:
            log.exception("Business chat error")


async def post_init(app):
    app.bot_data["client"] = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"], base_url=BASE_URL,
        timeout=45, max_retries=2,
    )


async def post_shutdown(app):
    client = app.bot_data.get("client")
    if client:
        await client.close()
    db.close()


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and GROQ_API_KEY environment variables.")
    Thread(target=start_health, daemon=True).start()
    app = (Application.builder().token(token).post_init(post_init)
           .post_shutdown(post_shutdown).build())

    private = filters.ChatType.PRIVATE
    for command, callback in [
        ("start", start), ("profile", profile_cmd), ("name", name_cmd),
        ("age", age_cmd), ("city", city_cmd), ("about", about_cmd),
        ("interests", interests_cmd), ("gender", gender_cmd),
        ("caps", caps_cmd), ("clear", clear_cmd),
        ("person_gender", person_gender_cmd),
    ]:
        app.add_handler(CommandHandler(command, callback, filters=private))

    app.add_handler(MessageHandler(
        private & filters.TEXT & ~filters.COMMAND, handle_private
    ))
    app.add_handler(BusinessConnectionHandler(business_connection))
    app.add_handler(MessageHandler(
        filters.UpdateType.BUSINESS_MESSAGE & filters.TEXT & ~filters.COMMAND,
        handle_business,
    ))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
