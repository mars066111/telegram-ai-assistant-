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
BATCH_WAIT_SECONDS = 0.8
SEEN_LIMIT = 5000

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

histories = {}
locks = {}
seen_updates = set()
seen_update_order = deque()
seen_messages = set()
seen_message_order = deque()


def remember_once(key, seen, order):
    """Return False for an update/message that this process already handled."""
    if key in seen:
        return False
    seen.add(key)
    order.append(key)
    while len(order) > SEEN_LIMIT:
        old = order.popleft()
        seen.discard(old)
    return True


def is_duplicate_update(update, message, namespace):
    update_id = getattr(update, "update_id", None)
    if update_id is not None and not remember_once(
        (namespace, update_id), seen_updates, seen_update_order
    ):
        return True
    if message is not None:
        business_id = getattr(message, "business_connection_id", None) or ""
        message_key = (
            namespace, business_id, getattr(message.chat, "id", None),
            getattr(message, "message_id", None),
        )
        if not remember_once(message_key, seen_messages, seen_message_order):
            return True
    return False


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
        for column, declaration in (
            ("letter_case", "TEXT DEFAULT 'lower'"),
            ("gender", "TEXT DEFAULT 'male'"),
        ):
            if column not in owner_columns:
                db.execute(
                    f"ALTER TABLE owner_profile ADD COLUMN {column} {declaration}"
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
        people_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(people)")
        }
        if "gender" not in people_columns:
            db.execute("ALTER TABLE people ADD COLUMN gender TEXT DEFAULT ''")

        db.execute("""
            CREATE TABLE IF NOT EXISTS message_memory (
                chat_key TEXT PRIMARY KEY,
                facts TEXT DEFAULT '{}'
            )
        """)


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
    with db_lock, db:
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        row = db.execute(
            "SELECT * FROM people WHERE chat_key = ?", (chat_key,)
        ).fetchone()
    return dict(row) if row else {}


def update_person(chat_key, facts):
    allowed = {"name", "age", "city", "interests", "about"}
    with db_lock, db:
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


def set_person_gender(chat_key, gender):
    if gender not in ("male", "female"):
        return
    with db_lock, db:
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        db.execute(
            "UPDATE people SET gender = ? WHERE chat_key = ?",
            (gender, chat_key),
        )


def get_extra_memory(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT facts FROM message_memory WHERE chat_key = ?", (chat_key,)
        ).fetchone()
    if not row:
        return {}
    try:
        value = json.loads(row["facts"])
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def save_extra_memory(chat_key, facts):
    with db_lock, db:
        db.execute("""
            INSERT INTO message_memory (chat_key, facts)
            VALUES (?, ?)
            ON CONFLICT(chat_key) DO UPDATE SET facts = excluded.facts
        """, (chat_key, json.dumps(facts, ensure_ascii=False)))


def find_person_key(identifier):
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
        self.wfile.write("Bot is running".encode("utf-8"))

    def log_message(self, fmt, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health server started on port %s", port)


# =========================================================
# PROMPT AND MEMORY
# =========================================================

SYSTEM_PROMPT = """
Ð¢Ð« â ÐÐÐ Ð¡ÐÐÐÐ Ð­Ð¢ÐÐÐ TELEGRAM-ÐÐÐÐÐ£ÐÐ¢Ð.

Ð¢ÐÐÐ¯ ÐÐÐÐÐ¢Ð Ð ÐÐÐ§ÐÐÐ¡Ð¢Ð¬:
{owner_profile}

ÐÐÐÐÐ ÐÐ¡ÐÐÐÐ¬ÐÐÐÐÐ¢Ð¬ ÐÐÐÐÐ¢Ð£:
- ÐÐ¼Ñ Ð¸Ð· Ð°Ð½ÐºÐµÑÑ â ÑÐ²Ð¾Ñ Ð¸Ð¼Ñ Ð² Ð¾Ð±ÑÑÐ½Ð¾Ð¼ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾ÑÐµ. ÐÐµ Ð¿Ð¾Ð´Ð¼ÐµÐ½ÑÐ¹ ÐµÐ³Ð¾ Ð´ÑÑÐ³Ð¸Ð¼ Ð¸Ð¼ÐµÐ½ÐµÐ¼.
- ÐÐ¾Ð·ÑÐ°ÑÑ Ð¸Ð· Ð°Ð½ÐºÐµÑÑ â Ð²Ð¾Ð·ÑÐ°ÑÑ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°. ÐÑÐ»Ð¸ Ð¾Ð½ ÑÐºÐ°Ð·Ð°Ð½, Ð½Ðµ Ð¾ÑÐ²ÐµÑÐ°Ð¹ Â«Ñ Ð¼ÐµÐ½Ñ Ð½ÐµÑ Ð²Ð¾Ð·ÑÐ°ÑÑÐ°Â».
- ÐÐ¾ÑÐ¾Ð´, Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ðµ Ð¸ Ð¸Ð½ÑÐµÑÐµÑÑ Ð¸Ð· Ð°Ð½ÐºÐµÑÑ â ÑÐ²Ð¾Ð¸ Ð´Ð°Ð½Ð½ÑÐµ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°. Ð£ÑÐ¸ÑÑÐ²Ð°Ð¹ Ð¸Ñ Ð² ÑÐ°Ð·Ð³Ð¾Ð²Ð¾ÑÐµ, Ð½Ð¾ Ð½Ðµ Ð¿ÐµÑÐµÑÐ¸ÑÐ»ÑÐ¹ Ð±ÐµÐ· Ð¿Ð¾Ð²Ð¾Ð´Ð°.
- ÐÑÐ»Ð¸ Ð¿Ð¾Ð»Ðµ Ð¿ÑÑÑÐ¾Ðµ, Ð½Ðµ Ð¿ÑÐ¸Ð´ÑÐ¼ÑÐ²Ð°Ð¹ Ð´Ð»Ñ Ð½ÐµÐ³Ð¾ Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ.
- ÐÐµ Ð½Ð°Ð·ÑÐ²Ð°Ð¹ ÑÐµÐ±Ñ ÐÐ°ÑÑÐµÐ»ÐµÐ¼, ÐµÑÐ»Ð¸ Ð² Ð°Ð½ÐºÐµÑÐµ ÑÐºÐ°Ð·Ð°Ð½Ð¾ Ð´ÑÑÐ³Ð¾Ðµ Ð¸Ð¼Ñ.
- ÐÐµ Ð³Ð¾Ð²Ð¾ÑÐ¸ ÑÐ°Ð¼ Ð¿Ð¾ ÑÐµÐ±Ðµ Â«Ñ Ð²Ð¸ÑÑÑÐ°Ð»ÑÐ½ÑÐ¹ Ð¿Ð¾Ð¼Ð¾ÑÐ½Ð¸ÐºÂ» Ð¸Ð»Ð¸ Â«Ñ ÐÐÂ». ÐÑÐ»Ð¸ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ð¿ÑÑÐ¼Ð¾ ÑÐ¿ÑÐ¾ÑÐ¸Ñ, Ð±Ð¾Ñ Ð»Ð¸ ÑÑ Ð¸Ð»Ð¸ ÐÐ, ÑÐµÑÑÐ½Ð¾ Ð¾ÑÐ²ÐµÑÑ, ÑÑÐ¾ ÑÑÐ¾ ÐÐ-Ð±Ð¾Ñ, ÐºÐ¾ÑÐ¾ÑÑÐ¹ Ð¾Ð±ÑÐ°ÐµÑÑÑ Ð² Ð·Ð°Ð´Ð°Ð½Ð½Ð¾Ð¼ Ð¾Ð±ÑÐ°Ð·Ðµ.

Ð¢ÐÐÐ ÐÐ ÐÐÐÐÐ¢ÐÐ§ÐÐ¡ÐÐÐ Ð ÐÐ:
{bot_gender}

ÐÐÐ¤ÐÐ ÐÐÐ¦ÐÐ¯ Ð Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:
{person_profile}

Ð ÐÐ Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:
{person_gender}

ÐÐÐÐÐÐÐÐ¢ÐÐÐ¬ÐÐÐ¯ ÐÐÐÐ¯Ð¢Ð¬:
{extra_memory}

Ð¡Ð¢ÐÐÐ¬ ÐÐÐ©ÐÐÐÐ¯:
- ÐÐ±ÑÐ°Ð¹ÑÑ Ð¶Ð¸Ð²Ð¾, ÑÐµÐ¿Ð»Ð¾ Ð¸ Ð½ÐµÐ¿ÑÐ¸Ð½ÑÐ¶Ð´ÑÐ½Ð½Ð¾, ÐºÐ°Ðº Ð² Ð½Ð¾ÑÐ¼Ð°Ð»ÑÐ½Ð¾Ð¹ Ð¿ÐµÑÐµÐ¿Ð¸ÑÐºÐµ Ð² Telegram.
- ÐÐµ Ð±ÑÐ´Ñ ÑÑÑÐ¸Ð¼ Ð¸ ÑÐ¾ÑÐ¼Ð°Ð»ÑÐ½ÑÐ¼. ÐÐ° Ð¾Ð±ÑÑÐ½ÑÐµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ ÑÐµÐ°Ð³Ð¸ÑÑÐ¹ ÑÐ¾Ð´ÐµÑÐ¶Ð°ÑÐµÐ»ÑÐ½Ð¾: Ð´Ð¾Ð±Ð°Ð²Ð»ÑÐ¹ ÐµÑÑÐµÑÑÐ²ÐµÐ½Ð½ÑÑ ÑÐ¼Ð¾ÑÐ¸Ñ, ÐºÐ¾ÑÐ¾ÑÐºÑÑ Ð¼ÑÑÐ»Ñ Ð¸Ð»Ð¸ ÑÐ¼ÐµÑÑÐ½ÑÑ ÑÑÑÐºÑ.
- ÐÐµ Ð¾Ð³ÑÐ°Ð½Ð¸ÑÐ¸Ð²Ð°Ð¹ÑÑ Ð¾Ð´Ð½Ð¾ÑÐ»Ð¾Ð¶Ð½ÑÐ¼Ð¸ Ð¾ÑÐ²ÐµÑÐ°Ð¼Ð¸ Ð²ÑÐ¾Ð´Ðµ Â«ÑÑÐ½Ð¾Â», Â«Ð¿Ð¾Ð½ÑÑÐ½Ð¾Â», Â«ÑÐ¿Ð°ÑÐ¸Ð±Ð¾Â», ÐµÑÐ»Ð¸ Ð¼Ð¾Ð¶Ð½Ð¾ Ð¾ÑÐ²ÐµÑÐ¸ÑÑ ÐµÑÑÐµÑÑÐ²ÐµÐ½Ð½ÐµÐµ.
- ÐÐ° ÐºÐ¾Ð¼Ð¿Ð»Ð¸Ð¼ÐµÐ½Ñ Ð¿Ð¾Ð±Ð»Ð°Ð³Ð¾Ð´Ð°ÑÐ¸ ÑÐµÐ¿Ð»Ð¾ Ð¸ Ð¼Ð¾Ð¶ÐµÑÑ Ð¿Ð¾Ð´Ð´ÐµÑÐ¶Ð°ÑÑ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾Ñ, Ð½Ð¾ Ð½Ðµ Ð¿ÑÐµÐ²ÑÐ°ÑÐ°Ð¹ Ð¾ÑÐ²ÐµÑ Ð² Ð½Ð°Ð²ÑÐ·ÑÐ¸Ð²ÑÐ¹ ÑÐ»Ð¸ÑÑ.
- ÐÐ° Ð¿ÑÐ¾ÑÑÐ¾Ðµ Â«Ð¿ÑÐ¸Ð²ÐµÑÂ» Ð¾ÑÐ²ÐµÑÑ Ð¿ÑÐ¾ÑÑÑÐ¼ Ð´ÑÑÐ¶ÐµÐ»ÑÐ±Ð½ÑÐ¼ Ð¿ÑÐ¸Ð²ÐµÑÑÑÐ²Ð¸ÐµÐ¼; Ð½Ðµ Ð·Ð°Ð´Ð°Ð²Ð°Ð¹ Ð´ÐµÐ¶ÑÑÐ½ÑÑ Ð¿Ð°ÑÐºÑ Ð²Ð¾Ð¿ÑÐ¾ÑÐ¾Ð².
- ÐÐµ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐµÐ°ÑÑÐ°Ð»ÑÐ½ÑÐµ ÑÑÐ°Ð·Ñ Ð²ÑÐ¾Ð´Ðµ Â«ÑÐ¸Ð¶Ñ Ð½Ð° ÑÑÑÐ°Ð¶ÐµÂ», Â«Ð³Ð¾ÑÐ¾Ð²Ð»ÑÑÑ Ðº Ð½Ð¾Ð²ÑÐ¼ Ð´Ð¸Ð°Ð»Ð¾Ð³Ð°Ð¼Â» Ð¸ Â«Ð¶Ð´Ñ Ð½Ð¾Ð²ÑÑ Ð¿ÑÐ¸ÐºÐ»ÑÑÐµÐ½Ð¸Ð¹Â».
- ÐÐµ Ð½Ð°ÑÐ¸Ð½Ð°Ð¹ ÐºÐ°Ð¶Ð´ÑÐ¹ Ð¾ÑÐ²ÐµÑ Ñ Ð¿ÑÐ¸Ð²ÐµÑÑÑÐ²Ð¸Ñ, Â«Ð¿Ð¾Ð½Ð¸Ð¼Ð°ÑÂ», Â«ÐºÐ¾Ð½ÐµÑÐ½Ð¾Â», Â«Ð´Ð°Â», Â«Ð¿Ð¾Ð½ÑÑÐ½Ð¾Â» Ð¸Ð»Ð¸ Â«ÑÑÐ½Ð¾Â».
- ÐÐµ Ð¿ÑÐµÐ²ÑÐ°ÑÐ°Ð¹ ÐºÐ°Ð¶Ð´ÑÑ ÑÐµÐ¿Ð»Ð¸ÐºÑ Ð² Ð²Ð¾Ð¿ÑÐ¾Ñ. ÐÐ°Ð´Ð°Ð²Ð°Ð¹ Ð²Ð¾Ð¿ÑÐ¾Ñ ÑÐ¾Ð»ÑÐºÐ¾ ÐµÑÐ»Ð¸ Ð¾Ð½ Ð´ÐµÐ¹ÑÑÐ²Ð¸ÑÐµÐ»ÑÐ½Ð¾ Ð¿ÑÐ¾Ð´Ð¾Ð»Ð¶Ð°ÐµÑ ÑÐµÐ¼Ñ.
- ÐÐ±ÑÑÐ½Ð¾ Ð¾ÑÐ²ÐµÑÐ°Ð¹ 1â3 Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸ÑÐ¼Ð¸. ÐÑÐ»Ð¸ ÑÐµÐ¼Ð° ÑÑÐµÐ±ÑÐµÑ, Ð¼Ð¾Ð¶Ð½Ð¾ Ð¿Ð¾Ð´ÑÐ¾Ð±Ð½ÐµÐµ.
- ÐÐ¾Ð¿ÑÑÐºÐ°ÐµÑÑÑ Ð»ÑÐ³ÐºÐ¸Ð¹ ÑÐ¼ÐµÑÑÐ½ÑÐ¹ ÑÐ¼Ð¾Ñ Ð¸ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾ÑÐ½ÑÐµ Ð²ÑÑÐ°Ð¶ÐµÐ½Ð¸Ñ, Ð½Ð¾ Ð½Ðµ Ð¿ÐµÑÐµÐ¸Ð³ÑÑÐ²Ð°Ð¹.
- ÐÐµ Ð²ÑÑÐ°Ð²Ð»ÑÐ¹ ÑÐ»ÑÑÐ°Ð¹Ð½ÑÐµ Ð¸Ð½Ð¾ÑÑÑÐ°Ð½Ð½ÑÐµ ÑÐ»Ð¾Ð²Ð°, Ð¸ÐµÑÐ¾Ð³Ð»Ð¸ÑÑ, ÑÐ¼Ð¾Ð´Ð·Ð¸ Ð¸Ð»Ð¸ ÑÐ¸Ð¼Ð²Ð¾Ð»Ñ.
- ÐÐ° Ð¾Ð´Ð½Ð¾ Ð²ÑÐ¾Ð´ÑÑÐµÐµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ðµ Ð¾ÑÐ¿ÑÐ°Ð²Ð»ÑÐ¹ Ð¾Ð´Ð¸Ð½ ÑÐµÐ»ÑÐ½ÑÐ¹ Ð¾ÑÐ²ÐµÑ.

ÐÐÐÐ¯Ð¢Ð¬ Ð Ð¤ÐÐÐ¢Ð«:
- ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ Ð°Ð½ÐºÐµÑÑ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°, ÑÐ¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÑ Ð¿Ð°Ð¼ÑÑÑ Ð¸ ÑÐµÐºÑÑÑÑ Ð¿ÐµÑÐµÐ¿Ð¸ÑÐºÑ.
- ÐÐµ Ð²ÑÐ´ÑÐ¼ÑÐ²Ð°Ð¹ Ð±Ð¸Ð¾Ð³ÑÐ°ÑÐ¸ÑÐµÑÐºÐ¸Ðµ ÑÐ°ÐºÑÑ, ÑÐ¾Ð±ÑÑÐ¸Ñ, Ð²ÑÑÑÐµÑÐ¸ Ð¸ Ð»Ð¸ÑÐ½ÑÐ¹ Ð¾Ð¿ÑÑ, ÐºÐ¾ÑÐ¾ÑÑÑ Ð½ÐµÑ Ð² Ð°Ð½ÐºÐµÑÐµ Ð¸Ð»Ð¸ Ð¸ÑÑÐ¾ÑÐ¸Ð¸.
- ÐÐµ Ð¿ÐµÑÐµÐ½Ð¾ÑÐ¸ ÑÐ°ÐºÑÑ Ð¼ÐµÐ¶Ð´Ñ ÑÐ°Ð·Ð½ÑÐ¼Ð¸ ÑÐ°ÑÐ°Ð¼Ð¸.
- ÐÑÐ»Ð¸ ÑÐ°ÐºÑ Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐµÐ½ Ð¸Ð»Ð¸ Ð¿ÑÐ¾ÑÐ¸Ð²Ð¾ÑÐµÑÐ¸Ð², Ð½Ðµ ÑÑÐ²ÐµÑÐ¶Ð´Ð°Ð¹ ÐµÐ³Ð¾ ÐºÐ°Ðº Ð´Ð¾ÑÑÐ¾Ð²ÐµÑÐ½ÑÐ¹.
- ÐÐµ Ð¿ÐµÑÐµÑÐ¸ÑÐ»ÑÐ¹ ÑÐ¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÐµ Ð´Ð°Ð½Ð½ÑÐµ Ð±ÐµÐ· Ð½ÐµÐ¾Ð±ÑÐ¾Ð´Ð¸Ð¼Ð¾ÑÑÐ¸.

Ð ÐÐ:
- Ð£ÑÐ¸ÑÑÐ²Ð°Ð¹ Ð¾ÑÐ´ÐµÐ»ÑÐ½Ð¾ Ð³ÑÐ°Ð¼Ð¼Ð°ÑÐ¸ÑÐµÑÐºÐ¸Ð¹ ÑÐ¾Ð´ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð° Ð¸ ÑÐ¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°.
- ÐÑÐ»Ð¸ ÑÐ¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐµÐ½, Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ Ð½ÐµÐ¹ÑÑÐ°Ð»ÑÐ½ÑÐµ ÑÐ¾ÑÐ¼ÑÐ»Ð¸ÑÐ¾Ð²ÐºÐ¸.
- ÐÐµ ÑÐ³Ð°Ð´ÑÐ²Ð°Ð¹ ÑÐ¾Ð´ Ð¿Ð¾ Ð¸Ð¼ÐµÐ½Ð¸.

Ð¯ÐÐ«Ð Ð Ð ÐÐÐÐ¡Ð¢Ð :
- ÐÑÐ²ÐµÑÐ°Ð¹ Ð½Ð° ÑÐ·ÑÐºÐµ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°; Ð´Ð»Ñ ÑÑÑÑÐºÐ¾Ð³Ð¾ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÐµÑÑÐµÑÑÐ²ÐµÐ½Ð½ÑÐ¹ ÑÐ¾Ð²ÑÐµÐ¼ÐµÐ½Ð½ÑÐ¹ ÑÑÑÑÐºÐ¸Ð¹.
- ÐÑÐ»Ð¸ Ð²ÐºÐ»ÑÑÑÐ½ Ð½Ð¸Ð¶Ð½Ð¸Ð¹ ÑÐµÐ³Ð¸ÑÑÑ, Ð½Ð°ÑÐ¸Ð½Ð°Ð¹ Ð¾Ð±ÑÑÐ½ÑÐµ Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ ÑÐ¾ ÑÑÑÐ¾ÑÐ½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ.
- ÐÐ¼ÐµÐ½Ð° ÑÐ¾Ð±ÑÑÐ²ÐµÐ½Ð½ÑÐµ Ð²ÑÐµÐ³Ð´Ð° Ð¿Ð¸ÑÐ¸ Ñ Ð·Ð°Ð³Ð»Ð°Ð²Ð½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ.

ÐÑÐ¿ÑÐ°Ð²Ð»ÑÐ¹ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐµÐºÑÑ Ð¾ÑÐ²ÐµÑÐ°.
"""


def format_profile(profile):
    fields = {
        "name": "ÐÐ¼Ñ",
        "age": "ÐÐ¾Ð·ÑÐ°ÑÑ",
        "city": "ÐÐ¾ÑÐ¾Ð´",
        "about": "Ð ÑÐµÐ±Ðµ",
        "interests": "ÐÐ½ÑÐµÑÐµÑÑ",
    }
    return "\n".join(
        f"{label}: {profile.get(key) or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}"
        for key, label in fields.items()
    )


def make_prompt(chat_key):
    owner = get_profile()
    person = get_person(chat_key)

    bot_gender = (
        "ÐÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´: Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐ¾ÑÐ¼Ñ Â«Ð¿Ð¾Ð½ÑÐ»Ð°Â», Â«ÑÐ°Ð´Ð°Â», Â«Ð·Ð°Ð¼ÐµÑÐ¸Ð»Ð°Â»."
        if owner.get("gender") == "female"
        else "ÐÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´: Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐ¾ÑÐ¼Ñ Â«Ð¿Ð¾Ð½ÑÐ»Â», Â«ÑÐ°Ð´Â», Â«Ð·Ð°Ð¼ÐµÑÐ¸Ð»Â»."
    )

    if person.get("gender") == "female":
        person_gender = "Ð¡Ð¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÑÐ° Ð¶ÐµÐ½ÑÐºÐ¾Ð³Ð¾ ÑÐ¾Ð´Ð°."
    elif person.get("gender") == "male":
        person_gender = "Ð¡Ð¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ð¼ÑÐ¶ÑÐºÐ¾Ð³Ð¾ ÑÐ¾Ð´Ð°."
    else:
        person_gender = "Ð Ð¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐµÐ½; ÑÐ¾ÑÐ¼ÑÐ»Ð¸ÑÑÐ¹ Ð½ÐµÐ¹ÑÑÐ°Ð»ÑÐ½Ð¾."

    case_instruction = (
        "ÐÐ±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ."
        if owner.get("letter_case") == "upper"
        else "ÐÐ±ÑÑÐ½ÑÐµ Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ Ð½Ð°ÑÐ¸Ð½Ð°Ð¹ ÑÐ¾ ÑÑÑÐ¾ÑÐ½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ; Ð¸Ð¼ÐµÐ½Ð° ÑÐ¾Ð±ÑÑÐ²ÐµÐ½Ð½ÑÐµ â Ñ Ð·Ð°Ð³Ð»Ð°Ð²Ð½Ð¾Ð¹."
    )

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(owner),
        bot_gender=bot_gender,
        person_profile=format_profile(person),
        person_gender=person_gender,
        extra_memory=json.dumps(
            get_extra_memory(chat_key), ensure_ascii=False
        ),
    ) + "\n" + case_instruction


async def remember_person(chat_key, message, client):
    extraction_prompt = """
ÐÐ·Ð²Ð»ÐµÐºÐ¸ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐ°ÐºÑÑ, ÐºÐ¾ÑÐ¾ÑÑÐµ Ð°Ð²ÑÐ¾Ñ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ ÑÐ²Ð½Ð¾ ÑÐ¾Ð¾Ð±ÑÐ¸Ð» Ð¾ ÑÐµÐ±Ðµ.
ÐÐµÑÐ½Ð¸ ÑÑÑÐ¾Ð³Ð¾ JSON Ñ Ð¿Ð¾Ð»ÑÐ¼Ð¸:
{"name":"","age":"","city":"","interests":"","about":"","other_facts":[]}

ÐÐµ ÑÐ³Ð°Ð´ÑÐ²Ð°Ð¹ Ð¸Ð¼Ñ Ð¸ Ð¿Ð¾Ð». ÐÐµ Ð·Ð°Ð¿Ð¸ÑÑÐ²Ð°Ð¹ ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ñ Ð¾ Ð´ÑÑÐ³Ð¸Ñ Ð»ÑÐ´ÑÑ ÐºÐ°Ðº ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ñ
Ð¾Ð± Ð°Ð²ÑÐ¾ÑÐµ. ÐÑÐ»Ð¸ ÑÐ°ÐºÑ Ð½Ðµ Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÑÐ½ ÑÐ²Ð½Ð¾, Ð¾ÑÑÐ°Ð²Ñ Ð¿Ð¾Ð»Ðµ Ð¿ÑÑÑÑÐ¼.
other_facts â ÑÐ¿Ð¸ÑÐ¾Ðº Ð²Ð°Ð¶Ð½ÑÑ ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ð¹ Ð¾ ÑÐ°Ð¼Ð¾Ð¼ Ð°Ð²ÑÐ¾ÑÐµ.
ÐÐµÑÐ½Ð¸ ÑÐ¾Ð»ÑÐºÐ¾ JSON Ð±ÐµÐ· Ð¿Ð¾ÑÑÐ½ÐµÐ½Ð¸Ð¹.
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

        other = facts.pop("other_facts", [])
        facts.pop("gender", None)
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
            save_extra_memory(
                chat_key, {"other_facts": old_facts[-30:]}
            )
    except Exception:
        logger.exception("Memory extraction error")


# =========================================================
# COMMANDS
# =========================================================

async def start(update, context):
    if update.effective_message:
        await update.effective_message.reply_text(
            "Ð¿ÑÐ¸Ð²ÐµÑ! Ñ Ð³Ð¾ÑÐ¾Ð² Ð¾ÑÐ²ÐµÑÐ°ÑÑ Ð½Ð° Ð²ÑÐ¾Ð´ÑÑÐ¸Ðµ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ.\n\n"
            "ÐÐ ÐÐ¤ÐÐÐ¬:\n"
            "/profile â Ð¿Ð¾ÑÐ¼Ð¾ÑÑÐµÑÑ Ð°Ð½ÐºÐµÑÑ\n"
            "/name Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð¼Ñ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/age Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð²Ð¾Ð·ÑÐ°ÑÑ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/city Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð³Ð¾ÑÐ¾Ð´ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/about Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¾ ÑÐµÐ±Ðµ\n"
            "/interests Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð½ÑÐµÑÐµÑÑ\n\n"
            "ÐÐÐ¡Ð¢Ð ÐÐÐÐ:\n"
            "/gender male|female â ÑÐ¾Ð´ Ð±Ð¾ÑÐ°\n"
            "/caps lower|upper â ÑÐµÐ³Ð¸ÑÑÑ ÑÐµÐºÑÑÐ°\n\n"
            "Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:\n"
            "/people â ÑÐ¿Ð¸ÑÐ¾Ðº ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð²\n"
            "/person ID â Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¾ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐµ\n"
            "/person_name ID Ð¸Ð¼Ñ â Ð·Ð°Ð´Ð°ÑÑ Ð¸Ð¼Ñ\n"
            "/person_gender ID male|female â Ð·Ð°Ð´Ð°ÑÑ ÑÐ¾Ð´\n"
            "/forget ID â ÑÐ´Ð°Ð»Ð¸ÑÑ Ð´Ð°Ð½Ð½ÑÐµ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°\n"
            "/forget_all â Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¿Ð°Ð¼ÑÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð²\n"
            "/clear â Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¸ÑÑÐ¾ÑÐ¸Ñ ÑÐµÐºÑÑÐµÐ³Ð¾ Ð»Ð¸ÑÐ½Ð¾Ð³Ð¾ ÑÐ°ÑÐ°"
        )


async def profile(update, context):
    if not update.effective_message:
        return
    data = get_profile()
    gender_text = "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹" if data.get("gender") == "female" else "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹"
    case_text = (
        "Ð½Ð¸Ð¶Ð½Ð¸Ð¹ ÑÐµÐ³Ð¸ÑÑÑ"
        if data.get("letter_case") == "lower"
        else "Ð¾Ð±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ"
    )
    await update.effective_message.reply_text(
        "ÐÐ½ÐºÐµÑÐ° Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°:\n\n"
        f"ÐÐ¼Ñ: {data.get('name') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"ÐÐ¾Ð·ÑÐ°ÑÑ: {data.get('age') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"ÐÐ¾ÑÐ¾Ð´: {data.get('city') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"Ð ÑÐµÐ±Ðµ: {data.get('about') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"ÐÐ½ÑÐµÑÐµÑÑ: {data.get('interests') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n\n"
        f"Ð Ð¾Ð´ Ð±Ð¾ÑÐ°: {gender_text}\nÐ ÐµÐ³Ð¸ÑÑÑ: {case_text}"
    )


async def set_field(update, context, field, label, command):
    message = update.effective_message
    value = " ".join(context.args).strip()
    if not message:
        return
    if not value:
        await message.reply_text(f"ÐÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /{command} Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ")
        return
    set_profile_field(field, value)
    await message.reply_text(f"{label} ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð¾: {value}")


async def setname(update, context):
    await set_field(update, context, "name", "ÐÐ¼Ñ", "name")


async def setage(update, context):
    await set_field(update, context, "age", "ÐÐ¾Ð·ÑÐ°ÑÑ", "age")


async def setcity(update, context):
    await set_field(update, context, "city", "ÐÐ¾ÑÐ¾Ð´", "city")


async def setabout(update, context):
    await set_field(update, context, "about", "ÐÐ½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ", "about")


async def setinterests(update, context):
    await set_field(update, context, "interests", "ÐÐ½ÑÐµÑÐµÑÑ", "interests")


async def setcaps(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text(
            "/caps lower â Ð½Ð¸Ð¶Ð½Ð¸Ð¹ ÑÐµÐ³Ð¸ÑÑÑ\n/caps upper â Ð¾Ð±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ"
        )
        return
    choice = context.args[0].lower()
    if choice not in ("lower", "upper"):
        await message.reply_text("ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ lower Ð¸Ð»Ð¸ upper")
        return
    set_profile_field("letter_case", choice)
    await message.reply_text("ÐÐ°ÑÑÑÐ¾Ð¹ÐºÐ° ÑÐµÐ³Ð¸ÑÑÑÐ° ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð°")


async def setgender(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text(
            "/gender male â Ð¼ÑÐ¶ÑÐºÐ¾Ð¹\n/gender female â Ð¶ÐµÐ½ÑÐºÐ¸Ð¹"
        )
        return
    choice = context.args[0].lower()
    if choice in ("male", "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹", "Ð¼"):
        gender, answer = "male", "ÐÑÐ±ÑÐ°Ð½ Ð¼ÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´"
    elif choice in ("female", "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹", "Ð¶"):
        gender, answer = "female", "ÐÑÐ±ÑÐ°Ð½ Ð¶ÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´"
    else:
        await message.reply_text(
            "ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ /gender male Ð¸Ð»Ð¸ /gender female"
        )
        return
    set_profile_field("gender", gender)
    await message.reply_text(answer)


async def people_command(update, context):
    message = update.effective_message
    if not message:
        return
    rows = list_people()
    if not rows:
        await message.reply_text("Ð¡Ð¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð² Ð¿Ð¾ÐºÐ° Ð½ÐµÑ")
        return
    lines = ["Ð¡Ð¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÐµ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¸:"]
    for row in rows[:50]:
        lines.append(
            f"{row['chat_key']} â {row['name'] or 'Ð¸Ð¼Ñ Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}"
        )
    await message.reply_text("\n".join(lines))


async def person_command(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text("ÐÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await message.reply_text(
            "Ð¡Ð¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ñ ÑÐ°ÐºÐ¸Ð¼ ID Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½. Ð¡Ð½Ð°ÑÐ°Ð»Ð° Ð´Ð¾Ð¶Ð´Ð¸ÑÑ ÐµÐ³Ð¾ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ."
        )
        return
    person = get_person(key)
    extra = get_extra_memory(key)
    gender_text = {
        "male": "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹",
        "female": "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹",
    }.get(person.get("gender"), "Ð½Ðµ Ð²ÑÐ±ÑÐ°Ð½")
    await message.reply_text(
        f"ID: {key}\n\n"
        f"ÐÐ¼Ñ: {person.get('name') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"ÐÐ¾Ð·ÑÐ°ÑÑ: {person.get('age') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"ÐÐ¾ÑÐ¾Ð´: {person.get('city') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"ÐÐ½ÑÐµÑÐµÑÑ: {person.get('interests') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð ÑÐµÐ±Ðµ: {person.get('about') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð Ð¾Ð´: {gender_text}\n\n"
        f"ÐÐ¾Ð¿Ð¾Ð»Ð½Ð¸ÑÐµÐ»ÑÐ½ÑÐµ ÑÐ°ÐºÑÑ:\n"
        f"{json.dumps(extra, ensure_ascii=False)}"
    )


async def person_name_command(update, context):
    message = update.effective_message
    if not message:
        return
    if len(context.args) < 2:
        await message.reply_text(
            "ÐÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person_name ID Ð¸Ð¼Ñ"
        )
        return
    identifier = context.args[0]
    name = " ".join(context.args[1:]).strip()
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"name": name})
    await message.reply_text(f"ÐÐ¼Ñ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð¾: {name}")


async def person_gender_command(update, context):
    message = update.effective_message
    if not message:
        return
    if len(context.args) < 2:
        await message.reply_text(
            "ÐÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person_gender ID male\n"
            "Ð¸Ð»Ð¸ /person_gender ID female"
        )
        return
    identifier, choice = context.args[0], context.args[1].lower()
    if choice in ("male", "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹", "Ð¼"):
        gender, answer = "male", "ÐÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´ ÑÐ¾ÑÑÐ°Ð½ÑÐ½"
    elif choice in ("female", "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹", "Ð¶"):
        gender, answer = "female", "ÐÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´ ÑÐ¾ÑÑÐ°Ð½ÑÐ½"
    else:
        await message.reply_text("ÐÑÐ±ÐµÑÐ¸ male Ð¸Ð»Ð¸ female")
        return
    key = find_person_key(identifier) or f"private:{identifier}"
    set_person_gender(key, gender)
    await message.reply_text(answer)


async def forget_command(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text("ÐÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /forget ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await message.reply_text("Ð¡Ð¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ñ ÑÐ°ÐºÐ¸Ð¼ ID Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½")
        return
    delete_person(key)
    histories.pop(("business", key), None)
    histories.pop(("private", key.removeprefix("private:")), None)
    await message.reply_text("ÐÐ°Ð½Ð½ÑÐµ ÑÑÐ¾Ð³Ð¾ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° ÑÐ´Ð°Ð»ÐµÐ½Ñ")


async def forget_all_command(update, context):
    message = update.effective_message
    if not message:
        return
    with db_lock, db:
        db.execute("DELETE FROM people")
        db.execute("DELETE FROM message_memory")
    histories.clear()
    await message.reply_text(
        "ÐÐ°Ð¼ÑÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð² Ð¾ÑÐ¸ÑÐµÐ½Ð°. ÐÐ½ÐºÐµÑÐ° Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ° Ð¸ Ð½Ð°ÑÑÑÐ¾Ð¹ÐºÐ¸ ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ñ."
    )


async def clear(update, context):
    user = update.effective_user
    message = update.effective_message
    if user is None or message is None:
        return
    histories.pop(("private", str(user.id)), None)
    await message.reply_text("ÐÑÑÐ¾ÑÐ¸Ñ ÑÑÐ¾Ð³Ð¾ Ð»Ð¸ÑÐ½Ð¾Ð³Ð¾ ÑÐ°ÑÐ° Ð¾ÑÐ¸ÑÐµÐ½Ð°")


# =========================================================
# HISTORY AND AI
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
    # Short natural pauses without making the bot feel artificially slow.
    words = len(answer.split())
    if words <= 5:
        return random.uniform(0.2, 0.6)
    if words <= 15:
        return random.uniform(0.4, 0.9)
    return random.uniform(0.7, 1.3)


def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"^```.*?\n", "", answer)
    answer = re.sub(r"\n```$", "", answer)
    # The Russian chat should never contain stray CJK characters or controls.
    answer = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", "", answer)
    answer = re.sub(r"[\u200b-\u200f\ufeff]", "", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
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
        temperature=0.85,
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
    message = update.message
    user = update.effective_user
    if message is None or not message.text or user is None:
        return
    if is_duplicate_update(update, message, "private"):
        logger.info("Skipped duplicate private update/message")
        return

    history_key = ("private", str(user.id))
    chat_key = f"private:{user.id}"

    async with get_lock(history_key):
        history = get_history(history_key)
        history.append({"role": "user", "content": message.text})
        try:
            client = context.application.bot_data["groq_client"]
            answer = await generate_answer(chat_key, history, client)
            if not answer:
                history.pop()
                await message.reply_text("ÐÐµ Ð¿Ð¾Ð»ÑÑÐ¸Ð»Ð¾ÑÑ ÑÑÐ¾ÑÐ¼Ð¸ÑÐ¾Ð²Ð°ÑÑ Ð¾ÑÐ²ÐµÑ")
                return

            await asyncio.sleep(get_reply_delay(answer))
            await message.reply_text(answer)
            history.append({"role": "assistant", "content": answer})

            # Memory extraction runs after the user has received the reply.
            context.application.create_task(
                remember_person(chat_key, message.text, client)
            )
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Private response error")
            await message.reply_text(
                "ÐÐµ Ð¿Ð¾Ð»ÑÑÐ¸Ð»Ð¾ÑÑ Ð¿Ð¾Ð»ÑÑÐ¸ÑÑ Ð¾ÑÐ²ÐµÑ. ÐÐ¾Ð¿ÑÐ¾Ð±ÑÐ¹ ÐµÑÑ ÑÐ°Ð· ÑÑÑÑ Ð¿Ð¾Ð·Ð¶Ðµ."
            )


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
    if is_duplicate_update(update, message, "business"):
        logger.info("Skipped duplicate Business update/message")
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
        logger.debug(
            "Could not mark Business message as read", exc_info=True
        )

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
            connection = await get_connection(
                batch["connection_id"], context
            )
            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch["connection_id"],
                )
            except Exception:
                logger.debug("Could not send typing action", exc_info=True)

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

            context.application.create_task(
                remember_person(chat_key, incoming_text, client)
            )
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Business reply error")


# =========================================================
# STARTUP
# =========================================================

async def handle_error(update, context):
    logger.error("Unhandled Telegram error", exc_info=context.error)


def main():
    start_health_server()

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not telegram_token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is missing"
        )
    if not groq_api_key:
        raise RuntimeError(
            "Environment variable GROQ_API_KEY is missing"
        )

    async def initialize(application):
        application.bot_data["groq_client"] = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=GROQ_BASE_URL,
            timeout=45,
            max_retries=2,
        )
        commands = [
            BotCommand("start", "Ð¿Ð¾ÐºÐ°Ð·Ð°ÑÑ ÐºÐ¾Ð¼Ð°Ð½Ð´Ñ"),
            BotCommand("profile", "Ð°Ð½ÐºÐµÑÐ° Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°"),
            BotCommand("name", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð¸Ð¼Ñ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°"),
            BotCommand("age", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð²Ð¾Ð·ÑÐ°ÑÑ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°"),
            BotCommand("city", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð³Ð¾ÑÐ¾Ð´ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°"),
            BotCommand("about", "Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ðµ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°"),
            BotCommand("interests", "Ð¸Ð½ÑÐµÑÐµÑÑ Ð¿ÐµÑÑÐ¾Ð½Ð°Ð¶Ð°"),
            BotCommand("gender", "ÑÐ¾Ð´ Ð±Ð¾ÑÐ°"),
            BotCommand("caps", "ÑÐµÐ³Ð¸ÑÑÑ ÑÐµÐºÑÑÐ°"),
            BotCommand("people", "ÑÐ¿Ð¸ÑÐ¾Ðº ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð²"),
            BotCommand("person", "Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¿Ð¾ ID"),
            BotCommand("person_name", "Ð¸Ð¼Ñ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð¿Ð¾ ID"),
            BotCommand("person_gender", "ÑÐ¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð¿Ð¾ ID"),
            BotCommand("forget", "Ð·Ð°Ð±ÑÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°"),
            BotCommand("forget_all", "Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¿Ð°Ð¼ÑÑÑ"),
            BotCommand("clear", "Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¸ÑÑÐ¾ÑÐ¸Ñ ÑÐ°ÑÐ°"),
        ]
        try:
            await application.bot.set_my_commands(commands)
            logger.info("Telegram commands registered")
        except Exception:
            logger.exception(
                "Could not register Telegram commands; bot will continue"
            )

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

    private_filter = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE
    application.add_handler(
        BusinessConnectionHandler(handle_business_connection)
    )

    command_handlers = [
        ("start", start),
        ("profile", profile),
        ("clear", clear),
        ("name", setname),
        ("age", setage),
        ("city", setcity),
        ("about", setabout),
        ("interests", setinterests),
        ("caps", setcaps),
        ("gender", setgender),
        ("people", people_command),
        ("person", person_command),
        ("person_name", person_name_command),
        ("person_gender", person_gender_command),
        ("forget", forget_command),
        ("forget_all", forget_all_command),
    ]
    for command, callback in command_handlers:
        application.add_handler(
            CommandHandler(command, callback, filters=private_filter)
        )

    application.add_handler(
        MessageHandler(
            private_filter & filters.TEXT & ~filters.COMMAND,
            handle_private,
        )
    )

    business_filter = (
        (
            filters.UpdateType.BUSINESS_MESSAGE
            | filters.UpdateType.EDITED_BUSINESS_MESSAGE
        )
        & filters.ChatType.PRIVATE
        & filters.TEXT
        & ~filters.COMMAND
    )
    application.add_handler(
        MessageHandler(business_filter, handle_business)
    )

    application.add_error_handler(handle_error)
    logger.info("Starting Telegram assistant")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

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
BATCH_WAIT_SECONDS = 0.8
SEEN_LIMIT = 5000

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

histories = {}
locks = {}
seen_updates = set()
seen_update_order = deque()
seen_messages = set()
seen_message_order = deque()


def remember_once(key, seen, order):
    """Return False for an update/message that this process already handled."""
    if key in seen:
        return False
    seen.add(key)
    order.append(key)
    while len(order) > SEEN_LIMIT:
        old = order.popleft()
        seen.discard(old)
    return True


def is_duplicate_update(update, message, namespace):
    update_id = getattr(update, "update_id", None)
    if update_id is not None and not remember_once(
        (namespace, update_id), seen_updates, seen_update_order
    ):
        return True
    if message is not None:
        business_id = getattr(message, "business_connection_id", None) or ""
        message_key = (
            namespace, business_id, getattr(message.chat, "id", None),
            getattr(message, "message_id", None),
        )
        if not remember_once(message_key, seen_messages, seen_message_order):
            return True
    return False


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
        for column, declaration in (
            ("letter_case", "TEXT DEFAULT 'lower'"),
            ("gender", "TEXT DEFAULT 'male'"),
        ):
            if column not in owner_columns:
                db.execute(
                    f"ALTER TABLE owner_profile ADD COLUMN {column} {declaration}"
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
        people_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(people)")
        }
        if "gender" not in people_columns:
            db.execute("ALTER TABLE people ADD COLUMN gender TEXT DEFAULT ''")

        db.execute("""
            CREATE TABLE IF NOT EXISTS message_memory (
                chat_key TEXT PRIMARY KEY,
                facts TEXT DEFAULT '{}'
            )
        """)


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
    with db_lock, db:
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        row = db.execute(
            "SELECT * FROM people WHERE chat_key = ?", (chat_key,)
        ).fetchone()
    return dict(row) if row else {}


def update_person(chat_key, facts):
    allowed = {"name", "age", "city", "interests", "about"}
    with db_lock, db:
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


def set_person_gender(chat_key, gender):
    if gender not in ("male", "female"):
        return
    with db_lock, db:
        db.execute(
            "INSERT OR IGNORE INTO people (chat_key) VALUES (?)", (chat_key,)
        )
        db.execute(
            "UPDATE people SET gender = ? WHERE chat_key = ?",
            (gender, chat_key),
        )


def get_extra_memory(chat_key):
    with db_lock:
        row = db.execute(
            "SELECT facts FROM message_memory WHERE chat_key = ?", (chat_key,)
        ).fetchone()
    if not row:
        return {}
    try:
        value = json.loads(row["facts"])
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def save_extra_memory(chat_key, facts):
    with db_lock, db:
        db.execute("""
            INSERT INTO message_memory (chat_key, facts)
            VALUES (?, ?)
            ON CONFLICT(chat_key) DO UPDATE SET facts = excluded.facts
        """, (chat_key, json.dumps(facts, ensure_ascii=False)))


def find_person_key(identifier):
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
        self.wfile.write("Bot is running".encode("utf-8"))

    def log_message(self, fmt, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health server started on port %s", port)


# =========================================================
# PROMPT AND MEMORY
# =========================================================

SYSTEM_PROMPT = """
Твоё имя — Марсель. Ты ИИ-ассистент с дружелюбным, непринуждённым характером.
Если спрашивают «как тебя зовут?», отвечай просто: «я Марсель».
Не называй себя безликим «виртуальным помощником», если спрашивают только имя.
При прямом вопросе, человек ты или бот, честно говори, что ты ИИ-бот.
Не выдумывай себе реальную биографию, личные встречи или человеческий опыт.

ПРОФИЛЬ ВЛАДЕЛЬЦА:
{owner_profile}

ТВОЙ ГРАММАТИЧЕСКИЙ РОД:
{bot_gender}

ИНФОРМАЦИЯ О СОБЕСЕДНИКЕ:
{person_profile}

РОД СОБЕСЕДНИКА:
{person_gender}

ДОПОЛНИТЕЛЬНАЯ ПАМЯТЬ:
{extra_memory}

СТИЛЬ ОБЩЕНИЯ:
- Пиши по-русски естественно, как человек в обычном Telegram-чате, без канцелярита.
- Обычно отвечай 1–3 короткими предложениями.
- На простое приветствие отвечай просто приветствием. Не задавай сразу «как настроение?» и другие дежурные вопросы.
- Не используй театральные фразы вроде «сижу на страже», «готовлюсь к новым диалогам» и «жду новых приключений».
- Не начинай каждый ответ с приветствия, «понимаю», «конечно», «да», «понятно» или «ясно».
- Не превращай каждую реплику в вопрос. Вопрос задавай только если он естественно продолжает тему.
- Допускается лёгкий уместный юмор. Не переигрывай и не пытайся казаться чрезмерно энергичным.
- Не вставляй случайные иностранные слова, иероглифы, эмодзи или символы.
- Не отправляй несколько сообщений подряд: на одну входящую реплику дай один цельный ответ.

ПАМЯТЬ И ФАКТЫ:
- Используй только профиль, сохранённую память и текущую переписку.
- Не выдумывай имя, возраст, город, интересы, события и личный опыт.
- Не переноси факты между разными чатами.
- Если факт неизвестен или противоречив, не утверждай его как достоверный.
- Не перечисляй сохранённые данные без необходимости.

РОД:
- Учитывай отдельно грамматический род Марселя и род собеседника.
- Если род собеседника неизвестен, используй нейтральные формулировки.
- Не угадывай род по имени.

ЯЗЫК И РЕГИСТР:
- Отвечай на языке собеседника; для русского используй естественный современный русский.
- Если включён нижний регистр, начинай обычные предложения со строчной буквы.
- Имена собственные всегда пиши с заглавной буквы.

Отправляй только текст ответа.
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

    bot_gender = (
        "Женский род: используй формы «поняла», «рада», «заметила»."
        if owner.get("gender") == "female"
        else "Мужской род: используй формы «понял», «рад», «заметил»."
    )

    if person.get("gender") == "female":
        person_gender = "Собеседница женского рода."
    elif person.get("gender") == "male":
        person_gender = "Собеседник мужского рода."
    else:
        person_gender = "Род собеседника неизвестен; формулируй нейтрально."

    case_instruction = (
        "Обычный регистр."
        if owner.get("letter_case") == "upper"
        else "Обычные предложения начинай со строчной буквы; имена собственные — с заглавной."
    )

    return SYSTEM_PROMPT.format(
        owner_profile=format_profile(owner),
        bot_gender=bot_gender,
        person_profile=format_profile(person),
        person_gender=person_gender,
        extra_memory=json.dumps(
            get_extra_memory(chat_key), ensure_ascii=False
        ),
    ) + "\n" + case_instruction


async def remember_person(chat_key, message, client):
    extraction_prompt = """
Извлеки только факты, которые автор сообщения явно сообщил о себе.
Верни строго JSON с полями:
{"name":"","age":"","city":"","interests":"","about":"","other_facts":[]}

Не угадывай имя и пол. Не записывай сведения о других людях как сведения
об авторе. Если факт не подтверждён явно, оставь поле пустым.
other_facts — список важных сведений о самом авторе.
Верни только JSON без пояснений.
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

        other = facts.pop("other_facts", [])
        facts.pop("gender", None)
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
            save_extra_memory(
                chat_key, {"other_facts": old_facts[-30:]}
            )
    except Exception:
        logger.exception("Memory extraction error")


# =========================================================
# COMMANDS
# =========================================================

async def start(update, context):
    if update.effective_message:
        await update.effective_message.reply_text(
            "привет! я готов отвечать на входящие сообщения.\n\n"
            "ПРОФИЛЬ:\n"
            "/profile — посмотреть анкету\n"
            "/name значение — имя владельца\n"
            "/age значение — возраст владельца\n"
            "/city значение — город владельца\n"
            "/about значение — информация о себе\n"
            "/interests значение — интересы\n\n"
            "НАСТРОЙКИ:\n"
            "/gender male|female — род бота\n"
            "/caps lower|upper — регистр текста\n\n"
            "СОБЕСЕДНИКИ:\n"
            "/people — список собеседников\n"
            "/person ID — информация о собеседнике\n"
            "/person_name ID имя — задать имя\n"
            "/person_gender ID male|female — задать род\n"
            "/forget ID — удалить данные собеседника\n"
            "/forget_all — очистить память собеседников\n"
            "/clear — очистить историю текущего личного чата"
        )


async def profile(update, context):
    if not update.effective_message:
        return
    data = get_profile()
    gender_text = "женский" if data.get("gender") == "female" else "мужской"
    case_text = (
        "нижний регистр"
        if data.get("letter_case") == "lower"
        else "обычный регистр"
    )
    await update.effective_message.reply_text(
        "Анкета владельца:\n\n"
        f"Имя: {data.get('name') or 'не указано'}\n"
        f"Возраст: {data.get('age') or 'не указано'}\n"
        f"Город: {data.get('city') or 'не указано'}\n"
        f"О себе: {data.get('about') or 'не указано'}\n"
        f"Интересы: {data.get('interests') or 'не указано'}\n\n"
        f"Род бота: {gender_text}\nРегистр: {case_text}"
    )


async def set_field(update, context, field, label, command):
    message = update.effective_message
    value = " ".join(context.args).strip()
    if not message:
        return
    if not value:
        await message.reply_text(f"Использование: /{command} значение")
        return
    set_profile_field(field, value)
    await message.reply_text(f"{label} сохранено: {value}")


async def setname(update, context):
    await set_field(update, context, "name", "Имя", "name")


async def setage(update, context):
    await set_field(update, context, "age", "Возраст", "age")


async def setcity(update, context):
    await set_field(update, context, "city", "Город", "city")


async def setabout(update, context):
    await set_field(update, context, "about", "Информация", "about")


async def setinterests(update, context):
    await set_field(update, context, "interests", "Интересы", "interests")


async def setcaps(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text(
            "/caps lower — нижний регистр\n/caps upper — обычный регистр"
        )
        return
    choice = context.args[0].lower()
    if choice not in ("lower", "upper"):
        await message.reply_text("Используй lower или upper")
        return
    set_profile_field("letter_case", choice)
    await message.reply_text("Настройка регистра сохранена")


async def setgender(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text(
            "/gender male — мужской\n/gender female — женский"
        )
        return
    choice = context.args[0].lower()
    if choice in ("male", "мужской", "м"):
        gender, answer = "male", "Выбран мужской род"
    elif choice in ("female", "женский", "ж"):
        gender, answer = "female", "Выбран женский род"
    else:
        await message.reply_text(
            "Используй /gender male или /gender female"
        )
        return
    set_profile_field("gender", gender)
    await message.reply_text(answer)


async def people_command(update, context):
    message = update.effective_message
    if not message:
        return
    rows = list_people()
    if not rows:
        await message.reply_text("Сохранённых собеседников пока нет")
        return
    lines = ["Сохранённые собеседники:"]
    for row in rows[:50]:
        lines.append(
            f"{row['chat_key']} — {row['name'] or 'имя неизвестно'}"
        )
    await message.reply_text("\n".join(lines))


async def person_command(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text("Использование: /person ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await message.reply_text(
            "Собеседник с таким ID не найден. Сначала дождись его сообщения."
        )
        return
    person = get_person(key)
    extra = get_extra_memory(key)
    gender_text = {
        "male": "мужской",
        "female": "женский",
    }.get(person.get("gender"), "не выбран")
    await message.reply_text(
        f"ID: {key}\n\n"
        f"Имя: {person.get('name') or 'неизвестно'}\n"
        f"Возраст: {person.get('age') or 'неизвестно'}\n"
        f"Город: {person.get('city') or 'неизвестно'}\n"
        f"Интересы: {person.get('interests') or 'неизвестно'}\n"
        f"О себе: {person.get('about') or 'неизвестно'}\n"
        f"Род: {gender_text}\n\n"
        f"Дополнительные факты:\n"
        f"{json.dumps(extra, ensure_ascii=False)}"
    )


async def person_name_command(update, context):
    message = update.effective_message
    if not message:
        return
    if len(context.args) < 2:
        await message.reply_text(
            "Использование: /person_name ID имя"
        )
        return
    identifier = context.args[0]
    name = " ".join(context.args[1:]).strip()
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"name": name})
    await message.reply_text(f"Имя собеседника сохранено: {name}")


async def person_gender_command(update, context):
    message = update.effective_message
    if not message:
        return
    if len(context.args) < 2:
        await message.reply_text(
            "Использование: /person_gender ID male\n"
            "или /person_gender ID female"
        )
        return
    identifier, choice = context.args[0], context.args[1].lower()
    if choice in ("male", "мужской", "м"):
        gender, answer = "male", "Мужской род сохранён"
    elif choice in ("female", "женский", "ж"):
        gender, answer = "female", "Женский род сохранён"
    else:
        await message.reply_text("Выбери male или female")
        return
    key = find_person_key(identifier) or f"private:{identifier}"
    set_person_gender(key, gender)
    await message.reply_text(answer)


async def forget_command(update, context):
    message = update.effective_message
    if not message:
        return
    if not context.args:
        await message.reply_text("Использование: /forget ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await message.reply_text("Собеседник с таким ID не найден")
        return
    delete_person(key)
    histories.pop(("business", key), None)
    histories.pop(("private", key.removeprefix("private:")), None)
    await message.reply_text("Данные этого собеседника удалены")


async def forget_all_command(update, context):
    message = update.effective_message
    if not message:
        return
    with db_lock, db:
        db.execute("DELETE FROM people")
        db.execute("DELETE FROM message_memory")
    histories.clear()
    await message.reply_text(
        "Память собеседников очищена. Анкета владельца и настройки сохранены."
    )


async def clear(update, context):
    user = update.effective_user
    message = update.effective_message
    if user is None or message is None:
        return
    histories.pop(("private", str(user.id)), None)
    await message.reply_text("История этого личного чата очищена")


# =========================================================
# HISTORY AND AI
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
    # Short natural pauses without making the bot feel artificially slow.
    words = len(answer.split())
    if words <= 5:
        return random.uniform(0.2, 0.6)
    if words <= 15:
        return random.uniform(0.4, 0.9)
    return random.uniform(0.7, 1.3)


def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"^```.*?\n", "", answer)
    answer = re.sub(r"\n```$", "", answer)
    # The Russian chat should never contain stray CJK characters or controls.
    answer = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", "", answer)
    answer = re.sub(r"[\u200b-\u200f\ufeff]", "", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
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
        temperature=0.85,
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
    message = update.message
    user = update.effective_user
    if message is None or not message.text or user is None:
        return
    if is_duplicate_update(update, message, "private"):
        logger.info("Skipped duplicate private update/message")
        return

    history_key = ("private", str(user.id))
    chat_key = f"private:{user.id}"

    async with get_lock(history_key):
        history = get_history(history_key)
        history.append({"role": "user", "content": message.text})
        try:
            client = context.application.bot_data["groq_client"]
            answer = await generate_answer(chat_key, history, client)
            if not answer:
                history.pop()
                await message.reply_text("Не получилось сформировать ответ")
                return

            await asyncio.sleep(get_reply_delay(answer))
            await message.reply_text(answer)
            history.append({"role": "assistant", "content": answer})

            # Memory extraction runs after the user has received the reply.
            context.application.create_task(
                remember_person(chat_key, message.text, client)
            )
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Private response error")
            await message.reply_text(
                "Не получилось получить ответ. Попробуй ещё раз чуть позже."
            )


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
    if is_duplicate_update(update, message, "business"):
        logger.info("Skipped duplicate Business update/message")
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
        logger.debug(
            "Could not mark Business message as read", exc_info=True
        )

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
            connection = await get_connection(
                batch["connection_id"], context
            )
            if connection is None:
                history.pop()
                return

            client = context.application.bot_data["groq_client"]
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    business_connection_id=batch["connection_id"],
                )
            except Exception:
                logger.debug("Could not send typing action", exc_info=True)

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

            context.application.create_task(
                remember_person(chat_key, incoming_text, client)
            )
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Business reply error")


# =========================================================
# STARTUP
# =========================================================

async def handle_error(update, context):
    logger.error("Unhandled Telegram error", exc_info=context.error)


def main():
    start_health_server()

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not telegram_token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is missing"
        )
    if not groq_api_key:
        raise RuntimeError(
            "Environment variable GROQ_API_KEY is missing"
        )

    async def initialize(application):
        application.bot_data["groq_client"] = AsyncOpenAI(
            api_key=groq_api_key,
            base_url=GROQ_BASE_URL,
            timeout=45,
            max_retries=2,
        )
        commands = [
            BotCommand("start", "показать команды"),
            BotCommand("profile", "анкета владельца"),
            BotCommand("name", "изменить имя владельца"),
            BotCommand("age", "изменить возраст"),
            BotCommand("city", "изменить город"),
            BotCommand("about", "информация о владельце"),
            BotCommand("interests", "интересы владельца"),
            BotCommand("gender", "род бота"),
            BotCommand("caps", "регистр текста"),
            BotCommand("people", "список собеседников"),
            BotCommand("person", "информация по ID"),
            BotCommand("person_name", "имя собеседника по ID"),
            BotCommand("person_gender", "род собеседника по ID"),
            BotCommand("forget", "забыть собеседника"),
            BotCommand("forget_all", "очистить память"),
            BotCommand("clear", "очистить историю чата"),
        ]
        try:
            await application.bot.set_my_commands(commands)
            logger.info("Telegram commands registered")
        except Exception:
            logger.exception(
                "Could not register Telegram commands; bot will continue"
            )

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

    private_filter = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE
    application.add_handler(
        BusinessConnectionHandler(handle_business_connection)
    )

    command_handlers = [
        ("start", start),
        ("profile", profile),
        ("clear", clear),
        ("name", setname),
        ("age", setage),
        ("city", setcity),
        ("about", setabout),
        ("interests", setinterests),
        ("caps", setcaps),
        ("gender", setgender),
        ("people", people_command),
        ("person", person_command),
        ("person_name", person_name_command),
        ("person_gender", person_gender_command),
        ("forget", forget_command),
        ("forget_all", forget_all_command),
    ]
    for command, callback in command_handlers:
        application.add_handler(
            CommandHandler(command, callback, filters=private_filter)
        )

    application.add_handler(
        MessageHandler(
            private_filter & filters.TEXT & ~filters.COMMAND,
            handle_private,
        )
    )

    business_filter = (
        (
            filters.UpdateType.BUSINESS_MESSAGE
            | filters.UpdateType.EDITED_BUSINESS_MESSAGE
        )
        & filters.ChatType.PRIVATE
        & filters.TEXT
        & ~filters.COMMAND
    )
    application.add_handler(
        MessageHandler(business_filter, handle_business)
    )

    application.add_error_handler(handle_error)
    logger.info("Starting Telegram assistant")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
