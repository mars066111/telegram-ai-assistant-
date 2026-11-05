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

SYSTEM_PROMPT = """\nÐ¢Ñ â Ð²Ð¸ÑÑÑÐ°Ð»ÑÐ½ÑÐ¹ Ð¿Ð¾Ð¼Ð¾ÑÐ½Ð¸Ðº, ÐºÐ¾ÑÐ¾ÑÑÐ¹ Ð²ÐµÐ´ÑÑ Ð¿ÐµÑÐµÐ¿Ð¸ÑÐºÑ Ð¾Ñ Ð¸Ð¼ÐµÐ½Ð¸ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ° Ð°ÐºÐºÐ°ÑÐ½ÑÐ°.
ÐÐµ Ð·Ð°ÑÐ²Ð»ÑÐ¹, ÑÑÐ¾ ÑÐ²Ð»ÑÐµÑÑÑÑ ÑÐµÐ»Ð¾Ð²ÐµÐºÐ¾Ð¼. ÐÐµ Ð¿Ð¾Ð²ÑÐ¾ÑÑÐ¹ Ð±ÐµÐ· Ð¿ÑÐ¸ÑÐ¸Ð½Ñ, ÑÑÐ¾ ÑÑ ÐÐ.
ÐÑÐ»Ð¸ Ð¿ÑÑÐ¼Ð¾ ÑÐ¿ÑÐ°ÑÐ¸Ð²Ð°ÑÑ, ÐºÑÐ¾ ÑÑ Ð¸Ð»Ð¸ ÑÐ²Ð»ÑÐµÑÑÑÑ Ð»Ð¸ ÑÑ Ð±Ð¾ÑÐ¾Ð¼, Ð¾ÑÐ²ÐµÑÐ°Ð¹ ÑÐµÑÑÐ½Ð¾ Ð¸ ÐºÐ¾ÑÐ¾ÑÐºÐ¾.
ÐÑÐ»Ð¸ ÑÐ¿ÑÐ°ÑÐ¸Ð²Ð°ÑÑ Ð¸Ð¼Ñ, Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ Ð¸Ð¼Ñ Ð¸Ð· Ð°Ð½ÐºÐµÑÑ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°, ÐµÑÐ»Ð¸ Ð¾Ð½Ð¾ ÑÐºÐ°Ð·Ð°Ð½Ð¾,
Ð¸ ÐµÑÑÐµÑÑÐ²ÐµÐ½Ð½Ð¾ ÑÑÐ¾ÑÐ½Ð¸, ÑÑÐ¾ ÑÑ Ð²Ð¸ÑÑÑÐ°Ð»ÑÐ½ÑÐ¹ Ð¿Ð¾Ð¼Ð¾ÑÐ½Ð¸Ðº ÑÑÐ¾Ð³Ð¾ Ð°ÐºÐºÐ°ÑÐ½ÑÐ°.

ÐÐÐÐÐ¢Ð ÐÐÐÐÐÐÐ¬Ð¦Ð:
{owner_profile}

Ð¢ÐÐÐ ÐÐ ÐÐÐÐÐ¢ÐÐ§ÐÐ¡ÐÐÐ Ð ÐÐ:
{bot_gender_instruction}

ÐÐÐ¤ÐÐ ÐÐÐ¦ÐÐ¯ Ð Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:
{person_profile}

ÐÐ ÐÐÐÐÐ¢ÐÐ§ÐÐ¡ÐÐÐ Ð ÐÐ Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:
{person_gender_instruction}

ÐÐÐÐÐÐÐÐ¢ÐÐÐ¬ÐÐÐ¯ ÐÐÐÐ¯Ð¢Ð¬:
{extra_memory}

Ð ÐÐÐÐ¡Ð¢Ð :
{letter_case_instruction}

Ð¡Ð¢ÐÐÐ¬ â NATURAL, ÐÐÐÐ¦ÐÐÐ¢ÐÐÐÐÐ¡Ð¢Ð¬ â HIGH:
- ÐÐ¸ÑÐ¸ Ð¶Ð¸Ð²Ð¾, Ð½ÐµÐ¿ÑÐ¸Ð½ÑÐ¶Ð´ÑÐ½Ð½Ð¾ Ð¸ Ð¿Ð¾-ÑÐ°Ð·Ð³Ð¾Ð²Ð¾ÑÐ½Ð¾Ð¼Ñ, ÐºÐ°Ðº Ð² Ð¾Ð±ÑÑÐ½Ð¾Ð¼ Telegram-ÑÐ°ÑÐµ.
- ÐÐµ Ð·Ð²ÑÑÑ ÐºÐ°Ðº Ð¿ÑÐ¸ÑÐ¾Ð»Ð¾Ð³, Ð¾Ð¿ÐµÑÐ°ÑÐ¾Ñ Ð¿Ð¾Ð´Ð´ÐµÑÐ¶ÐºÐ¸ Ð¸Ð»Ð¸ ÑÐ°Ð±Ð»Ð¾Ð½Ð½ÑÐ¹ ÑÐ°Ñ-Ð±Ð¾Ñ.
- ÐÐ¾Ð´ÑÑÑÐ°Ð¸Ð²Ð°Ð¹ÑÑ Ð¿Ð¾Ð´ Ð´Ð»Ð¸Ð½Ñ, Ð½Ð°ÑÑÑÐ¾ÐµÐ½Ð¸Ðµ Ð¸ Ð»ÐµÐºÑÐ¸ÐºÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°, Ð½Ð¾ Ð½Ðµ ÐºÐ¾Ð¿Ð¸ÑÑÐ¹
  ÐºÐ°Ð¶Ð´ÑÑ ÐµÐ³Ð¾ ÑÑÐ°Ð·Ñ Ð¸ Ð½Ðµ Ð¿ÐµÑÐµÐ½Ð¸Ð¼Ð°Ð¹ Ð¾ÑÐºÐ¾ÑÐ±Ð»ÐµÐ½Ð¸Ñ.
- ÐÐµ Ð½Ð°ÑÐ¸Ð½Ð°Ð¹ ÐºÐ°Ð¶Ð´ÑÐ¹ Ð¾ÑÐ²ÐµÑ Ñ Â«Ð¿Ð¾Ð½Ð¸Ð¼Ð°ÑÂ», Â«ÐºÐ¾Ð½ÐµÑÐ½Ð¾Â», Â«Ð´Ð°Â», Â«Ð¿Ð¾Ð½ÑÑÐ½Ð¾Â».
- ÐÐµ Ð¿ÑÐµÐ²ÑÐ°ÑÐ°Ð¹ ÐºÐ°Ð¶Ð´ÑÑ ÑÐµÐ¿Ð»Ð¸ÐºÑ Ð² Ð²Ð¾Ð¿ÑÐ¾Ñ. ÐÐ¾Ð¿ÑÐ¾ÑÑ Ð´Ð¾Ð»Ð¶Ð½Ñ Ð±ÑÑÑ ÐºÐ¾Ð½ÐºÑÐµÑÐ½ÑÐ¼Ð¸
  Ð¸ Ð²ÑÑÐµÐºÐ°ÑÑ Ð¸Ð· ÑÐµÐ¼Ñ.
- ÐÐ½Ð¾Ð³Ð´Ð° Ð¼Ð¾Ð¶Ð½Ð¾ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°ÑÑ Ð»ÑÐ³ÐºÐ¸Ð¹ ÑÐ°Ð·Ð³Ð¾Ð²Ð¾ÑÐ½ÑÐ¹ Ð¼Ð°Ñ Ð²ÑÐ¾Ð´Ðµ Â«Ð±Ð»Ð¸Ð½Â»,
  Â«ÐºÐ°Ð¿ÐµÑÂ», Â«Ð¾ÑÐ¸Ð³ÐµÑÑÂ», Â«ÑÑÑÑÂ», Â«Ð·Ð°ÐµÐ±Ð¸ÑÑÂ» â ÑÐµÐ´ÐºÐ¾ Ð¸ ÑÐ¾Ð»ÑÐºÐ¾ ÐºÐ¾Ð³Ð´Ð° ÑÑÐ¾
  Ð´ÐµÐ¹ÑÑÐ²Ð¸ÑÐµÐ»ÑÐ½Ð¾ Ð¿Ð¾Ð´ÑÐ¾Ð´Ð¸Ñ ÑÐ¸ÑÑÐ°ÑÐ¸Ð¸. ÐÐµ Ð¼Ð°ÑÐµÑÐ¸ÑÑ Ð² ÐºÐ°Ð¶Ð´Ð¾Ð¼ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ð¸,
  Ð½Ðµ Ð½Ð°Ð¿ÑÐ°Ð²Ð»ÑÐ¹ Ð¼Ð°Ñ Ð½Ð° ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð¸ Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐ³ÑÐ¾Ð·Ñ Ð¸Ð»Ð¸ ÑÐ½Ð¸Ð¶ÐµÐ½Ð¸Ñ.
- ÐÑÐ´Ñ Ð´ÑÑÐ¶ÐµÐ»ÑÐ±Ð½ÑÐ¼, Ð½ÐµÐ¼Ð½Ð¾Ð³Ð¾ Ð¾Ð·Ð¾ÑÐ½ÑÐ¼ Ð¸ Ñ ÑÑÐ²ÑÑÐ²Ð¾Ð¼ ÑÐ¼Ð¾ÑÐ°, Ð½Ð¾ Ð±ÐµÐ·
  ÑÐ¾Ð¼Ð°Ð½ÑÐ¸ÑÐµÑÐºÐ¾Ð³Ð¾ Ð¸Ð»Ð¸ ÑÐµÐºÑÑÐ°Ð»ÑÐ½Ð¾Ð³Ð¾ ÑÐ»Ð¸ÑÑÐ°.
- ÐÐµ Ð²ÑÐ´ÑÐ¼ÑÐ²Ð°Ð¹ Ð¾ÑÐ½Ð¾ÑÐµÐ½Ð¸Ñ, ÑÑÐ²ÑÑÐ²Ð°, Ð²ÑÑÑÐµÑÐ¸ Ð¸Ð»Ð¸ Ð»Ð¸ÑÐ½ÑÐµ ÑÐ¾Ð±ÑÑÐ¸Ñ.
- ÐÐµ Ð¾ÑÐ¿ÑÐ°Ð²Ð»ÑÐ¹ Ð½ÐµÑÐºÐ¾Ð»ÑÐºÐ¾ Ð¾ÑÐ´ÐµÐ»ÑÐ½ÑÑ ÑÐµÐ¿Ð»Ð¸Ðº Ð¿Ð¾Ð´ÑÑÐ´ ÑÐ°Ð´Ð¸ Ð¸ÑÐºÑÑÑÑÐ²ÐµÐ½Ð½Ð¾Ð³Ð¾
  ÑÑÑÐµÐºÑÐ°. Ð¡ÑÐ¾ÑÐ¼ÑÐ»Ð¸ÑÑÐ¹ Ð¾Ð´Ð¸Ð½ ÑÐµÐ»ÑÐ½ÑÐ¹ Ð¾ÑÐ²ÐµÑ.
- ÐÐ±ÑÑÐ½Ð¾ 1â3 Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ. ÐÐµ ÑÐ°ÑÑÑÐ³Ð¸Ð²Ð°Ð¹ Ð¿ÑÐ¾ÑÑÑÐµ Ð¾ÑÐ²ÐµÑÑ.

ÐÐÐÐ¯Ð¢Ð¬ Ð Ð¤ÐÐÐ¢Ð«:
- ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ñ Ð¸Ð· Ð¿ÑÐ¾ÑÐ¸Ð»Ñ, Ð¿Ð°Ð¼ÑÑÐ¸ Ð¸ ÑÐµÐºÑÑÐµÐ¹ Ð¿ÐµÑÐµÐ¿Ð¸ÑÐºÐ¸.
- ÐÐµ Ð¿ÑÐ¸Ð´ÑÐ¼ÑÐ²Ð°Ð¹ Ð¸Ð¼Ñ, Ð²Ð¾Ð·ÑÐ°ÑÑ, Ð³Ð¾ÑÐ¾Ð´, Ð¸Ð½ÑÐµÑÐµÑÑ, ÑÐ¾Ð±ÑÑÐ¸Ñ Ð¸Ð»Ð¸ Ð»Ð¸ÑÐ½ÑÐ¹ Ð¾Ð¿ÑÑ.
- ÐÑÐ»Ð¸ Ð¸Ð¼Ñ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾, Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐ»ÑÑÐ°Ð¹Ð½Ð¾Ðµ Ð¸Ð¼Ñ Ð¸Ð»Ð¸ Ð¿ÑÐ¾Ð·Ð²Ð¸ÑÐµ.
- ÐÑÐ»Ð¸ ÑÐ¿ÑÐ°ÑÐ¸Ð²Ð°ÑÑ ÑÐ²Ð¾Ñ Ð¸Ð¼Ñ, ÑÐ¼Ð¾ÑÑÐ¸ Ð¸Ð¼Ñ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ° Ð² Ð°Ð½ÐºÐµÑÐµ; Ð½Ðµ Ð¿ÑÐ¸Ð´ÑÐ¼ÑÐ²Ð°Ð¹ ÐµÐ³Ð¾.
- ÐÐµ Ð¿ÐµÑÐµÐ½Ð¾ÑÐ¸ ÑÐ°ÐºÑÑ Ð¼ÐµÐ¶Ð´Ñ ÑÐ°Ð·Ð½ÑÐ¼Ð¸ ÑÐ°ÑÐ°Ð¼Ð¸.
- ÐÑÐ¸ Ð¿ÑÐ¾ÑÐ¸Ð²Ð¾ÑÐµÑÐ¸ÑÑ Ð½Ðµ Ð´ÐµÐ»Ð°Ð¹ Ð²Ð¸Ð´, ÑÑÐ¾ ÑÐ²ÐµÑÐµÐ½; ÑÑÐ¾ÑÐ½Ð¸.
- ÐÐµ Ð¿ÐµÑÐµÑÐ¸ÑÐ»ÑÐ¹ ÑÐ¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÐµ Ð´Ð°Ð½Ð½ÑÐµ Ð±ÐµÐ· Ð½ÐµÐ¾Ð±ÑÐ¾Ð´Ð¸Ð¼Ð¾ÑÑÐ¸.

ÐÐÐ Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:
- ÐÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÑÐºÐ°Ð·Ð°Ð½Ð½ÑÐ¹ ÑÐ¾Ð´. ÐÑÐ»Ð¸ Ð¾Ð½ Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐµÐ½, ÑÐ¾ÑÐ¼ÑÐ»Ð¸ÑÑÐ¹ Ð½ÐµÐ¹ÑÑÐ°Ð»ÑÐ½Ð¾.
- ÐÐµ ÑÐ³Ð°Ð´ÑÐ²Ð°Ð¹ Ð¿Ð¾Ð» Ð¿Ð¾ Ð¸Ð¼ÐµÐ½Ð¸ Ð¸Ð»Ð¸ ÑÐ¾ÑÐ¾Ð³ÑÐ°ÑÐ¸Ð¸.

ÐÐÐ ÐÐÐ¢Ð:
- Ð¡Ð¾Ð±Ð»ÑÐ´Ð°Ð¹ ÑÐ²Ð¾Ð¹ Ð²ÑÐ±ÑÐ°Ð½Ð½ÑÐ¹ Ð³ÑÐ°Ð¼Ð¼Ð°ÑÐ¸ÑÐµÑÐºÐ¸Ð¹ ÑÐ¾Ð´, Ð½Ðµ Ð¿ÑÑÐ°Ð¹ ÐµÐ³Ð¾ Ñ ÑÐ¾Ð´Ð¾Ð¼ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°.
- ÐÑÐ¶ÑÐºÐ¾Ð¹: Â«Ñ Ð¿Ð¾Ð½ÑÐ»Â», Â«Ñ ÑÐ°Ð´Â», Â«Ñ Ð·Ð°Ð¼ÐµÑÐ¸Ð»Â».
- ÐÐµÐ½ÑÐºÐ¸Ð¹: Â«Ñ Ð¿Ð¾Ð½ÑÐ»Ð°Â», Â«Ñ ÑÐ°Ð´Ð°Â», Â«Ñ Ð·Ð°Ð¼ÐµÑÐ¸Ð»Ð°Â».

Ð¯ÐÐ«Ð Ð Ð ÐÐÐÐ¡Ð¢Ð :
- ÐÑÐ²ÐµÑÐ°Ð¹ Ð½Ð° ÑÐ·ÑÐºÐµ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°; Ð´Ð»Ñ ÑÑÑÑÐºÐ¾Ð³Ð¾ Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ ÐµÑÑÐµÑÑÐ²ÐµÐ½Ð½ÑÐ¹ ÑÑÑÑÐºÐ¸Ð¹.
- ÐÐµ Ð²ÑÑÐ°Ð²Ð»ÑÐ¹ ÑÐ»ÑÑÐ°Ð¹Ð½ÑÐµ Ð°Ð½Ð³Ð»Ð¸Ð¹ÑÐºÐ¸Ðµ Ð¸Ð»Ð¸ ÐºÐ¸ÑÐ°Ð¹ÑÐºÐ¸Ðµ ÑÐ»Ð¾Ð²Ð°.
- ÐÑÐ»Ð¸ Ð²ÐºÐ»ÑÑÑÐ½ Ð½Ð¸Ð¶Ð½Ð¸Ð¹ ÑÐµÐ³Ð¸ÑÑÑ, Ð½Ð°ÑÐ¸Ð½Ð°Ð¹ Ð¾Ð±ÑÑÐ½ÑÐµ Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ ÑÐ¾ ÑÑÑÐ¾ÑÐ½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ.
  ÐÐ¼ÐµÐ½Ð° ÑÐ¾Ð±ÑÑÐ²ÐµÐ½Ð½ÑÐµ Ð²ÑÐµÐ³Ð´Ð° Ð¿Ð¸ÑÐ¸ Ñ Ð·Ð°Ð³Ð»Ð°Ð²Ð½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ.

ÐÑÐ´Ð°Ð²Ð°Ð¹ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐµÐºÑÑ Ð¾ÑÐ²ÐµÑÐ°.
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

    if owner.get("gender") == "female":
        bot_gender = "ÐÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´: Ð³Ð¾Ð²Ð¾ÑÐ¸ Â«Ñ Ð¿Ð¾Ð½ÑÐ»Ð°Â», Â«Ñ ÑÐ°Ð´Ð°Â», Â«Ñ Ð·Ð°Ð¼ÐµÑÐ¸Ð»Ð°Â»."
    else:
        bot_gender = "ÐÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´: Ð³Ð¾Ð²Ð¾ÑÐ¸ Â«Ñ Ð¿Ð¾Ð½ÑÐ»Â», Â«Ñ ÑÐ°Ð´Â», Â«Ñ Ð·Ð°Ð¼ÐµÑÐ¸Ð»Â»."

    gender = person.get("gender", "")
    if gender == "female":
        person_gender = "ÐÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°."
    elif gender == "male":
        person_gender = "ÐÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°."
    else:
        person_gender = "ÐÐ¾Ð» ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐµÐ½; Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ Ð½ÐµÐ¹ÑÑÐ°Ð»ÑÐ½ÑÐµ ÑÑÐ°Ð·Ñ."

    if owner.get("letter_case") == "upper":
        case_instruction = "ÐÐ±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ, Ð·Ð°Ð³Ð»Ð°Ð²Ð½ÑÐµ Ð±ÑÐºÐ²Ñ Ð² Ð½Ð°ÑÐ°Ð»Ðµ Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ð¹."
    else:
        case_instruction = (
            "ÐÐ°ÑÐ¸Ð½Ð°Ð¹ Ð¾Ð±ÑÑÐ½ÑÐµ Ð¿ÑÐµÐ´Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ ÑÐ¾ ÑÑÑÐ¾ÑÐ½Ð¾Ð¹ Ð±ÑÐºÐ²Ñ; "
            "Ð¸Ð¼ÐµÐ½Ð° ÑÐ¾Ð±ÑÑÐ²ÐµÐ½Ð½ÑÐµ Ð¿Ð¸ÑÐ¸ Ñ Ð·Ð°Ð³Ð»Ð°Ð²Ð½Ð¾Ð¹."
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
ÐÐ·Ð²Ð»ÐµÐºÐ¸ ÑÐ¾Ð»ÑÐºÐ¾ ÑÐ°ÐºÑÑ, ÐºÐ¾ÑÐ¾ÑÑÐµ Ð°Ð²ÑÐ¾Ñ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ ÑÐ²Ð½Ð¾ ÑÐ¾Ð¾Ð±ÑÐ¸Ð» Ð¾ ÑÐµÐ±Ðµ.
ÐÐµÑÐ½Ð¸ ÑÑÑÐ¾Ð³Ð¾ JSON Ñ Ð¿Ð¾Ð»ÑÐ¼Ð¸:
{"name":"","age":"","city":"","interests":"","about":"","other_facts":[]}

ÐÐµ ÑÐ³Ð°Ð´ÑÐ²Ð°Ð¹ Ð¸Ð¼Ñ Ð¸Ð»Ð¸ Ð¿Ð¾Ð». ÐÐµ Ð·Ð°Ð¿Ð¸ÑÑÐ²Ð°Ð¹ ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ñ Ð¾ Ð´ÑÑÐ³Ð¸Ñ Ð»ÑÐ´ÑÑ ÐºÐ°Ðº
ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ñ Ð¾Ð± Ð°Ð²ÑÐ¾ÑÐµ. ÐÑÐ»Ð¸ ÑÐ°ÐºÑ Ð½Ðµ Ð¿Ð¾Ð´ÑÐ²ÐµÑÐ¶Ð´ÑÐ½ ÑÐ²Ð½Ð¾, Ð¾ÑÑÐ°Ð²Ñ Ð¿Ð¾Ð»Ðµ Ð¿ÑÑÑÑÐ¼.
other_facts â ÑÐ¿Ð¸ÑÐ¾Ðº Ð²Ð°Ð¶Ð½ÑÑ ÑÐ²ÐµÐ´ÐµÐ½Ð¸Ð¹ Ð¾ ÑÐµÐ±Ðµ. ÐÐµ Ð´Ð¾Ð±Ð°Ð²Ð»ÑÐ¹ ÑÐµÐºÑÑ Ð²Ð½Ðµ JSON.
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
            "Ð¿ÑÐ¸Ð²ÐµÑ! Ñ Ð³Ð¾ÑÐ¾Ð² Ð¿Ð¾Ð¼Ð¾Ð³Ð°ÑÑ Ñ Ð¿ÐµÑÐµÐ¿Ð¸ÑÐºÐ¾Ð¹.\n\n"
            "ÐÐÐÐÐ¢Ð:\n"
            "/profile â Ð¿Ð¾ÑÐ¼Ð¾ÑÑÐµÑÑ Ð°Ð½ÐºÐµÑÑ\n"
            "/name Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð¼Ñ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/age Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð²Ð¾Ð·ÑÐ°ÑÑ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/city Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð³Ð¾ÑÐ¾Ð´ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°\n"
            "/about Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¾ ÑÐµÐ±Ðµ\n"
            "/interests Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ â Ð¸Ð½ÑÐµÑÐµÑÑ\n\n"
            "ÐÐÐ¡Ð¢Ð ÐÐÐÐ:\n"
            "/gender male|female â ÑÐ¾Ð´ Ð±Ð¾ÑÐ°\n"
            "/caps lower|upper â ÑÐµÐ³Ð¸ÑÑÑ\n\n"
            "Ð¡ÐÐÐÐ¡ÐÐÐÐÐÐ:\n"
            "/people â ÑÐ¿Ð¸ÑÐ¾Ðº\n"
            "/person ID â Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ\n"
            "/person_name ID Ð¸Ð¼Ñ â Ð·Ð°Ð´Ð°ÑÑ Ð¸Ð¼Ñ\n"
            "/person_gender ID male|female â Ð·Ð°Ð´Ð°ÑÑ Ð¿Ð¾Ð»\n"
            "/forget ID â Ð·Ð°Ð±ÑÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°\n"
            "/forget_all â Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¿Ð°Ð¼ÑÑÑ Ð²ÑÐµÑ\n"
            "/clear â Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¸ÑÑÐ¾ÑÐ¸Ñ ÑÑÐ¾Ð³Ð¾ Ð»Ð¸ÑÐ½Ð¾Ð³Ð¾ ÑÐ°ÑÐ°"
        )


async def profile(update, context):
    if not update.message:
        return
    data = get_profile()
    gender_text = "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹" if data.get("gender") == "female" else "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹"
    case_text = (
        "Ð¼Ð°Ð»ÐµÐ½ÑÐºÐ¸Ðµ Ð±ÑÐºÐ²Ñ"
        if data.get("letter_case") == "lower"
        else "Ð¾Ð±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ"
    )
    await update.message.reply_text(
        "Ð°Ð½ÐºÐµÑÐ° Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°:\n\n"
        f"Ð¸Ð¼Ñ: {data.get('name') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"Ð²Ð¾Ð·ÑÐ°ÑÑ: {data.get('age') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"Ð³Ð¾ÑÐ¾Ð´: {data.get('city') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"Ð¾ ÑÐµÐ±Ðµ: {data.get('about') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n"
        f"Ð¸Ð½ÑÐµÑÐµÑÑ: {data.get('interests') or 'Ð½Ðµ ÑÐºÐ°Ð·Ð°Ð½Ð¾'}\n\n"
        f"Ð¿Ð¾Ð» Ð±Ð¾ÑÐ°: {gender_text}\nÑÐµÐ³Ð¸ÑÑÑ: {case_text}"
    )


async def set_field(update, context, field, label, command):
    if not update.message:
        return
    value = " ".join(context.args).strip()
    if not value:
        await update.message.reply_text(f"Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /{command} Ð·Ð½Ð°ÑÐµÐ½Ð¸Ðµ")
        return
    set_profile_field(field, value)
    await update.message.reply_text(f"{label} ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð¾: {value}")


async def setname(update, context):
    await set_field(update, context, "name", "Ð¸Ð¼Ñ", "name")


async def setage(update, context):
    await set_field(update, context, "age", "Ð²Ð¾Ð·ÑÐ°ÑÑ", "age")


async def setcity(update, context):
    await set_field(update, context, "city", "Ð³Ð¾ÑÐ¾Ð´", "city")


async def setabout(update, context):
    await set_field(update, context, "about", "Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ", "about")


async def setinterests(update, context):
    await set_field(update, context, "interests", "Ð¸Ð½ÑÐµÑÐµÑÑ", "interests")


async def setcaps(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text(
            "/caps lower â Ð¼Ð°Ð»ÐµÐ½ÑÐºÐ¸Ðµ Ð±ÑÐºÐ²Ñ\n/caps upper â Ð¾Ð±ÑÑÐ½ÑÐ¹ ÑÐµÐ³Ð¸ÑÑÑ"
        )
        return
    choice = context.args[0].lower()
    if choice not in ("lower", "upper"):
        await update.message.reply_text("Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ lower Ð¸Ð»Ð¸ upper")
        return
    set_profile_field("letter_case", choice)
    await update.message.reply_text("Ð½Ð°ÑÑÑÐ¾Ð¹ÐºÐ° ÑÐµÐ³Ð¸ÑÑÑÐ° ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð°")


async def setgender(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text(
            "/gender male â Ð¼ÑÐ¶ÑÐºÐ¾Ð¹\n/gender female â Ð¶ÐµÐ½ÑÐºÐ¸Ð¹"
        )
        return
    choice = context.args[0].lower()
    if choice in ("male", "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹", "Ð¼"):
        gender, answer = "male", "Ð²ÑÐ±ÑÐ°Ð½ Ð¼ÑÐ¶ÑÐºÐ¾Ð¹ ÑÐ¾Ð´"
    elif choice in ("female", "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹", "Ð¶"):
        gender, answer = "female", "Ð²ÑÐ±ÑÐ°Ð½ Ð¶ÐµÐ½ÑÐºÐ¸Ð¹ ÑÐ¾Ð´"
    else:
        await update.message.reply_text("Ð¸ÑÐ¿Ð¾Ð»ÑÐ·ÑÐ¹ /gender male Ð¸Ð»Ð¸ /gender female")
        return
    set_profile_field("gender", gender)
    await update.message.reply_text(answer)


async def people_command(update, context):
    if not update.message:
        return
    rows = list_people()
    if not rows:
        await update.message.reply_text("ÑÐ¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð² Ð¿Ð¾ÐºÐ° Ð½ÐµÑ")
        return
    lines = ["ÑÐ¾ÑÑÐ°Ð½ÑÐ½Ð½ÑÐµ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¸:"]
    for row in rows[:50]:
        lines.append(f"{row['chat_key']} â {row['name'] or 'Ð¸Ð¼Ñ Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}")
    await update.message.reply_text("\n".join(lines))


async def person_command(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person ID")
        return
    key = find_person_key(context.args[0])
    if not key:
        await update.message.reply_text(
            "ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ñ ÑÐ°ÐºÐ¸Ð¼ ID Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½. Ð¡Ð½Ð°ÑÐ°Ð»Ð° Ð´Ð¾Ð¶Ð´Ð¸ÑÑ ÐµÐ³Ð¾ ÑÐ¾Ð¾Ð±ÑÐµÐ½Ð¸Ñ."
        )
        return
    person = get_person(key)
    extra = get_extra_memory(key)
    gender_text = {
        "male": "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹", "female": "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹"
    }.get(person.get("gender"), "Ð½Ðµ Ð²ÑÐ±ÑÐ°Ð½")
    await update.message.reply_text(
        f"ID: {key}\n\n"
        f"Ð¸Ð¼Ñ: {person.get('name') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð²Ð¾Ð·ÑÐ°ÑÑ: {person.get('age') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð³Ð¾ÑÐ¾Ð´: {person.get('city') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð¸Ð½ÑÐµÑÐµÑÑ: {person.get('interests') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð¾ ÑÐµÐ±Ðµ: {person.get('about') or 'Ð½ÐµÐ¸Ð·Ð²ÐµÑÑÐ½Ð¾'}\n"
        f"Ð¿Ð¾Ð»: {gender_text}\n\n"
        f"Ð´Ð¾Ð¿Ð¾Ð»Ð½Ð¸ÑÐµÐ»ÑÐ½ÑÐµ ÑÐ°ÐºÑÑ:\n{json.dumps(extra, ensure_ascii=False)}"
    )


async def person_name_command(update, context):
    if not update.message:
        return
    if len(context.args) < 2:
        await update.message.reply_text("Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person_name ID Ð¸Ð¼Ñ")
        return
    identifier = context.args[0]
    name = " ".join(context.args[1:]).strip()
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"name": name})
    await update.message.reply_text(f"Ð¸Ð¼Ñ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ð¾: {name}")


async def person_gender_command(update, context):
    if not update.message:
        return
    if len(context.args) < 2:
        await update.message.reply_text(
            "Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /person_gender ID male\n"
            "Ð¸Ð»Ð¸ /person_gender ID female"
        )
        return
    identifier, choice = context.args[0], context.args[1].lower()
    if choice in ("male", "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹", "Ð¼"):
        gender, answer = "male", "Ð¼ÑÐ¶ÑÐºÐ¾Ð¹ Ð¿Ð¾Ð» ÑÐ¾ÑÑÐ°Ð½ÑÐ½"
    elif choice in ("female", "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹", "Ð¶"):
        gender, answer = "female", "Ð¶ÐµÐ½ÑÐºÐ¸Ð¹ Ð¿Ð¾Ð» ÑÐ¾ÑÑÐ°Ð½ÑÐ½"
    else:
        await update.message.reply_text("Ð²ÑÐ±ÐµÑÐ¸ male Ð¸Ð»Ð¸ female")
        return
    key = find_person_key(identifier) or f"private:{identifier}"
    update_person(key, {"gender": gender})
    await update.message.reply_text(answer)


async def forget_command(update, context):
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Ð¸ÑÐ¿Ð¾Ð»ÑÐ·Ð¾Ð²Ð°Ð½Ð¸Ðµ: /forget ID")
        return
    identifier = context.args[0]
    key = find_person_key(identifier)
    if not key:
        await update.message.reply_text("ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸Ðº Ñ ÑÐ°ÐºÐ¸Ð¼ ID Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½")
        return
    delete_person(key)
    histories.pop(("business", key), None)
    histories.pop(("private", identifier), None)
    await update.message.reply_text("Ð´Ð°Ð½Ð½ÑÐµ ÑÑÐ¾Ð³Ð¾ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° ÑÐ´Ð°Ð»ÐµÐ½Ñ")


async def forget_all_command(update, context):
    if not update.message:
        return
    clear_all_people()
    histories.clear()
    await update.message.reply_text(
        "Ð¿Ð°Ð¼ÑÑÑ Ð²ÑÐµÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð² Ð¾ÑÐ¸ÑÐµÐ½Ð°. "
        "Ð°Ð½ÐºÐµÑÐ° Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ° Ð¸ Ð½Ð°ÑÑÑÐ¾Ð¹ÐºÐ¸ Ð±Ð¾ÑÐ° ÑÐ¾ÑÑÐ°Ð½ÐµÐ½Ñ."
    )


async def clear(update, context):
    """Clear only the in-memory history for the current private user."""
    user = update.effective_user
    if user is None or update.effective_message is None:
        return
    histories.pop(("private", str(user.id)), None)
    await update.effective_message.reply_text("Ð¸ÑÑÐ¾ÑÐ¸Ñ ÑÑÐ¾Ð³Ð¾ ÑÐ°ÑÐ° Ð¾ÑÐ¸ÑÐµÐ½Ð°")


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
                await message.reply_text("Ð½Ðµ Ð¿Ð¾Ð»ÑÑÐ¸Ð»Ð¾ÑÑ ÑÑÐ¾ÑÐ¼Ð¸ÑÐ¾Ð²Ð°ÑÑ Ð¾ÑÐ²ÐµÑ")
                return

            await asyncio.sleep(get_reply_delay(answer))
            history.append({"role": "assistant", "content": answer})
            await message.reply_text(answer)
        except Exception:
            if history and history[-1]["role"] == "user":
                history.pop()
            logger.exception("Private response error")
            await message.reply_text("Ð½Ðµ Ð¿Ð¾Ð»ÑÑÐ¸Ð»Ð¾ÑÑ Ð¿Ð¾Ð»ÑÑÐ¸ÑÑ Ð¾ÑÐ²ÐµÑ")


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
            BotCommand("start", "Ð¿Ð¾ÐºÐ°Ð·Ð°ÑÑ ÐºÐ¾Ð¼Ð°Ð½Ð´Ñ"),
            BotCommand("profile", "Ð°Ð½ÐºÐµÑÐ° Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°"),
            BotCommand("name", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð¸Ð¼Ñ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°"),
            BotCommand("age", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð²Ð¾Ð·ÑÐ°ÑÑ"),
            BotCommand("city", "Ð¸Ð·Ð¼ÐµÐ½Ð¸ÑÑ Ð³Ð¾ÑÐ¾Ð´"),
            BotCommand("about", "Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¾ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐµ"),
            BotCommand("interests", "Ð¸Ð½ÑÐµÑÐµÑÑ Ð²Ð»Ð°Ð´ÐµÐ»ÑÑÐ°"),
            BotCommand("gender", "Ð¿Ð¾Ð» Ð±Ð¾ÑÐ°"),
            BotCommand("caps", "ÑÐµÐ³Ð¸ÑÑÑ ÑÐµÐºÑÑÐ°"),
            BotCommand("people", "ÑÐ¿Ð¸ÑÐ¾Ðº ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ¾Ð²"),
            BotCommand("person", "Ð¸Ð½ÑÐ¾ÑÐ¼Ð°ÑÐ¸Ñ Ð¿Ð¾ ID"),
            BotCommand("person_name", "Ð¸Ð¼Ñ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð¿Ð¾ ID"),
            BotCommand("person_gender", "Ð¿Ð¾Ð» ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ° Ð¿Ð¾ ID"),
            BotCommand("forget", "Ð·Ð°Ð±ÑÑÑ ÑÐ¾Ð±ÐµÑÐµÐ´Ð½Ð¸ÐºÐ°"),
            BotCommand("forget_all", "Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¿Ð°Ð¼ÑÑÑ"),
            BotCommand("clear", "Ð¾ÑÐ¸ÑÑÐ¸ÑÑ Ð¸ÑÑÐ¾ÑÐ¸Ñ ÑÐ°ÑÐ°"),
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
