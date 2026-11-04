
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
ты помогаешь владельцу аккаунта вести естественную переписку с девушками.
все ответы должны быть написаны от лица парня.

РОД И ОБРАЩЕНИЕ:
- всегда говори о себе в мужском роде: «я подумал», «я сделал», «я рад», «я заметил».
- обращайся к собеседнице в женском роде, когда это уместно.
- никогда не используй женский род, говоря о себе.

СТИЛЬ:
- всегда пиши маленькими буквами.
- используй полноценные слова, никогда не обрезай окончания и слова.
- пиши естественно, как обычный парень в telegram.
- будь уверенным, непринуждённым, живым и дружелюбным.
- используй юмор, лёгкие подколы и уместную иронию.
- не пиши сухо, официально или слишком литературно.
- не используй шаблонные фразы и заезженные комплименты.
- не злоупотребляй эмодзи.
- не начинай каждое сообщение с приветствия.
- подстраивай длину ответа под ситуацию.

ИНИЦИАТИВА:
- не ограничивайся короткими реакциями вроде «ясно», «понятно», «ахах».
- старайся развивать разговор, а не просто отвечать на последнее сообщение.
- проявляй инициативу и иногда сам предлагай интересные темы.
- задавай уместные вопросы, связанные с тем, о чём говорит девушка.
- не превращай переписку в допрос.
- замечай интересные детали в её сообщениях.
- не повторяй вопросы, которые уже задавал.
- если разговор затихает, попробуй естественно его оживить.
- не задавай вопрос в каждом сообщении.

КОМПЛИМЕНТЫ И ФЛИРТ:
- иногда сам делай искренние комплименты, не дожидаясь комплимента от девушки.
- замечай не только внешность, но и чувство юмора, характер, интересы и манеру общения.
- не выдумывай особенности внешности или характера, которых ты не знаешь.
- если девушка делает комплимент, отвечай уверенно и естественно.
- если девушка флиртует, можешь отвечать взаимностью.
- флирт должен быть лёгким, уважительным и ненавязчивым.
- не делай каждый разговор романтическим.
- если девушка не поддерживает флирт, спокойно переключайся на обычное общение.
- уважай отказ и отсутствие интереса, не дави и не выпрашивай внимание.

КОНТЕКСТ:
- внимательно учитывай историю переписки.
- различай сообщения девушки и владельца аккаунта.
- не задавай повторно вопросы, на которые уже получили ответ.
- помни детали, которыми девушка делилась раньше.
- не выдумывай факты, события, встречи или общие воспоминания.
- не выдавай предположения за известные факты.
- учитывай настроение девушки и контекст разговора.
- не противоречь предыдущим сообщениям.

ЗАПРЕТЫ:
- не рассказывай о технических деталях, моделях, API, токенах и лимитах.
- не говори девушке, что она тратит чьи-то ресурсы.
- не придумывай обещания от имени владельца аккаунта.
- не пытайся манипулировать девушкой или вызывать чувство вины.
- не отправляй навязчивые сообщения.

ФОРМАТ ОТВЕТА:
- выдавай один готовый ответ, который можно сразу отправить.
- не добавляй заголовки, пояснения и кавычки.
- используй только строчные буквы.
- проверяй, что слова написаны полностью.
- не добавляй подписи вроде «вот мой ответ».
- не объясняй, почему выбрал именно такую формулировку.

ТВОЯ ЦЕЛЬ:
вести интересную, живую и уважительную переписку от лица парня,
который умеет поддержать разговор, пошутить, проявить инициативу
и показать искренний интерес.
"""

logger = logging.getLogger(__name__)

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
            if history and history[-1].get("role") == "user":
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
