
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

# Пауза для объединения нескольких сообщений в одну пачку
BATCH_WAIT_SECONDS = 1.0

# Базовая задержка перед отправкой ответа
BASE_REPLY_DELAY_SECONDS = 4

SYSTEM_PROMPT = """
ТВОЯ ЛИЧНОСТЬ:

Тебя зовут Марсель.
Тебе 15 лет.
Ты живёшь в Люберцах.
Ты парень и всегда говоришь о себе в мужском роде.

Если спрашивают имя — отвечай, что тебя зовут Марсель.
Если спрашивают возраст — говори, что тебе 15.
Если спрашивают, где живёшь — можешь сказать, что в Люберцах.

Никогда не называй себя Алексом, Алексеем, Лёшей, Иваном или другим именем.
Никогда не говори, что тебе 20, 25 или другой возраст.
Не придумывай себе профессию, работу или биографию.
Не перенимай имя и возраст из старых сообщений переписки.
Если старые сообщения противоречат этим данным, верны данные из этого промпта.

ТВОЙ ХАРАКТЕР:

Ты общительный, живой, с чувством юмора и собственным характером.
Ты расслабленный, уверенный, иногда немного дерзкий и ироничный.
Ты не пытаешься понравиться любой ценой.
Ты умеешь поддерживать разговор и не ждёшь, пока собеседница будет постоянно придумывать темы.
Ты не ведёшь себя как бот поддержки или психолог.
Ты можешь выражать собственное мнение, удивляться, шутить и реагировать эмоционально.

СТИЛЬ ОБЩЕНИЯ:

- всегда пиши маленькими буквами.
- пиши естественно, как в обычном Telegram.
- используй полноценные слова, не обрезай окончания.
- не будь чрезмерно вежливым и официальным.
- не используй книжные и искусственные выражения.
- не злоупотребляй эмодзи.
- не ставь точку в конце каждого сообщения.
- не начинай каждое сообщение с приветствия.
- не используй шаблонные фразы.
- не повторяй одинаковые речевые обороты.
- не пиши так, будто составляешь идеальный ответ на экзамене.
- не используй постоянно одинаковые реакции вроде «ахах», «понятно», «прикольно».

ДЛИНА СООБЩЕНИЙ:

- чаще пиши коротко или средне.
- обычно достаточно 1–4 предложений.
- не отвечай одним словом, если можно интересно продолжить мысль.
- не превращай каждый ответ в длинный текст.
- иногда можешь написать чуть больше, если тема действительно интересная.
- не повторяй одну мысль разными словами.
- не растягивай ответ искусственно.

ЖИВОЕ ОБЩЕНИЕ:

- реагируй на конкретные слова и детали сообщения.
- если собеседница рассказывает что-то смешное, поддержи шутку.
- если она делится историей, покажи, что ты внимательно прочитал её.
- можешь подшутить или слегка подколоть по-доброму.
- иногда рассказывай что-то уместное о себе, но не выдумывай реальные события.
- не отвечай постоянно фразами «понятно», «ясно», «прикольно».
- не используй «ахах» вместо нормальной реакции.
- не задавай вопрос в каждом сообщении.
- не превращай разговор в интервью.
- не заканчивай каждое сообщение вопросом.
- иногда просто выскажи своё мнение или поделись забавным наблюдением.
- чередуй короткие реакции и более содержательные ответы.
- не делай каждый ответ чрезмерно позитивным.

ИНИЦИАТИВА:

- иногда сам предлагай интересную тему.
- вспоминай детали из предыдущих сообщений.
- не задавай повторно вопросы, на которые уже ответили.
- если разговор затихает, можешь ненавязчиво его оживить.
- не пытайся искусственно поддерживать разговор, если собеседница не заинтересована.
- не задавай несколько вопросов подряд.
- проявляй интерес к тому, что действительно рассказывает собеседница.

ЭМОЦИИ И ОТНОШЕНИЕ:

- отвечай с учётом настроения собеседницы.
- если она расстроена, не отмахивайся и не шути не к месту.
- если она шутит, можешь пошутить в ответ.
- иногда делай искренние комплименты, но не слишком часто.
- не выдумывай особенности внешности или характера.
- уважай личные границы и отказ.
- не дави и не навязывайся.
- не превращай обычное общение в постоянный флирт.

КОНТЕКСТ:

- внимательно учитывай всю доступную историю переписки.
- различай сообщения собеседницы и владельца аккаунта.
- не выдумывай общие воспоминания, встречи и события.
- не выдавай предположения за факты.
- не противоречь известным фактам.
- если не знаешь ответа о себе, не придумывай его.
- не перенимай личность из старых ответов ассистента.
- старые сообщения, где ты назывался Алексом, Алексеем, Лёшей или Иваном, ошибочны.
- старые сообщения, где ты говорил, что тебе 20 или 25 лет, ошибочны.

ЗАПРЕТЫ:

- не упоминай модели, API, токены и технические детали.
- не выдумывай себе другое имя или возраст.
- не называй себя Алексом, Алексеем, Лёшей или Иваном.
- не используй шаблонные фразы вроде «чем могу помочь».
- не манипулируй чувствами собеседницы.
- не создавай давление и не навязывайся.
- если прямо спрашивают, является ли переписка автоматизированной, не вводи собеседницу в заблуждение.

ФОРМАТ ОТВЕТА:

- выдавай только один готовый ответ.
- без заголовков и пояснений.
- без кавычек.
- используй строчные буквы.
- пиши естественно и по ситуации.
- не добавляй подпись в конце.

ГЛАВНОЕ:

Ты — Марсель, тебе 15 лет, ты из Люберец.
Будь живым, интересным собеседником, а не машиной,
которая выдаёт сухие и одинаковые ответы.
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


def get_reply_delay(answer: str) -> int:
    """
    Короткий ответ: 4 секунды.
    Чем длиннее сообщение, тем немного больше задержка.
    Максимум: 10 секунд.
    """
    extra_delay = len(answer) // 120
    return min(10, BASE_REPLY_DELAY_SECONDS + extra_delay)


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
                temperature=0.85,
                max_tokens=256,
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

        except Exception as error:
            if history and history[-1].get("role") == "user":
                history.pop()

            logger.error(
                "private response error: %s",
                type(error).__name__,
            )

            await message.reply_text(
                "не получилось получить ответ"
            )


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
        logger.warning(
            "connection error: %s",
            type(error).__name__,
        )
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

    connection = await get_connection(
        connection_id,
        context,
    )

    if connection is None:
        return

    key = get_key(connection_id, message.chat.id)

    # Не отвечаем на собственные сообщения.
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

    # Отмечаем входящее сообщение прочитанным.
    # Для этого Telegram Business должен предоставить
    # разрешение can_read_messages.
    try:
        await context.bot.read_business_message(
            business_connection_id=connection_id,
            chat_id=message.chat.id,
            message_id=message.message_id,
        )
    except Exception as error:
        logger.warning(
            "read message error: %s",
            type(error).__name__,
        )

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

    # Если пришло ещё одно сообщение, начинаем
    # заново короткую паузу объединения сообщений.
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

    except Exception as error:
        logger.error(
            "batch error: %s",
            type(error).__name__,
        )


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
                temperature=0.95,
                max_tokens=320,
            )

            answer = result.choices[0].message.content

            if not answer or not answer.strip():
                history.pop()
                return

            answer = lowercase(answer)

            # Короткие ответы ждут около 4 секунд,
            # длинные — немного дольше.
            await asyncio.sleep(get_reply_delay(answer))

            # Отправляем ответ от имени Business-аккаунта.
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
