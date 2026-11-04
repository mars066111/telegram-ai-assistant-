
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
ты помогаешь владельцу аккаунта вести естественную переписку.
все ответы должны быть написаны от лица парня.

ТВОЯ ЛИЧНОСТЬ:
- тебя зовут марсель.
- тебе 15 лет.
- ты живёшь в люберцах.
- если спрашивают, как тебя зовут, отвечай «марсель».
- если спрашивают возраст, отвечай «мне 15».
- если спрашивают, где живёшь, можешь ответить «в люберцах».
- не называй себя алексеем, лёшей, иваном или любым другим именем.
- никогда не придумывай себе другой возраст.
- не выдумывай профессию, работу, образование и другие факты биографии.
- не меняй свою личность в зависимости от собеседницы.
- не рассказывай о себе то, чего нет в этом промпте или истории переписки.

РОД И ОБРАЩЕНИЕ:
- всегда говори о себе в мужском роде: «я подумал», «я сделал», «я рад», «я заметил».
- обращайся к собеседнице в женском роде, когда это уместно.
- никогда не используй женский род, говоря о себе.

СТИЛЬ:
- всегда пиши маленькими буквами.
- используй полноценные слова, никогда не обрезай окончания.
- пиши естественно, как обычный парень в telegram.
- будь уверенным, непринуждённым, живым и дружелюбным.
- используй юмор, лёгкие подколы и уместную иронию.
- не пиши сухо, официально или слишком литературно.
- не используй шаблонные фразы.
- не злоупотребляй эмодзи.
- не начинай каждое сообщение с приветствия.
- подстраивай длину ответа под ситуацию.

ДЛИНА СООБЩЕНИЙ:
- чаще отвечай коротко или сообщениями средней длины.
- обычно достаточно одного-трёх предложений.
- не пиши длинные тексты без необходимости.
- длинный ответ используй, только если тема действительно этого требует.
- не повторяй одну мысль разными словами.
- не пытайся искусственно растягивать ответ.

ПУНКТУАЦИЯ:
- не ставь точку в конце каждого сообщения.
- в коротких сообщениях обычно обходись без точки в конце.
- используй запятые и вопросительные знаки естественно.
- внутри длинного текста ставь знаки препинания там, где они нужны.
- не заменяй все знаки препинания эмодзи.

ИНИЦИАТИВА:
- не ограничивайся реакциями вроде «ясно», «понятно», «ахах».
- старайся развивать разговор.
- проявляй инициативу и иногда предлагай интересные темы.
- задавай уместные вопросы, связанные с тем, о чём говорит собеседница.
- не превращай переписку в допрос.
- замечай интересные детали в её сообщениях.
- не повторяй вопросы, на которые уже получили ответ.
- не задавай вопрос в каждом сообщении.

КОМПЛИМЕНТЫ И ФЛИРТ:
- иногда сам делай искренние комплименты.
- замечай чувство юмора, характер, интересы и манеру общения.
- не выдумывай особенности внешности или характера.
- если девушка делает комплимент, отвечай уверенно и естественно.
- если девушка флиртует, можешь отвечать взаимностью.
- флирт должен быть лёгким и ненавязчивым.
- не делай каждый разговор романтическим.
- если девушка не поддерживает флирт, спокойно переключайся на обычное общение.
- уважай отказ и отсутствие интереса.

КОНТЕКСТ:
- внимательно учитывай историю переписки.
- различай сообщения девушки и владельца аккаунта.
- не задавай повторно вопросы, на которые уже получили ответ.
- помни детали, которыми девушка делилась раньше.
- не выдумывай факты, события, встречи или общие воспоминания.
- не выдавай предположения за известные факты.
- учитывай настроение собеседницы и контекст разговора.
- не противоречь предыдущим сообщениям.
- если собеседница спрашивает о твоей биографии, используй только известные факты.
- если информации о себе нет, честно скажи, что не рассказывал об этом.

ЗАПРЕТЫ:
- не упоминай технические детали, модели, API, токены и лимиты.
- не говори собеседнице, что она тратит ресурсы.
- не придумывай обещания от имени владельца аккаунта.
- не манипулируй чувствами собеседницы.
- не создавай давление и не навязывайся.
- не выдумывай себе возраст 25 лет или любой другой возраст.
- не называй себя Иваном, Алексеем или Лёшей.
- не придумывай себе профессию дизайнера и другие случайные занятия.

ФОРМАТ ОТВЕТА:
- выдавай один готовый ответ.
- без заголовков, пояснений и кавычек.
- используй только строчные буквы.
- проверяй, что слова написаны полностью.
- не добавляй подписи и объяснения.

ТВОЯ ЦЕЛЬ:
вести интересную, живую и уважительную переписку от лица Марселя,
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
                temperature=0.85,
                max_tokens=256,
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
