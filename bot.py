#!/usr/bin/env python3
"""Telegram bot with a daily, community-adjusted illustrative clock."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import math
import os
import random
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont


SIZE = 1200
BG = "#10131a"
WHITE = "#f4f7fb"
MUTED = "#9da7b5"
MIN_VOTES = 2
MAX_VOTE = 10
RANDOM_RISE_CHANCE = 0.10
RANDOM_RISE = 1
TIMEZONE = ZoneInfo("Europe/Moscow")
POST_HOUR = 9
POST_MINUTE = 0
STATE_PATH = Path(os.getenv("STATE_PATH", "state.json"))


class State:
    # ponytail: one local JSON file; use SQLite if the bot runs in several processes.
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"chats": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            data.setdefault("chats", {})
            return data
        except (OSError, json.JSONDecodeError):
            logging.warning("Не удалось прочитать %s; начинаю с чистого состояния", self.path)
            return {"chats": {}}

    def save(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def chat(self, chat_id: int) -> dict[str, Any]:
        chat = self.data["chats"].setdefault(
            str(chat_id),
            {"value": 50, "votes": {}, "signals": {}},
        )
        chat.setdefault("value", 50)
        chat.setdefault("votes", {})
        chat.setdefault("signals", {})
        return chat


def clamp(value: int) -> int:
    return max(0, min(100, value))


def parse_vote(args: str) -> int:
    parts = args.strip().split()
    value = int(parts[0]) if parts else 1
    if not 1 <= value <= MAX_VOTE:
        raise ValueError(f"Укажите число от 1 до {MAX_VOTE}")
    return value


def record_vote(chat: dict[str, Any], user_id: int, delta: int) -> int:
    chat["votes"][str(user_id)] = delta
    return len(chat["votes"])


def classify_message(text: str) -> int:
    """Return a small signal from a chat message, not a claim about reality."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if "мобилизац" not in text and "мобк" not in text and "призыв" not in text:
        return 0

    negative = (
        "не планир",
        "не будет",
        "не объяв",
        "не ввод",
        "не рассматрива",
        "не готов",
        "не требуется",
        "нет необходимости",
        "планов нет",
        "не собира",
        "отмен",
        "отрица",
        "опроверг",
        "фейк",
    )
    positive = (
        "объявл",
        "начал",
        "начинает",
        "готовят",
        "готовится",
        "проведут",
        "проводят",
        "будет моб",
        "мобк будет",
        "мобка будет",
        "мобилизац будет",
        "вторая волна",
        "новый призыв",
        "новая мобилизац",
        "увеличат призыв",
    )
    if any(word in text for word in negative):
        return -2
    if any(word in text for word in positive):
        return 2
    return 0


def settle_chat(chat: dict[str, Any], random_delta: int) -> dict[str, int]:
    votes = list(chat["votes"].values())
    signals = list(chat["signals"].values())
    vote_delta = round(sum(votes) / len(votes)) if len(votes) >= MIN_VOTES else 0
    message_delta = round(sum(signals) / len(signals)) if signals else 0
    old_value = chat["value"]
    total_delta = vote_delta + message_delta + random_delta
    chat["value"] = clamp(old_value + total_delta)
    chat["votes"] = {}
    chat["signals"] = {}
    return {
        "old": old_value,
        "new": chat["value"],
        "votes": len(votes),
        "signals": len(signals),
        "vote_delta": vote_delta,
        "message_delta": message_delta,
        "random_delta": random_delta,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)
    paths = [Path("/usr/share/fonts/truetype/dejavu") / name for name in names]
    paths += [Path("/usr/share/fonts/truetype/liberation2") / name for name in names]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def point(angle: float, radius: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return SIZE / 2 + math.cos(radians) * radius, SIZE / 2 + math.sin(radians) * radius


def arc_points(start: float, end: float, radius: float, steps: int = 240) -> list[tuple[float, float]]:
    return [point(start + (end - start) * i / steps, radius) for i in range(steps + 1)]


def arc_band(start: float, end: float, outer: float, inner: float) -> list[tuple[float, float]]:
    return arc_points(start, end, outer) + arc_points(end, start, inner)


def render_clock(value: int) -> io.BytesIO:
    image = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(image)
    center = (SIZE // 2, SIZE // 2)

    draw.ellipse((105, 105, SIZE - 105, SIZE - 105), outline="#252c38", width=5)
    draw.polygon(arc_band(135, 405, 431, 379), fill="#28313e")
    for start, end, color in ((135, 243, "#38b87c"), (243, 324, "#e8b84a"), (324, 405, "#e05252")):
        draw.polygon(arc_band(start, end, 431, 379), fill=color)

    for tick in range(0, 101, 10):
        if tick == 50:
            continue
        angle = 135 + 270 * tick / 100
        draw.line((point(angle, 470), point(angle, 420 if tick % 20 else 395)), fill=WHITE, width=8 if tick % 20 else 12)

    needle_end = point(135 + 270 * value / 100, 350)
    draw.line((center, needle_end), fill=WHITE, width=16)
    draw.ellipse((center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25), fill=WHITE)
    draw.ellipse((center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10), fill="#e05252")

    title_font = font(48, bold=True)
    value_font = font(170, bold=True)
    label_font = font(34)
    small_font = font(27)
    title = "ИНДЕКС НАПРЯЖЁННОСТИ"
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((SIZE - (box[2] - box[0])) / 2, 65), title, font=title_font, fill=WHITE)
    text = f"{value}%"
    box = draw.textbbox((0, 0), text, font=value_font)
    draw.text(((SIZE - (box[2] - box[0])) / 2, 480), text, font=value_font, fill=WHITE)

    for text, angle, radius in (("0", 135, 535), ("50", 270, 455), ("100", 405, 535)):
        x, y = point(angle, radius)
        box = draw.textbbox((0, 0), text, font=label_font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=label_font, fill=MUTED)

    caption = "Тема: мобилизация в РФ"
    box = draw.textbbox((0, 0), caption, font=label_font)
    draw.text(((SIZE - (box[2] - box[0])) / 2, 870), caption, font=label_font, fill=WHITE)
    note = "Оценочная визуализация, не прогноз и не официальная статистика"
    box = draw.textbbox((0, 0), note, font=small_font)
    draw.text(((SIZE - (box[2] - box[0])) / 2, 930), note, font=small_font, fill=MUTED)

    output = io.BytesIO()
    output.name = "doomsday-clock.png"
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output


def help_text() -> str:
    return (
        "Команды:\n"
        "/clock — текущие часы и картинка\n"
        "/up [1-10] — предложить повысить\n"
        "/down [1-10] — предложить понизить\n"
        "/vote — текущие голоса\n\n"
        f"В конце дня решение принимается при минимум {MIN_VOTES} уникальных голосах. "
        "Бот также учитывает сообщения чата с явными формулировками о мобилизации."
    )


async def send_clock(bot: Any, chat_id: int, value: int, caption: str) -> None:
    from aiogram.types import BufferedInputFile

    await bot.send_photo(
        chat_id,
        BufferedInputFile(render_clock(value).getvalue(), filename="doomsday-clock.png"),
        caption=caption,
    )


def register_handlers(dp: Any, state: State) -> None:
    from aiogram import F, Router
    from aiogram.filters import Command, CommandObject, CommandStart
    from aiogram.types import Message

    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        async with state.lock:
            state.chat(message.chat.id)
            state.save()
        await message.answer("Это художественный индекс напряжённости.\n\n" + help_text())

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(help_text())

    @router.message(Command("clock"))
    async def clock_command(message: Message) -> None:
        async with state.lock:
            chat = state.chat(message.chat.id)
            value = chat["value"]
            state.save()
        await send_clock(message.bot, message.chat.id, value, f"Сейчас: {value}%\nОценочная визуализация, не прогноз.")

    async def vote_command(message: Message, command: CommandObject, direction: int) -> None:
        try:
            amount = parse_vote(command.args or "")
        except ValueError as error:
            await message.answer(str(error))
            return
        user_id = message.from_user.id if message.from_user else message.chat.id
        async with state.lock:
            chat = state.chat(message.chat.id)
            count = record_vote(chat, user_id, direction * amount)
            state.save()
        direction_text = "повысить" if direction > 0 else "понизить"
        await message.answer(f"Предложение {direction_text} на {amount} п. принято. Голосов сегодня: {count}.")

    @router.message(Command("up"))
    async def up_command(message: Message, command: CommandObject) -> None:
        await vote_command(message, command, 1)

    @router.message(Command("down"))
    async def down_command(message: Message, command: CommandObject) -> None:
        await vote_command(message, command, -1)

    @router.message(Command("vote"))
    async def vote_status(message: Message) -> None:
        async with state.lock:
            chat = state.chat(message.chat.id)
            votes = list(chat["votes"].values())
            signals = list(chat["signals"].values())
            value = chat["value"]
            state.save()
        total = sum(votes)
        average = round(total / len(votes), 1) if votes else 0
        await message.answer(
            f"Сейчас: {value}%\n"
            f"Явных голосов: {len(votes)}\n"
            f"Среднее предложение: {average:+g} п.\n"
            f"Автосигналов из сообщений: {len(signals)}"
        )

    @router.message(F.text)
    async def read_chat_message(message: Message) -> None:
        if message.text.startswith("/") or not message.from_user or message.from_user.is_bot:
            return
        signal = classify_message(message.text)
        async with state.lock:
            chat_key = str(message.chat.id)
            is_new_chat = chat_key not in state.data["chats"]
            chat = state.chat(message.chat.id)
            if signal:
                # One signal per participant per day: a later matching message replaces the earlier one.
                chat["signals"][str(message.from_user.id)] = signal
            if is_new_chat or signal:
                state.save()


    dp.include_router(router)


async def daily_worker(bot: Any, state: State) -> None:
    while True:
        now = datetime.now(TIMEZONE)
        target = now.replace(hour=POST_HOUR, minute=POST_MINUTE, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())

        random_delta = RANDOM_RISE if random.random() < RANDOM_RISE_CHANCE else 0
        async with state.lock:
            results = []
            for chat_id, chat in state.data["chats"].items():
                results.append((int(chat_id), settle_chat(chat, random_delta)))
            state.save()

        for chat_id, result in results:
            caption = (
                f"Ежедневный итог: {result['new']}%\n"
                f"Голоса: {result['vote_delta']:+d} п. ({result['votes']} голосов)\n"
                f"Сообщения: {result['message_delta']:+d} п. ({result['signals']} сигналов)\n"
                f"Случайный шум: {result['random_delta']:+d} п.\n"
                "Это оценочная визуализация, не прогноз."
            )
            try:
                await send_clock(bot, chat_id, result["new"], caption)
            except Exception as error:
                logging.warning("Не удалось отправить ежедневный пост в %s: %s", chat_id, error)


def self_check() -> None:
    assert parse_vote("") == 1
    assert parse_vote("10") == 10
    assert classify_message("Власти заявили: новую мобилизацию не планируют") == -2
    assert classify_message("Объявлена новая мобилизация") == 2
    chat = {"value": 50, "votes": {"1": 4, "2": 2}, "signals": {"1": -2}}
    result = settle_chat(chat, 1)
    assert result["new"] == 52
    assert not chat["votes"] and not chat["signals"]
    assert render_clock(42).read(8) == b"\x89PNG\r\n\x1a\n"
    print("ok")


async def main() -> None:
    from aiogram import Bot, Dispatcher

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN")
    state = State(STATE_PATH)
    bot = Bot(token)
    dp = Dispatcher()
    register_handlers(dp, state)
    worker = asyncio.create_task(daily_worker(bot, state))
    try:
        await dp.start_polling(bot)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


if __name__ == "__main__":
    if "--check" in sys.argv:
        self_check()
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        asyncio.run(main())
