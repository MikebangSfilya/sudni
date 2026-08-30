#!/usr/bin/env python3
"""Telegram bot with a community-adjusted hourly shake index."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image, ImageDraw, ImageFont


SIZE = 1200
BG = "#10131a"
WHITE = "#f4f7fb"
MUTED = "#9da7b5"
DEFAULT_VALUE = 50
MIN_VOTES = 2
MIN_SIGNALS = 2
MAX_VOTE = 10
MAX_CHANGE_PER_REVIEW = 10
# At the fastest supported interval this still preserves fourteen days.
MAX_HISTORY = 14 * 24 * 12
VOTE_TTL = timedelta(hours=24)
SIGNAL_TTL = timedelta(hours=6)
STATE_PATH = Path(os.getenv("STATE_PATH", "state.json"))


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        logging.warning("%s=%r не является числом; использую %s", name, raw, default)
        return default


def load_timezone() -> ZoneInfo:
    name = os.getenv("BOT_TIMEZONE", "Europe/Moscow")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logging.warning("Часовой пояс %r не найден; использую Europe/Moscow", name)
        return ZoneInfo("Europe/Moscow")


TIMEZONE = load_timezone()
# The upper bound guarantees that a review happens at least once an hour.
REVIEW_INTERVAL_SECONDS = env_int("REVIEW_INTERVAL_SECONDS", 3600, 300, 3600)
DAILY_POST_HOUR = env_int("DAILY_POST_HOUR", 9, 0, 23)
DAILY_POST_MINUTE = env_int("DAILY_POST_MINUTE", 0, 0, 59)
QUICK_VOTE_BUTTONS = (
    ("▲ Трясёт", "shake:up"),
    ("▼ Отпускает", "shake:down"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def new_chat() -> dict[str, Any]:
    return {
        "value": DEFAULT_VALUE,
        "votes": {},
        "signals": {},
        "history": [],
        "last_review_at": None,
        "last_result": None,
    }


class State:
    """Small atomic JSON store suitable for a single bot process."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 2, "chats": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("корень JSON должен быть объектом")
            if not isinstance(data.get("chats"), dict):
                data["chats"] = {}
            data["version"] = 2
            return data
        except (OSError, json.JSONDecodeError, ValueError) as error:
            logging.error(
                "Не удалось прочитать %s (%s); начинаю с чистого состояния",
                self.path,
                error,
            )
            return {"version": 2, "chats": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        with temp.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(self.path)

    def chat(self, chat_id: int) -> dict[str, Any]:
        chat = self.data["chats"].setdefault(str(chat_id), new_chat())
        if not isinstance(chat, dict):
            logging.warning("Повреждённое состояние чата %s заменено", chat_id)
            chat = new_chat()
            self.data["chats"][str(chat_id)] = chat
        for key, value in new_chat().items():
            chat.setdefault(key, value)
        try:
            chat["value"] = clamp(int(chat.get("value", DEFAULT_VALUE)))
        except (TypeError, ValueError):
            chat["value"] = DEFAULT_VALUE
        if not isinstance(chat.get("votes"), dict):
            chat["votes"] = {}
        if not isinstance(chat.get("signals"), dict):
            chat["signals"] = {}
        if not isinstance(chat.get("history"), list):
            chat["history"] = []
        return chat


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def parse_vote(args: str) -> int:
    parts = args.strip().split()
    if len(parts) > 1:
        raise ValueError(f"Формат: число от 1 до {MAX_VOTE}")
    try:
        value = int(parts[0]) if parts else 1
    except ValueError as error:
        raise ValueError(f"Укажите целое число от 1 до {MAX_VOTE}") from error
    if not 1 <= value <= MAX_VOTE:
        raise ValueError(f"Укажите число от 1 до {MAX_VOTE}")
    return value


def observation_value(raw: Any) -> int | None:
    """Read both current observations and legacy integer values."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("value"), int):
        return raw["value"]
    return None


def observation_time(raw: Any, fallback: datetime) -> datetime:
    if isinstance(raw, dict):
        return parse_datetime(raw.get("at")) or fallback
    return fallback


def active_observations(
    observations: dict[str, Any], now: datetime, ttl: timedelta
) -> tuple[list[int], set[str]]:
    values: list[int] = []
    expired: set[str] = set()
    for user_id, raw in observations.items():
        value = observation_value(raw)
        # Legacy observations have no timestamp; keep them for one review after migration.
        created_at = observation_time(raw, now)
        if value is None or created_at < now - ttl:
            expired.add(user_id)
        else:
            values.append(value)
    return values, expired


def record_observation(
    observations: dict[str, Any], user_id: int, value: int, created_at: datetime
) -> int:
    observations[str(user_id)] = {"value": value, "at": created_at.isoformat()}
    return len(observations)


def classify_message(text: str) -> int:
    """Return a conservative chat signal, not a claim about reality."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    if not re.search(r"\b(мобилизац\w*|мобк\w*|призыв\w*)\b", text):
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
    if any(fragment in text for fragment in negative):
        return -2
    if any(fragment in text for fragment in positive):
        return 2
    return 0


def settle_chat(chat: dict[str, Any], reviewed_at: datetime) -> dict[str, Any]:
    votes, expired_votes = active_observations(chat["votes"], reviewed_at, VOTE_TTL)
    signals, expired_signals = active_observations(chat["signals"], reviewed_at, SIGNAL_TTL)
    for user_id in expired_votes:
        chat["votes"].pop(user_id, None)
    for user_id in expired_signals:
        chat["signals"].pop(user_id, None)

    votes_used = len(votes) >= MIN_VOTES
    signals_used = len(signals) >= MIN_SIGNALS
    # Median makes coordinated outliers less powerful than a simple sum.
    vote_delta = round(median(votes)) if votes_used else 0
    message_delta = round(sum(signals) / len(signals)) if signals_used else 0
    total_delta = clamp(
        vote_delta + message_delta,
        -MAX_CHANGE_PER_REVIEW,
        MAX_CHANGE_PER_REVIEW,
    )
    old_value = clamp(int(chat.get("value", DEFAULT_VALUE)))
    new_value = clamp(old_value + total_delta)

    if votes_used:
        chat["votes"] = {}
    if signals_used:
        chat["signals"] = {}

    result: dict[str, Any] = {
        "at": reviewed_at.isoformat(),
        "old": old_value,
        "new": new_value,
        "votes": len(votes),
        "signals": len(signals),
        "vote_delta": vote_delta,
        "message_delta": message_delta,
        "total_delta": new_value - old_value,
        "votes_used": len(votes) if votes_used else 0,
        "signals_used": len(signals) if signals_used else 0,
        "pending_votes": 0 if votes_used else len(votes),
        "pending_signals": 0 if signals_used else len(signals),
    }
    chat["value"] = new_value
    chat["last_review_at"] = reviewed_at.isoformat()
    chat["last_result"] = result
    chat["history"].append(result)
    chat["history"] = chat["history"][-MAX_HISTORY:]
    return result


def shake_level(value: int) -> str:
    if value < 25:
        return "штиль"
    if value < 50:
        return "потряхивает"
    if value < 75:
        return "трясёт"
    return "сильно трясёт"


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


def arc_points(
    start: float, end: float, radius: float, steps: int = 240
) -> list[tuple[float, float]]:
    return [point(start + (end - start) * i / steps, radius) for i in range(steps + 1)]


def arc_band(start: float, end: float, outer: float, inner: float) -> list[tuple[float, float]]:
    return arc_points(start, end, outer) + arc_points(end, start, inner)


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: float,
    text: str,
    selected_font: Any,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=selected_font)
    draw.text(((SIZE - (box[2] - box[0])) / 2, y), text, font=selected_font, fill=fill)


def render_clock(value: int) -> io.BytesIO:
    value = clamp(value)
    image = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(image)
    center = (SIZE // 2, SIZE // 2)

    draw.ellipse((105, 105, SIZE - 105, SIZE - 105), outline="#252c38", width=5)
    draw.polygon(arc_band(135, 405, 431, 379), fill="#28313e")
    for start, end, color in (
        (135, 202.5, "#38b87c"),
        (202.5, 270, "#8fbd55"),
        (270, 337.5, "#e8b84a"),
        (337.5, 405, "#e05252"),
    ):
        draw.polygon(arc_band(start, end, 431, 379), fill=color)

    for tick in range(0, 101, 10):
        if tick == 50:
            continue
        angle = 135 + 270 * tick / 100
        draw.line(
            (point(angle, 470), point(angle, 420 if tick % 20 else 395)),
            fill=WHITE,
            width=8 if tick % 20 else 12,
        )

    needle_end = point(135 + 270 * value / 100, 350)
    draw.line((center, needle_end), fill=WHITE, width=16)
    draw.ellipse((center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25), fill=WHITE)
    draw.ellipse((center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10), fill="#e05252")

    title_font = font(52, bold=True)
    value_font = font(180, bold=True)
    label_font = font(34)
    level_font = font(44, bold=True)
    centered_text(draw, 28, "ИНДЕКС ТРЯСКИ", title_font, WHITE)
    centered_text(draw, 300, str(value), value_font, WHITE)
    centered_text(draw, 665, f"{shake_level(value).upper()} · ИЗ 100", level_font, WHITE)

    for label, angle, radius in (("0", 135, 535), ("50", 270, 455), ("100", 405, 535)):
        x, y = point(angle, radius)
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2),
            label,
            font=label_font,
            fill=MUTED,
        )

    centered_text(draw, 870, "Тема: мобилизация в РФ", label_font, WHITE)

    output = io.BytesIO()
    output.name = "shake-index.png"
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output


def local_time_text(value: Any) -> str:
    parsed = parse_datetime(value)
    if not parsed:
        return "ещё не было"
    return parsed.astimezone(TIMEZONE).strftime("%d.%m %H:%M")


def help_text() -> str:
    return (
        "Индекс тряски обновляется не реже раза в час.\n\n"
        "Команды:\n"
        "/clock — индекс и циферблат\n"
        "/up [1–10] — предложить повышение\n"
        "/down [1–10] — предложить снижение\n"
        "/vote — накопленные голоса и сигналы\n"
        "/why — детали последнего пересчёта\n"
        "/history — изменения за последние сутки\n\n"
        f"Для влияния нужны минимум {MIN_VOTES} уникальных голоса или {MIN_SIGNALS} независимых "
        "сигнала из сообщений. Голос живёт 24 часа, сигнал — 6 часов. "
        "На карточках можно голосовать кнопками."
    )


def movement_icon(delta: int) -> str:
    if delta > 0:
        return "📈"
    if delta < 0:
        return "📉"
    return "➖"


def review_caption(result: dict[str, Any], heading: str = "Почасовой пересчёт") -> str:
    delta = result["total_delta"]
    old_level = shake_level(result["old"])
    new_level = shake_level(result["new"])
    lines = [
        f"{movement_icon(delta)} {heading}: {result['new']}/100 — {new_level}",
        f"Изменение: {delta:+d} п.",
        f"Голоса: {result['vote_delta']:+d} п. ({result['votes']})",
        f"Сигналы чата: {result['message_delta']:+d} п. ({result['signals']})",
    ]
    if old_level != new_level:
        lines.insert(1, f"⚡ Новый режим: {old_level} → {new_level}")
    return "\n".join(lines)


def vote_keyboard() -> Any:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=text, callback_data=data)
                for text, data in QUICK_VOTE_BUTTONS
            ]
        ]
    )


async def send_clock(bot: Any, chat_id: int, value: int, caption: str) -> None:
    from aiogram.types import BufferedInputFile

    image = await asyncio.to_thread(render_clock, value)
    await bot.send_photo(
        chat_id,
        BufferedInputFile(image.getvalue(), filename="shake-index.png"),
        caption=caption,
        reply_markup=vote_keyboard(),
    )


def register_handlers(dp: Any, state: State) -> None:
    from aiogram import F, Router
    from aiogram.filters import Command, CommandObject, CommandStart
    from aiogram.types import CallbackQuery, Message

    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        async with state.lock:
            state.chat(message.chat.id)
            state.save()
        await message.answer(help_text())

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(help_text())

    @router.message(Command("clock", "index"))
    async def clock_command(message: Message) -> None:
        async with state.lock:
            chat = state.chat(message.chat.id)
            value = chat["value"]
            reviewed = local_time_text(chat["last_review_at"])
        caption = (
            f"Сейчас: {value}/100 — {shake_level(value)}\n"
            f"Последний пересчёт: {reviewed}"
        )
        await send_clock(message.bot, message.chat.id, value, caption)

    async def vote_command(message: Message, command: CommandObject, direction: int) -> None:
        try:
            amount = parse_vote(command.args or "")
        except ValueError as error:
            await message.answer(str(error))
            return
        user_id = message.from_user.id if message.from_user else message.chat.id
        async with state.lock:
            chat = state.chat(message.chat.id)
            count = record_observation(chat["votes"], user_id, direction * amount, utc_now())
            state.save()
        direction_text = "повысить" if direction > 0 else "понизить"
        await message.answer(
            f"Предложение {direction_text} на {amount} п. принято. "
            f"Активных голосов: {count}/{MIN_VOTES} минимум. Пересчёт — в течение часа."
        )

    @router.message(Command("up"))
    async def up_command(message: Message, command: CommandObject) -> None:
        await vote_command(message, command, 1)

    @router.message(Command("down"))
    async def down_command(message: Message, command: CommandObject) -> None:
        await vote_command(message, command, -1)

    @router.callback_query(F.data.in_({"shake:up", "shake:down"}))
    async def quick_vote(callback: CallbackQuery) -> None:
        if not callback.message:
            await callback.answer("Карточка уже недоступна")
            return
        direction = 1 if callback.data == "shake:up" else -1
        async with state.lock:
            chat = state.chat(callback.message.chat.id)
            count = record_observation(
                chat["votes"], callback.from_user.id, direction, utc_now()
            )
            state.save()
        action = "Трясёт сильнее: +1" if direction > 0 else "Отпускает: −1"
        await callback.answer(f"{action}. Активных голосов: {count}")

    @router.message(Command("vote", "status"))
    async def vote_status(message: Message) -> None:
        async with state.lock:
            chat = state.chat(message.chat.id)
            votes, _ = active_observations(chat["votes"], utc_now(), VOTE_TTL)
            signals, _ = active_observations(chat["signals"], utc_now(), SIGNAL_TTL)
            value = chat["value"]
            reviewed = local_time_text(chat["last_review_at"])
        vote_median = median(votes) if votes else 0
        await message.answer(
            f"Индекс: {value}/100 — {shake_level(value)}\n"
            f"Активных голосов: {len(votes)} (медиана {vote_median:+g} п.)\n"
            f"Сигналов из сообщений: {len(signals)}\n"
            f"Последний пересчёт: {reviewed}"
        )

    @router.message(Command("why"))
    async def why_command(message: Message) -> None:
        async with state.lock:
            result = state.chat(message.chat.id).get("last_result")
        if not result:
            await message.answer(
                "Пересчёта ещё не было. Он произойдёт автоматически в течение часа."
            )
            return
        heading = f"Пересчёт {local_time_text(result.get('at'))}"
        await message.answer(review_caption(result, heading))

    @router.message(Command("history"))
    async def history_command(message: Message) -> None:
        cutoff = utc_now() - timedelta(hours=24)
        async with state.lock:
            history = list(state.chat(message.chat.id)["history"])
        recent = [
            item for item in history
            if isinstance(item, dict)
            and (parse_datetime(item.get("at")) or cutoff) >= cutoff
        ]
        changes = [item for item in recent if item.get("total_delta")]
        if not changes:
            await message.answer(
                f"За последние 24 часа изменений не было. Пересчётов: {len(recent)}."
            )
            return
        values = [recent[0]["old"], *(item["new"] for item in recent)]
        total_change = values[-1] - values[0]
        lines = [
            f"{movement_icon(total_change)} За 24 часа: {values[0]} → {values[-1]} "
            f"({total_change:+d})",
            f"Диапазон: {min(values)}–{max(values)}",
            "",
            "Последние изменения:",
        ]
        for item in changes[-12:]:
            lines.append(
                f"{local_time_text(item.get('at'))}: {item['old']} → {item['new']} "
                f"({item['total_delta']:+d})"
            )
        if len(changes) > 12:
            lines.append(f"…и ещё {len(changes) - 12}")
        await message.answer("\n".join(lines))

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
                # A later matching message replaces the participant's earlier signal.
                record_observation(chat["signals"], message.from_user.id, signal, utc_now())
            if is_new_chat or signal:
                state.save()

    dp.include_router(router)


async def review_all_chats(bot: Any, state: State) -> None:
    reviewed_at = utc_now()
    async with state.lock:
        results: list[tuple[int, dict[str, Any]]] = []
        for raw_chat_id in list(state.data["chats"]):
            try:
                chat_id = int(raw_chat_id)
            except (TypeError, ValueError):
                logging.warning("Пропускаю некорректный id чата %r", raw_chat_id)
                continue
            results.append((chat_id, settle_chat(state.chat(chat_id), reviewed_at)))
        if results:
            state.save()

    # Do not spam quiet chats: publish an hourly card only when the index changed.
    for chat_id, result in results:
        if not result["total_delta"]:
            continue
        try:
            await send_clock(bot, chat_id, result["new"], review_caption(result))
        except Exception as error:
            logging.warning("Не удалось отправить почасовой пост в %s: %s", chat_id, error)


async def hourly_worker(bot: Any, state: State) -> None:
    while True:
        started = asyncio.get_running_loop().time()
        try:
            await review_all_chats(bot, state)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Ошибка почасового пересчёта")
        elapsed = asyncio.get_running_loop().time() - started
        await asyncio.sleep(max(1, REVIEW_INTERVAL_SECONDS - elapsed))


async def daily_summary_worker(bot: Any, state: State) -> None:
    while True:
        now = datetime.now(TIMEZONE)
        target = now.replace(
            hour=DAILY_POST_HOUR,
            minute=DAILY_POST_MINUTE,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())

        cutoff = utc_now() - timedelta(hours=24)
        async with state.lock:
            summaries: list[tuple[int, int, list[dict[str, Any]]]] = []
            for raw_chat_id in list(state.data["chats"]):
                try:
                    chat_id = int(raw_chat_id)
                except (TypeError, ValueError):
                    logging.warning("Пропускаю некорректный id чата %r", raw_chat_id)
                    continue
                chat = state.chat(chat_id)
                recent = [
                    item for item in chat["history"]
                    if isinstance(item, dict)
                    and (parse_datetime(item.get("at")) or cutoff) >= cutoff
                ]
                summaries.append((chat_id, chat["value"], recent))

        for chat_id, value, recent in summaries:
            start_value = recent[0]["old"] if recent else value
            change = value - start_value
            values = [start_value, *(item["new"] for item in recent)]
            votes_used = sum(item.get("votes_used", 0) for item in recent)
            signals_used = sum(item.get("signals_used", 0) for item in recent)
            caption = (
                f"{movement_icon(change)} Сводка за сутки: {value}/100 — {shake_level(value)}\n"
                f"Изменение за 24 часа: {change:+d} п.\n"
                f"Диапазон: {min(values)}–{max(values)}\n"
                f"Активность: {votes_used} голосов · {signals_used} сигналов"
            )
            try:
                await send_clock(bot, chat_id, value, caption)
            except Exception as error:
                logging.warning("Не удалось отправить суточную сводку в %s: %s", chat_id, error)


async def configure_bot(bot: Any) -> None:
    from aiogram.types import BotCommand

    commands = [
        BotCommand(command="clock", description="текущий индекс тряски"),
        BotCommand(command="index", description="текущий индекс тряски"),
        BotCommand(command="up", description="предложить повышение на 1–10"),
        BotCommand(command="down", description="предложить снижение на 1–10"),
        BotCommand(command="vote", description="активные голоса и сигналы"),
        BotCommand(command="status", description="активные голоса и сигналы"),
        BotCommand(command="why", description="почему индекс изменился"),
        BotCommand(command="history", description="изменения за сутки"),
        BotCommand(command="help", description="как работает бот"),
    ]
    await bot.set_my_commands(commands)


def self_check() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_vote("") == 1
    assert parse_vote("10") == 10
    assert classify_message("Власти заявили: новую мобилизацию не планируют") == -2
    assert classify_message("Объявлена новая мобилизация") == 2
    chat = new_chat()
    record_observation(chat["votes"], 1, 4, now)
    record_observation(chat["votes"], 2, 2, now)
    record_observation(chat["signals"], 1, -2, now)
    record_observation(chat["signals"], 2, -2, now)
    result = settle_chat(chat, now)
    assert result["new"] == 51
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
    workers: list[asyncio.Task[None]] = []
    try:
        try:
            await configure_bot(bot)
        except Exception as error:
            # A temporary Bot API failure should not prevent polling from starting.
            logging.warning("Не удалось обновить меню команд: %s", error)
        workers = [
            asyncio.create_task(hourly_worker(bot, state), name="hourly-review"),
            asyncio.create_task(daily_summary_worker(bot, state), name="daily-summary"),
        ]
        await dp.start_polling(bot)
    finally:
        for worker in workers:
            worker.cancel()
        for worker in workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        await bot.session.close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        self_check()
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(message)s",
            stream=sys.stdout,
        )
        asyncio.run(main())
