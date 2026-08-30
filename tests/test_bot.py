import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import bot


NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


class VoteParsingTests(unittest.TestCase):
    def test_default_and_bounds(self) -> None:
        self.assertEqual(bot.parse_vote(""), 1)
        self.assertEqual(bot.parse_vote("10"), 10)
        for invalid in ("0", "11", "one", "1 2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bot.parse_vote(invalid)


class SchedulingTests(unittest.TestCase):
    def test_review_interval_cannot_exceed_one_hour(self) -> None:
        with patch.dict("os.environ", {"TEST_REVIEW_INTERVAL": "99999"}):
            interval = bot.env_int("TEST_REVIEW_INTERVAL", 3600, 300, 3600)
        self.assertEqual(interval, 3600)
        self.assertLessEqual(bot.REVIEW_INTERVAL_SECONDS, 3600)


class MessageClassificationTests(unittest.TestCase):
    def test_requires_topic_and_explicit_direction(self) -> None:
        self.assertEqual(bot.classify_message("Просто тревожная новость"), 0)
        self.assertEqual(bot.classify_message("Поговорим о мобилизации"), 0)
        self.assertEqual(bot.classify_message("Объявлена новая мобилизация"), 2)
        self.assertEqual(
            bot.classify_message("Новую мобилизацию не планируют, это фейк"), -2
        )


class SettlementTests(unittest.TestCase):
    def test_single_vote_waits_for_quorum(self) -> None:
        chat = bot.new_chat()
        bot.record_observation(chat["votes"], 1, 6, NOW)

        first = bot.settle_chat(chat, NOW)

        self.assertEqual(first["new"], bot.DEFAULT_VALUE)
        self.assertEqual(first["pending_votes"], 1)
        self.assertEqual(len(chat["votes"]), 1)

        bot.record_observation(chat["votes"], 2, 2, NOW)
        second = bot.settle_chat(chat, NOW + timedelta(hours=1))
        self.assertEqual(second["new"], bot.DEFAULT_VALUE + 4)
        self.assertFalse(chat["votes"])

    def test_expired_observations_are_removed(self) -> None:
        chat = bot.new_chat()
        bot.record_observation(chat["votes"], 1, 10, NOW - bot.VOTE_TTL - timedelta(seconds=1))
        bot.record_observation(chat["signals"], 1, 2, NOW - bot.SIGNAL_TTL - timedelta(seconds=1))

        result = bot.settle_chat(chat, NOW)

        self.assertEqual(result["total_delta"], 0)
        self.assertFalse(chat["votes"])
        self.assertFalse(chat["signals"])

    def test_change_and_index_are_bounded(self) -> None:
        chat = bot.new_chat()
        chat["value"] = 97
        for user_id in (1, 2):
            bot.record_observation(chat["votes"], user_id, 10, NOW)
            bot.record_observation(chat["signals"], user_id, 2, NOW)

        result = bot.settle_chat(chat, NOW)

        self.assertEqual(result["new"], 100)
        self.assertEqual(result["total_delta"], 3)

    def test_history_is_bounded(self) -> None:
        chat = bot.new_chat()
        for offset in range(bot.MAX_HISTORY + 5):
            bot.settle_chat(chat, NOW + timedelta(hours=offset))
        self.assertEqual(len(chat["history"]), bot.MAX_HISTORY)


class StateTests(unittest.TestCase):
    def test_legacy_state_is_migrated_and_saved_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(
                json.dumps({"chats": {"42": {"value": 110, "votes": {"1": 2}}}}),
                encoding="utf-8",
            )

            state = bot.State(path)
            chat = state.chat(42)
            state.save()

            self.assertEqual(chat["value"], 100)
            self.assertEqual(chat["history"], [])
            self.assertEqual(json.loads(path.read_text())["version"], 2)
            self.assertFalse(path.with_name(".state.json.tmp").exists())

    def test_lock_is_bound_inside_running_loop(self) -> None:
        async def create_state() -> bool:
            with tempfile.TemporaryDirectory() as directory:
                state = bot.State(Path(directory) / "state.json")
                async with state.lock:
                    return True

        self.assertTrue(asyncio.run(create_state()))


class RenderingTests(unittest.TestCase):
    def test_rendered_card_is_png(self) -> None:
        self.assertEqual(bot.render_clock(42).read(8), b"\x89PNG\r\n\x1a\n")
        self.assertEqual(bot.shake_level(24), "штиль")
        self.assertEqual(bot.shake_level(25), "потряхивает")
        self.assertEqual(bot.shake_level(75), "сильно трясёт")

    def test_caption_marks_level_transition_without_disclaimer(self) -> None:
        result = {
            "old": 49,
            "new": 52,
            "total_delta": 3,
            "vote_delta": 3,
            "message_delta": 0,
            "votes": 2,
            "signals": 0,
        }
        caption = bot.review_caption(result)
        self.assertIn("Новый режим: потряхивает → трясёт", caption)
        self.assertNotIn("прогноз", caption.lower())
        self.assertNotIn("статистик", caption.lower())

    def test_card_has_quick_vote_buttons(self) -> None:
        self.assertEqual(
            [data for _, data in bot.QUICK_VOTE_BUTTONS],
            ["shake:up", "shake:down"],
        )


if __name__ == "__main__":
    unittest.main()
