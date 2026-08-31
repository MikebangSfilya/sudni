import asyncio
import importlib.util
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


class LLMConfigTests(unittest.TestCase):
    def test_openrouter_is_default_deepseek_route(self) -> None:
        config = bot.load_llm_config({"LLM_API_KEY": "openrouter-key"})

        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(config.app_name, "Судный день")
        self.assertIsNone(config.problem())

    def test_openai_preset_only_needs_neutral_key(self) -> None:
        config = bot.load_llm_config(
            {"LLM_PROVIDER": "openai", "LLM_API_KEY": "openai-key"}
        )

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-5.6-luna")
        self.assertIsNone(config.base_url)
        self.assertIsNone(config.problem())

    def test_deepseek_preset_sets_endpoint_and_model(self) -> None:
        config = bot.load_llm_config(
            {"LLM_PROVIDER": "deepseek", "LLM_API_KEY": "deepseek-key"}
        )

        self.assertEqual(config.label, "DeepSeek")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertIsNone(config.problem())

    def test_custom_provider_requires_endpoint_and_model(self) -> None:
        incomplete = bot.load_llm_config(
            {"LLM_PROVIDER": "local", "LLM_API_KEY": "key"}
        )
        complete = bot.load_llm_config(
            {
                "LLM_PROVIDER": "local",
                "LLM_API_KEY": "key",
                "LLM_BASE_URL": "http://llm.local:8000/v1/",
                "LLM_MODEL": "qwen",
            }
        )

        self.assertIn("LLM_MODEL", incomplete.problem() or "")
        self.assertEqual(complete.base_url, "http://llm.local:8000/v1")
        self.assertIsNone(complete.problem())

    def test_client_receives_configured_endpoint(self) -> None:
        class FakeAsyncOpenAI:
            options: dict[str, object] = {}

            def __init__(self, **options: object) -> None:
                self.options = options
                FakeAsyncOpenAI.options = options

        fake_module = type("FakeOpenAIModule", (), {"AsyncOpenAI": FakeAsyncOpenAI})
        config = bot.load_llm_config(
            {
                "LLM_PROVIDER": "local",
                "LLM_API_KEY": "secret",
                "LLM_BASE_URL": "https://llm.example/v1",
                "LLM_MODEL": "model-1",
            }
        )
        analyzer = bot.LLMAnalyzer(config=config)

        with patch.dict("sys.modules", {"openai": fake_module}):
            analyzer.client()

        self.assertEqual(FakeAsyncOpenAI.options["api_key"], "secret")
        self.assertEqual(
            FakeAsyncOpenAI.options["base_url"], "https://llm.example/v1"
        )

    def test_openrouter_client_receives_attribution_headers(self) -> None:
        class FakeAsyncOpenAI:
            options: dict[str, object] = {}

            def __init__(self, **options: object) -> None:
                FakeAsyncOpenAI.options = options

        fake_module = type("FakeOpenAIModule", (), {"AsyncOpenAI": FakeAsyncOpenAI})
        config = bot.load_llm_config(
            {
                "LLM_PROVIDER": "openrouter",
                "LLM_API_KEY": "secret",
                "LLM_APP_URL": "https://example.com",
                "LLM_APP_NAME": "Shake Index",
            }
        )
        analyzer = bot.LLMAnalyzer(config=config)

        with patch.dict("sys.modules", {"openai": fake_module}):
            analyzer.client()

        self.assertEqual(
            FakeAsyncOpenAI.options["base_url"], "https://openrouter.ai/api/v1"
        )
        self.assertEqual(
            FakeAsyncOpenAI.options["default_headers"],
            {
                "HTTP-Referer": "https://example.com",
                "X-OpenRouter-Title": "Shake Index",
            },
        )


class MessageClassificationTests(unittest.TestCase):
    def test_requires_topic_and_explicit_direction(self) -> None:
        self.assertEqual(bot.classify_message("Просто тревожная новость"), 0)
        self.assertEqual(bot.classify_message("Поговорим о мобилизации"), 0)
        self.assertEqual(bot.classify_message("Объявлена новая мобилизация"), 2)
        self.assertEqual(
            bot.classify_message("Новую мобилизацию не планируют, это фейк"), -2
        )


class MessageBufferTests(unittest.TestCase):
    def test_transcript_anonymizes_authors_and_preserves_slang(self) -> None:
        chat = bot.new_chat()
        bot.record_chat_message(chat, 987654321, "мобка — это рофл или не рофл?", NOW, 10)

        transcript = bot.serialize_chat_chunk(chat["messages"])

        self.assertIn("anon-1", transcript)
        self.assertIn("мобка — это рофл или не рофл?", transcript)
        self.assertNotIn("987654321", transcript)
        self.assertNotIn("user_id", chat["messages"][0])

    def test_large_window_is_split_without_losing_messages(self) -> None:
        chat = bot.new_chat()
        for message_id in range(10):
            bot.record_chat_message(chat, message_id % 2, "x" * 100, NOW, message_id)

        chunks = bot.split_message_chunks(chat["messages"], 300)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 10)

    def test_consuming_snapshot_preserves_new_arrivals(self) -> None:
        chat = bot.new_chat()
        bot.record_chat_message(chat, 1, "старое", NOW, 1)
        snapshot = list(chat["messages"])
        bot.record_chat_message(chat, 2, "новое", NOW + timedelta(minutes=1), 2)

        bot.consume_chat_messages(chat, snapshot)

        self.assertEqual([message["text"] for message in chat["messages"]], ["новое"])


class LLMAnalysisTests(unittest.TestCase):
    def test_unattributed_event_claims_are_attributed_to_chat(self) -> None:
        cases = (
            (
                "Пишут, что завтра объявят мобилизацию",
                "Участники обсуждают сообщения о возможном объявлении мобилизации.",
                "Завтра объявят мобилизацию.",
            ),
            (
                "Да это опять фейк, никакой мобки не будет",
                "Участники преимущественно относятся к слуху скептически.",
                "Мобилизации не будет.",
            ),
            (
                "Ну всё, завтра всех заберут, ахахах",
                "Участники воспринимают сообщение как возможную шутку или иронию.",
                "Завтра всех заберут, ахахах.",
            ),
        )

        for chat_message, safe_summary, unsafe_summary in cases:
            with self.subTest(chat_message=chat_message):
                safe_analysis = bot.normalize_llm_analysis(
                    {
                        "decision": "hold",
                        "delta": 0,
                        "confidence": 0.8,
                        "relevant_messages": 1,
                        "summary": safe_summary,
                        "factors": [safe_summary],
                    },
                    message_count=1,
                    model="test-model",
                )
                guarded_analysis = bot.normalize_llm_analysis(
                    {
                        "decision": "hold",
                        "delta": 0,
                        "confidence": 0.8,
                        "relevant_messages": 1,
                        "summary": unsafe_summary,
                        "factors": [unsafe_summary],
                    },
                    message_count=1,
                    model="test-model",
                )

                self.assertEqual(safe_analysis["summary"], safe_summary)
                self.assertTrue(
                    guarded_analysis["summary"].startswith(
                        "Участники выражают это так: «"
                    )
                )
                self.assertTrue(
                    guarded_analysis["factors"][0].startswith(
                        "Участники выражают это так: «"
                    )
                )
                if "ахахах" in unsafe_summary:
                    self.assertIn("ахахах", guarded_analysis["summary"])

    def test_prompt_limits_analysis_to_chat_and_preserves_irony(self) -> None:
        self.assertIn("не проверяешь факты", bot.LLM_INSTRUCTIONS)
        self.assertIn("не оценивай вероятность реального события", bot.LLM_INSTRUCTIONS)
        self.assertIn("иронию, сарказм", bot.LLM_INSTRUCTIONS)

    def test_low_confidence_suggestion_is_not_applied(self) -> None:
        analysis = bot.normalize_llm_analysis(
            {
                "decision": "increase",
                "delta": 3,
                "confidence": 0.4,
                "relevant_messages": 5,
                "summary": "Тревога растёт.",
                "factors": ["слухи"],
            },
            message_count=10,
            model="test-model",
        )

        self.assertEqual(analysis["suggested_delta"], 3)
        self.assertEqual(analysis["delta"], 0)

    def test_llm_replaces_keyword_signal_in_same_window(self) -> None:
        chat = bot.new_chat()
        bot.record_observation(chat["signals"], 1, 2, NOW)
        bot.record_observation(chat["signals"], 2, 2, NOW)
        analysis = bot.normalize_llm_analysis(
            {
                "decision": "decrease",
                "delta": -2,
                "confidence": 0.9,
                "relevant_messages": 4,
                "summary": "Слух разобрали и сочли рофлом.",
                "factors": ["опровержение", "ирония"],
            },
            message_count=6,
            model="test-model",
        )

        result = bot.settle_chat(chat, NOW, analysis)

        self.assertEqual(result["llm_delta"], -2)
        self.assertEqual(result["message_delta"], 0)
        self.assertEqual(result["new"], bot.DEFAULT_VALUE - 2)
        self.assertFalse(chat["signals"])

    def test_api_error_uses_local_fallback(self) -> None:
        chat = bot.new_chat()
        bot.record_observation(chat["signals"], 1, 2, NOW)
        bot.record_observation(chat["signals"], 2, 2, NOW)
        analysis = bot.unavailable_llm_analysis("error", 5, "test-model")

        result = bot.settle_chat(chat, NOW, analysis)

        self.assertEqual(result["llm_delta"], 0)
        self.assertEqual(result["message_delta"], 2)
        self.assertEqual(result["new"], bot.DEFAULT_VALUE + 2)

    def test_relevant_hold_can_be_published(self) -> None:
        result = {
            "total_delta": 0,
            "llm": {"status": "analyzed", "relevant_messages": 4},
        }
        self.assertEqual(bot.should_publish_result(result), bot.LLM_POST_HOLDS)

    def test_analyzer_without_key_needs_no_openai_package(self) -> None:
        analyzer = bot.LLMAnalyzer(api_key="")
        analysis = asyncio.run(analyzer.analyze(1, []))
        self.assertEqual(analysis["status"], "disabled")

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "pydantic is not installed")
    def test_structured_request_contract(self) -> None:
        class FakeParsed:
            def model_dump(self) -> dict[str, object]:
                return {
                    "discussion_trend": "increase",
                    "discussion_delta": 2,
                    "analysis_confidence": 0.9,
                    "relevant_messages": 3,
                    "discussion_summary": "Участники стали обсуждать тему серьёзнее.",
                    "discussion_factors": ["В чате стало меньше иронии."],
                }

        class FakeResponse:
            output_parsed = FakeParsed()
            usage = type("Usage", (), {"input_tokens": 123, "output_tokens": 45})()

        class FakeResponses:
            kwargs: dict[str, object] = {}

            async def parse(self, **kwargs: object) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        chat = bot.new_chat()
        for message_id, text in enumerate(("мобка?", "похоже не рофл", "есть пруфы"), 1):
            bot.record_chat_message(chat, message_id, text, NOW, message_id)
        client = FakeClient()
        analyzer = bot.LLMAnalyzer(
            api_key="test-key", provider="openai", client=client
        )

        analysis = asyncio.run(analyzer.analyze(123456789, chat["messages"]))

        self.assertEqual(analysis["delta"], 2)
        self.assertEqual(analysis["input_tokens"], 123)
        self.assertEqual(analysis["output_tokens"], 45)
        self.assertFalse(client.responses.kwargs["store"])
        self.assertEqual(client.responses.kwargs["model"], "gpt-5.6-luna")
        self.assertNotIn("123456789", str(client.responses.kwargs["input"]))
        self.assertIn("anon-1", str(client.responses.kwargs["input"]))
        schema = client.responses.kwargs["text_format"].model_json_schema()
        self.assertIn("discussion_summary", schema["properties"])
        self.assertNotIn("summary", schema["properties"])

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "pydantic is not installed")
    def test_deepseek_request_uses_only_supported_privacy_parameters(self) -> None:
        class FakeParsed:
            def model_dump(self) -> dict[str, object]:
                return {
                    "discussion_trend": "hold",
                    "discussion_delta": 0,
                    "analysis_confidence": 0.8,
                    "relevant_messages": 3,
                    "discussion_summary": "Обсуждение не изменило общий настрой.",
                    "discussion_factors": [],
                }

        class FakeResponse:
            output_parsed = FakeParsed()
            usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()

        class FakeResponses:
            kwargs: dict[str, object] = {}

            async def parse(self, **kwargs: object) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        chat = bot.new_chat()
        for message_id in range(1, 4):
            bot.record_chat_message(chat, message_id, "обсуждаем мобку", NOW, message_id)
        client = FakeClient()
        analyzer = bot.LLMAnalyzer(
            api_key="deepseek-key", provider="deepseek", client=client
        )

        analysis = asyncio.run(analyzer.analyze(42, chat["messages"]))

        self.assertEqual(analysis["provider"], "deepseek")
        self.assertEqual(client.responses.kwargs["model"], "deepseek-v4-flash")
        self.assertIn("reasoning", client.responses.kwargs)
        self.assertNotIn("store", client.responses.kwargs)
        self.assertNotIn("safety_identifier", client.responses.kwargs)

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "pydantic is not installed")
    def test_openrouter_request_requires_schema_and_private_routing(self) -> None:
        class FakeParsed:
            def model_dump(self) -> dict[str, object]:
                return {
                    "discussion_trend": "increase",
                    "discussion_delta": 1,
                    "analysis_confidence": 0.8,
                    "relevant_messages": 3,
                    "discussion_summary": "В чате тревога немного усилилась.",
                    "discussion_factors": ["Участники обсуждают тему серьёзнее."],
                }

        class FakeResponse:
            output_parsed = FakeParsed()
            usage = type("Usage", (), {"input_tokens": 20, "output_tokens": 8})()

        class FakeResponses:
            kwargs: dict[str, object] = {}

            async def parse(self, **kwargs: object) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            responses = FakeResponses()

        chat = bot.new_chat()
        for message_id in range(1, 4):
            bot.record_chat_message(chat, message_id, "мобка уже не рофл", NOW, message_id)
        client = FakeClient()
        analyzer = bot.LLMAnalyzer(
            api_key="openrouter-key", provider="openrouter", client=client
        )

        analysis = asyncio.run(analyzer.analyze(42, chat["messages"]))

        routing = client.responses.kwargs["extra_body"]["provider"]
        self.assertEqual(analysis["provider"], "openrouter")
        self.assertEqual(
            client.responses.kwargs["model"], "deepseek/deepseek-v4-flash"
        )
        self.assertTrue(routing["require_parameters"])
        self.assertEqual(routing["data_collection"], "deny")
        self.assertNotIn("reasoning", client.responses.kwargs)


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
            self.assertEqual(json.loads(path.read_text())["version"], 3)
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

    def test_caption_marks_level_transition(self) -> None:
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

    def test_card_explains_topic_and_disclaims_prediction(self) -> None:
        with patch.object(bot, "centered_text") as centered_text:
            bot.render_clock(42)

        card_text = [call.args[2] for call in centered_text.call_args_list]
        self.assertIn("Настроение чата по теме мобилизации", card_text)
        self.assertIn(bot.PUBLIC_DISCLAIMER, card_text)

    def test_llm_caption_labels_discussion_analysis(self) -> None:
        result = {
            "old": 50,
            "new": 51,
            "total_delta": 1,
            "vote_delta": 0,
            "message_delta": 0,
            "llm_delta": 1,
            "votes": 0,
            "signals": 0,
            "llm": {
                "status": "analyzed",
                "confidence": 0.8,
                "summary": "Участники чаще обсуждают слухи.",
                "factors": ["В чате усилилась тревога."],
                "relevant_messages": 3,
                "message_count": 5,
            },
        }

        caption = bot.review_caption(result)

        self.assertIn("уверенность разбора 80%", caption)
        self.assertIn("Почему чат изменился:", caption)

    def test_card_has_quick_vote_buttons(self) -> None:
        self.assertEqual(
            [data for _, data in bot.QUICK_VOTE_BUTTONS],
            ["shake:up", "shake:down"],
        )

    def test_telegram_caption_is_bounded(self) -> None:
        caption = bot.telegram_caption("я" * 2000)
        self.assertEqual(len(caption), bot.TELEGRAM_CAPTION_LIMIT)
        self.assertTrue(caption.endswith("…"))


if __name__ == "__main__":
    unittest.main()
