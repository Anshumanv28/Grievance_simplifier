"""Unit tests for translation, POS, output generation, and retry logic."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root for imports when running tests
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from output_builder import (
    COL_ENGLISH,
    COL_HINDI_REF,
    COL_NON_COLLOQUIAL,
    COL_POS,
    COL_SIMPLIFIED,
    _word_appears_in_text,
    create_final_output,
)
from translation import BhashiniTranslator, DEFAULT_BATCH_SIZE


class TestBhashiniTranslator(unittest.TestCase):
    """Tests for BhashiniTranslator."""

    def test_parse_response_valid(self):
        translator = BhashiniTranslator(api_key="test")
        data = {
            "pipelineResponse": [
                {
                    "output": [
                        {"source": "Hi", "target": "नमस्ते"},
                        {"source": "Bye", "target": "अलविदा"},
                    ]
                }
            ]
        }
        result = translator._parse_response(data, 2)
        self.assertEqual(result, ["नमस्ते", "अलविदा"])

    def test_parse_response_missing_output(self):
        translator = BhashiniTranslator(api_key="test")
        data = {"pipelineResponse": []}
        with self.assertRaises(ValueError):
            translator._parse_response(data, 1)

    def test_parse_response_wrong_length(self):
        translator = BhashiniTranslator(api_key="test")
        data = {
            "pipelineResponse": [
                {"output": [{"source": "a", "target": "b"}]}
            ]
        }
        with self.assertRaises(ValueError):
            translator._parse_response(data, 2)

    def test_build_payload_batch_size(self):
        translator = BhashiniTranslator(api_key="test", batch_size=3)
        payload = translator._build_payload(["a", "b", "c"])
        inputs = payload["inputData"]["input"]
        self.assertEqual(len(inputs), 3)
        self.assertEqual([x["source"] for x in inputs], ["a", "b", "c"])

    def test_translate_batch_empty(self):
        translator = BhashiniTranslator(api_key="test")
        result = translator.translate_batch([])
        self.assertEqual(result, [])

    @patch("translation.requests.post")
    def test_translate_batch_returns_parsed_result(self, mock_post):
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "pipelineResponse": [{"output": [{"source": "hello", "target": "नमस्ते"}]}]
        }
        t = BhashiniTranslator(api_key="k")
        out = t.translate_batch(["hello"])
        self.assertEqual(out, ["नमस्ते"])


class TestOutputBuilder(unittest.TestCase):
    """Tests for output_builder."""

    def test_word_appears_in_text_lemma(self):
        self.assertTrue(_word_appears_in_text("मैं हूँ", "मैं", ["मैं"]))
        self.assertTrue(_word_appears_in_text("  मैं  ", "मैं", []))

    def test_word_appears_in_text_form(self):
        self.assertTrue(_word_appears_in_text("किताबें", "किताब", ["किताब", "किताबें"]))

    def test_word_appears_in_text_empty(self):
        self.assertFalse(_word_appears_in_text("", "x", ["x"]))
        self.assertFalse(_word_appears_in_text("  ", "x", ["x"]))

    def test_word_appears_in_text_absent(self):
        self.assertFalse(_word_appears_in_text("मैं हूँ", "नहीं", ["नहीं"]))

    def test_create_final_output_columns_and_summary(self):
        df = pd.DataFrame({
            "grievance_text": ["please help", "another grievance"],
            "hindi_translation": ["कृपया मदद करें समस्या", "दूसरी शिकायत"],
        })
        word_dict = {
            "समस्या": {"pos": "NOUN", "word_forms": ["समस्या"]},
            "शिकायत": {"pos": "NOUN", "word_forms": ["शिकायत"]},
        }
        gemini_result = {
            "समस्या": {"is_non_colloquial": True, "simplified": "मसला", "pos": "NOUN"},
            "शिकायत": {"is_non_colloquial": True, "simplified": "शिकायत", "pos": "NOUN"},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            summary_path = f.name
        try:
            final = create_final_output(
                df,
                text_column="grievance_text",
                hindi_column="hindi_translation",
                word_dict=word_dict,
                gemini_result=gemini_result,
                summary_path=summary_path,
            )
            self.assertIn(COL_ENGLISH, final.columns)
            self.assertIn(COL_NON_COLLOQUIAL, final.columns)
            self.assertIn(COL_SIMPLIFIED, final.columns)
            self.assertIn(COL_POS, final.columns)
            self.assertIn(COL_HINDI_REF, final.columns)
            self.assertEqual(final[COL_NON_COLLOQUIAL].tolist(), ["समस्या", "शिकायत"])
            self.assertTrue(Path(summary_path).exists())
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("total_output_rows", summary)
            self.assertIn("non_colloquial_percentage", summary)
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_create_final_output_no_non_colloquial(self):
        df = pd.DataFrame({
            "grievance_text": ["help"],
            "hindi_translation": ["मदद"],
        })
        word_dict = {"मदद": {"pos": "NOUN", "word_forms": ["मदद"]}}
        gemini_result = {"मदद": {"is_non_colloquial": False, "simplified": "मदद", "pos": "NOUN"}}
        final = create_final_output(
            df, "grievance_text", "hindi_translation", word_dict, gemini_result,
        )
        self.assertEqual(len(final), 0)


class TestTranslationRetry(unittest.TestCase):
    """Test translation batch success with mocked API."""

    @patch("translation.requests.post")
    def test_translate_batch_success(self, mock_post):
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "pipelineResponse": [{"output": [{"source": "hi", "target": "नमस्ते"}]}]
        }
        t = BhashiniTranslator(api_key="k")
        out = t.translate_batch(["hi"])
        self.assertEqual(out, ["नमस्ते"])


if __name__ == "__main__":
    unittest.main()
