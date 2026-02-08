"""
Gemini-based identification of non-colloquial Hindi words and simplified alternatives.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Batch sizes for Gemini (500-1000 words per request)
DEFAULT_WORD_BATCH_SIZE = 600
MAX_RETRIES = 3
RATE_DELAY_SEC = 1.0

PROMPT_TEMPLATE = """Analyze this list of Hindi words extracted from government grievances.
For each word:
1. Identify if it's non-colloquial/formal/technical Hindi (is_non_colloquial: true or false).
2. If non-colloquial, provide a simplified colloquial Hindi alternative in "simplified"; otherwise use the same word in "simplified".
3. Include the POS tag (e.g. NOUN, VERB, ADJ) in "pos".

Words with their POS (word -> pos): %s

Return ONLY a valid JSON object, no markdown or explanation. Format:
{"word1": {"is_non_colloquial": true/false, "simplified": "simple_word", "pos": "NOUN"}, "word2": ...}
"""


class GeminiAnalyzer:
    """Identify non-colloquial Hindi words and suggest simplified alternatives via Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
        word_batch_size: int = DEFAULT_WORD_BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        rate_delay_sec: float = RATE_DELAY_SEC,
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.word_batch_size = word_batch_size
        self.max_retries = max_retries
        self.rate_delay_sec = rate_delay_sec
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_delay_sec:
            time.sleep(self.rate_delay_sec - elapsed)
        self._last_request_time = time.monotonic()

    def _words_to_prompt_input(self, word_dict: Dict[str, Dict[str, Any]]) -> str:
        """Format word -> pos for prompt."""
        pairs = [f"{w} -> {d.get('pos', 'X')}" for w, d in word_dict.items()]
        return ", ".join(pairs)

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Extract JSON object from model output; handle markdown code blocks."""
        raw = (raw or "").strip()
        # Strip markdown code block if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
        return json.loads(raw)

    def _analyze_batch(
        self,
        word_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Call Gemini for one batch of words. Returns {word: {is_non_colloquial, simplified, pos}}."""
        prompt_input = self._words_to_prompt_input(word_dict)
        prompt = PROMPT_TEMPLATE % prompt_input
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                response = self.model.generate_content(prompt)
                text = (response.text or "").strip()
                parsed = self._parse_json_response(text)
                # Normalize keys to match input words (in case model returns different key format)
                result = {}
                for word, data in word_dict.items():
                    # Prefer exact word key, else try lemma
                    node = parsed.get(word)
                    if node is None:
                        for k, v in parsed.items():
                            if k.strip() == word or v.get("simplified") == word:
                                node = v
                                break
                    if isinstance(node, dict):
                        result[word] = {
                            "is_non_colloquial": node.get("is_non_colloquial", False),
                            "simplified": node.get("simplified") or word,
                            "pos": node.get("pos") or data.get("pos", "X"),
                        }
                    else:
                        result[word] = {
                            "is_non_colloquial": False,
                            "simplified": word,
                            "pos": data.get("pos", "X"),
                        }
                return result
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning("Gemini parse attempt %s failed: %s", attempt + 1, e)
            except Exception as e:
                last_error = e
                logger.warning("Gemini request attempt %s failed: %s", attempt + 1, e)
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        raise last_error  # type: ignore

    def identify_non_colloquial(
        self,
        word_dict: Dict[str, Dict[str, Any]],
        persist_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process unique words in batches; return full mapping word -> {is_non_colloquial, simplified, pos}.
        Optionally persist to JSON.
        """
        t_start = time.monotonic()
        logger.info(
            "Gemini analysis started (words=%s, batch_size=%s)",
            len(word_dict),
            self.word_batch_size,
        )
        items = list(word_dict.items())
        all_results: Dict[str, Dict[str, Any]] = {}
        batch_starts = list(range(0, len(items), self.word_batch_size))
        for start in tqdm(batch_starts, desc="Gemini", unit="batch"):
            batch_items = items[start : start + self.word_batch_size]
            batch_dict = dict(batch_items)
            batch_result = self._analyze_batch(batch_dict)
            all_results.update(batch_result)
        elapsed = time.monotonic() - t_start
        logger.info("Gemini analysis completed (elapsed=%.2fs)", elapsed)
        if persist_path:
            with open(persist_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info("Gemini results saved to %s", persist_path)
        return all_results
