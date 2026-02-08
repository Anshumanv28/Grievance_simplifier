"""
Bhashini API translation module.
Batch translation with retry logic, rate limiting, and checkpoint/resume.
"""

import logging
import time
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 300 req/min = 5 req/sec => min 0.2s between requests
RATE_LIMIT_DELAY_SEC = 0.2
DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 1000


class BhashiniTranslator:
    """Translates English text to Hindi via Bhashini pipeline API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
        batch_size: int = DEFAULT_BATCH_SIZE,
        delay_sec: float = RATE_LIMIT_DELAY_SEC,
        max_retries: int = MAX_RETRIES,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.delay_sec = delay_sec
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce delay between requests to stay under 300 req/min."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay_sec:
            time.sleep(self.delay_sec - elapsed)
        self._last_request_time = time.monotonic()

    def _build_payload(self, texts: List[str]) -> dict:
        """Build pipeline request body. Empty strings are sent as-is."""
        input_list = [{"source": (t or "").strip() or ""} for t in texts]
        return {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": "en",
                            "targetLanguage": "hi",
                        },
                        "serviceId": "bhashini/iiith/nmt-all",
                        "numTranslation": "True",
                    },
                }
            ],
            "inputData": {
                "input": input_list,
                "audio": [{"audioContent": None}],
            },
        }

    def _parse_response(self, data: dict, expected_count: int) -> List[str]:
        """Extract target translations from pipeline response. Raises on malformed data."""
        try:
            responses = data.get("pipelineResponse") or []
            if not responses:
                raise ValueError("pipelineResponse missing or empty")
            out_list = (responses[0].get("output") or [])
            if len(out_list) != expected_count:
                raise ValueError(
                    f"output length {len(out_list)} != expected {expected_count}"
                )
            return [item.get("target") or "" for item in out_list]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Malformed pipeline response: {e}") from e

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a single batch. Applies rate limit and retries with exponential backoff.
        Returns list of Hindi strings in same order as input.
        """
        if not texts:
            return []
        texts = [(t or "").strip() for t in texts]
        payload = self._build_payload(texts)
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                result = self._parse_response(resp.json(), len(texts))
                return result
            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    "Bhashini request attempt %s failed: %s",
                    attempt + 1,
                    e,
                )
            except ValueError as e:
                last_error = e
                logger.warning(
                    "Bhashini parse attempt %s failed: %s",
                    attempt + 1,
                    e,
                )
            if attempt < self.max_retries - 1:
                backoff = 2 ** attempt
                time.sleep(backoff)
        raise last_error  # type: ignore

    def translate_all(
        self,
        df: pd.DataFrame,
        text_column: str,
        checkpoint_file: str,
        output_column: str = "hindi_translation",
    ) -> pd.DataFrame:
        """
        Translate all rows. Loads checkpoint if present and resumes; saves every checkpoint_interval rows.
        """
        t_start = time.monotonic()
        if output_column not in df.columns:
            df = df.copy()
            df[output_column] = ""

        # Resume from checkpoint
        start_idx = 0
        try:
            ck = pd.read_csv(checkpoint_file, nrows=1)
            if output_column in ck.columns:
                full_ck = pd.read_csv(checkpoint_file)
                if len(full_ck) > 0 and full_ck[output_column].notna().any():
                    last_filled = full_ck[output_column].notna() & (
                        full_ck[output_column].astype(str).str.strip() != ""
                    )
                    if last_filled.any():
                        start_idx = int(last_filled.idxmax()) + 1
                    else:
                        start_idx = 0
                    if start_idx > 0:
                        df = full_ck
                        logger.info("Resuming from checkpoint row %s", start_idx)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Starting from 0.", e)

        total = len(df)
        texts = df[text_column].astype(str).tolist()
        logger.info(
            "Translation started (total=%s, batch_size=%s, resume_from=%s)",
            total,
            self.batch_size,
            start_idx,
        )

        with tqdm(total=total, initial=start_idx, unit="row", desc="Translate") as pbar:
            i = start_idx
            while i < total:
                batch = texts[i : i + self.batch_size]
                try:
                    translated = self.translate_batch(batch)
                except Exception as e:
                    logger.error("Batch at index %s failed: %s", i, e)
                    raise
                for j, t in enumerate(translated):
                    if i + j < total:
                        df.iloc[i + j, df.columns.get_loc(output_column)] = t
                i += len(batch)
                pbar.update(len(batch))

                if i > 0 and i % self.checkpoint_interval == 0:
                    df.to_csv(checkpoint_file, index=False)
                    logger.info("Checkpoint saved at row %s", i)

        df.to_csv(checkpoint_file, index=False)
        elapsed = time.monotonic() - t_start
        translated_count = (df[output_column].astype(str).str.strip() != "").sum()
        logger.info(
            "Translation completed (translated=%s rows, elapsed=%.2fs)",
            translated_count,
            elapsed,
        )
        return df
