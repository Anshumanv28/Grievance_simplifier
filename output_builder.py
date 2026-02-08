"""
Assemble final CSV and summary statistics from translation, POS, and Gemini results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Final CSV columns
COL_ENGLISH = "original_english_text"
COL_NON_COLLOQUIAL = "non_colloquial_hindi_word"
COL_SIMPLIFIED = "simplified_hindi_alternative"
COL_POS = "pos_tag"
COL_HINDI_REF = "full_hindi_translation"


def _word_appears_in_text(
    text: str,
    lemma: str,
    word_forms: List[str],
) -> bool:
    """Return True if lemma or any word form appears in text (substring or whole-word)."""
    if not (text or "").strip():
        return False
    text = " " + (text or "").strip() + " "
    if lemma and lemma in text:
        return True
    for w in word_forms or []:
        if w and w in text:
            return True
    return False


def create_final_output(
    df: pd.DataFrame,
    text_column: str,
    hindi_column: str,
    word_dict: Dict[str, Dict[str, Any]],
    gemini_result: Dict[str, Dict[str, Any]],
    summary_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build final CSV: one row per (grievance row, non-colloquial word) with columns
    original_english_text, non_colloquial_hindi_word, simplified_hindi_alternative, pos_tag, full_hindi_translation.
    """
    non_colloquial_lemmas = {
        w for w, d in gemini_result.items()
        if d.get("is_non_colloquial") is True
    }
    logger.info(
        "Building final output (rows=%s, non_colloquial_words=%s)",
        len(df),
        len(non_colloquial_lemmas),
    )
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        english = row.get(text_column) or ""
        hindi = row.get(hindi_column) or ""
        if not hindi and not english:
            continue
        for lemma in non_colloquial_lemmas:
            g = gemini_result.get(lemma, {})
            pos_info = word_dict.get(lemma, {})
            word_forms = pos_info.get("word_forms") or [lemma]
            if not _word_appears_in_text(hindi, lemma, word_forms):
                continue
            rows.append({
                COL_ENGLISH: english,
                COL_NON_COLLOQUIAL: lemma,
                COL_SIMPLIFIED: g.get("simplified") or lemma,
                COL_POS: g.get("pos") or pos_info.get("pos", "X"),
                COL_HINDI_REF: hindi,
            })
    final_df = pd.DataFrame(rows)
    total_words = len(word_dict)
    non_colloquial_count = len(non_colloquial_lemmas)
    summary = {
        "total_grievance_rows": len(df),
        "total_output_rows": len(final_df),
        "unique_words_analyzed": total_words,
        "non_colloquial_word_count": non_colloquial_count,
        "non_colloquial_percentage": round(100.0 * non_colloquial_count / total_words, 2) if total_words else 0,
    }
    logger.info(
        "Final output: %s rows; unique words %s; non-colloquial %s (%.2f%%)",
        len(final_df),
        total_words,
        non_colloquial_count,
        summary["non_colloquial_percentage"],
    )
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return final_df
