"""
Run the full grievance pipeline: translation (Bhashini) → POS (Stanza) → Gemini → final output.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from logger_utils import setup_logging, stage_timer
from translation import BhashiniTranslator
from pos_analysis import StanzaPOSAnalyzer
from gemini_analysis import GeminiAnalyzer
from output_builder import create_final_output

# Default paths
DEFAULT_INPUT_CSV = "final_clean.csv"
DEFAULT_TEXT_COLUMN = "grievance_text"
DEFAULT_CHECKPOINT = "translation_checkpoint.csv"
DEFAULT_TRANSLATION_CSV = "translation_output.csv"
DEFAULT_POS_CSV = "pos_output.csv"
DEFAULT_GEMINI_CSV = "gemini_output.csv"
DEFAULT_POS_JSON = "pos_words.json"
DEFAULT_GEMINI_JSON = "gemini_words.json"
DEFAULT_OUTPUT_CSV = "final_output.csv"
DEFAULT_SUMMARY_JSON = "summary.json"
LOG_DIR = "logs"
SAMPLE_SIZE = 3
MAX_SAMPLE_LEN = 80


def validate_input(df: pd.DataFrame, text_column: str) -> None:
    """Ensure required column exists and has valid rows."""
    if text_column not in df.columns:
        raise ValueError(f"Missing column: {text_column}. Columns: {list(df.columns)}")
    non_empty = df[text_column].astype(str).str.strip().ne("")
    if non_empty.sum() == 0:
        raise ValueError("No non-empty text in input.")
    logging.getLogger(__name__).info(
        "Input validated: %s rows, %s non-empty",
        len(df),
        non_empty.sum(),
    )


def _slice_df_rows(df: pd.DataFrame, start_row: Optional[int], end_row: Optional[int]):
    """Return df sliced to 1-based start_row..end_row (inclusive)."""
    if start_row is None and end_row is None:
        return df
    start_idx = (start_row - 1) if start_row is not None else 0
    end_idx = end_row if end_row is not None else len(df)
    return df.iloc[start_idx:end_idx]


def _slice_word_dict(word_dict: dict, start_row: Optional[int], end_row: Optional[int]) -> dict:
    """Return word_dict with only items in 1-based start_row..end_row (by sorted key order)."""
    if start_row is None and end_row is None:
        return word_dict
    items = sorted(word_dict.items())
    start_idx = (start_row - 1) if start_row is not None else 0
    end_idx = end_row if end_row is not None else len(items)
    return dict(items[start_idx:end_idx])


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Grievance pipeline: translate → POS → Gemini → output")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Text column name")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Translation checkpoint CSV")
    parser.add_argument("--translation-csv", default=DEFAULT_TRANSLATION_CSV, help="Stage 1 output CSV")
    parser.add_argument("--pos-csv", default=DEFAULT_POS_CSV, help="Stage 2 output CSV")
    parser.add_argument("--gemini-csv", default=DEFAULT_GEMINI_CSV, help="Stage 3 output CSV")
    parser.add_argument("--pos-json", default=DEFAULT_POS_JSON, help="POS word list JSON")
    parser.add_argument("--gemini-json", default=DEFAULT_GEMINI_JSON, help="Gemini results JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Final output CSV")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_JSON, help="Summary stats JSON")
    parser.add_argument("--skip-translation", action="store_true", help="Use existing checkpoint only")
    parser.add_argument("--start-row", type=int, default=None, metavar="N", help="Start translation from row N (1-based)")
    parser.add_argument("--end-row", type=int, default=None, metavar="N", help="Stop translation after row N (1-based, inclusive)")
    parser.add_argument("--translation-only", action="store_true", help="Run only translation (Stage 1), then exit")
    parser.add_argument("--pos-only", action="store_true", help="Run only POS (Stage 2), then exit; uses checkpoint")
    parser.add_argument("--pos-start-row", type=int, default=None, metavar="N", help="POS: use rows from N (1-based)")
    parser.add_argument("--pos-end-row", type=int, default=None, metavar="N", help="POS: use rows up to N (1-based, inclusive)")
    parser.add_argument("--gemini-only", action="store_true", help="Run only Gemini (Stage 3), then exit; uses pos_words.json")
    parser.add_argument("--gemini-start-row", type=int, default=None, metavar="N", help="Gemini: use words from row N (1-based, by sorted order)")
    parser.add_argument("--gemini-end-row", type=int, default=None, metavar="N", help="Gemini: use words up to row N (1-based, inclusive)")
    parser.add_argument("--skip-pos", action="store_true", help="Load POS from --pos-json if present")
    parser.add_argument("--skip-gemini", action="store_true", help="Load Gemini from --gemini-json if present")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Log directory")
    args = parser.parse_args()

    if args.translation_only:
        args.skip_pos = True
        args.skip_gemini = True
    if args.pos_only:
        args.skip_translation = True
        args.skip_gemini = True
    if args.gemini_only:
        args.skip_translation = True
        args.skip_pos = True

    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)

    bhashini_key = os.getenv("BHASHINI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not bhashini_key and not args.skip_translation:
        logger.error("BHASHINI_API_KEY not set. Set in .env or skip translation with --skip-translation")
        sys.exit(1)
    if not gemini_key and not args.skip_gemini and not args.translation_only and not args.pos_only:
        logger.error("GEMINI_API_KEY not set. Set in .env or skip with --skip-gemini")
        sys.exit(1)

    pipeline_start = time.monotonic()
    hindi_column = "hindi_translation"

    # --- POS-only: load checkpoint, slice, run POS, exit ---
    if args.pos_only:
        logger.info("POS-only run: loading checkpoint %s", args.checkpoint)
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError("Checkpoint not found. Run translation first.")
        df = pd.read_csv(args.checkpoint)
        if hindi_column not in df.columns:
            raise ValueError("Checkpoint has no hindi_translation column.")
        df = _slice_df_rows(df, args.pos_start_row, args.pos_end_row)
        logger.info("POS: using %s rows (pos_start_row=%s, pos_end_row=%s)", len(df), args.pos_start_row, args.pos_end_row)
        with stage_timer(logger, "Stage 2: POS extraction"):
            analyzer = StanzaPOSAnalyzer()
            word_dict = analyzer.extract_words(
                df[hindi_column].astype(str).tolist(),
                persist_path=args.pos_json,
            )
        pos_rows = [
            {"lemma": k, "pos": v.get("pos", "X"), "word_forms": "|".join(v.get("word_forms", []))}
            for k, v in word_dict.items()
        ]
        pd.DataFrame(pos_rows).to_csv(args.pos_csv, index=False)
        logger.info("Wrote Stage 2 output to %s", args.pos_csv)
        for i, (lemma, data) in enumerate(list(word_dict.items())[:SAMPLE_SIZE]):
            logger.info(
                "[Stage 2 sample] lemma: \"%s\", pos: \"%s\", word_forms: %s",
                lemma[:MAX_SAMPLE_LEN],
                data.get("pos", "X"),
                data.get("word_forms", [])[:5],
            )
        logger.info("Pipeline completed (POS only) in %.2fs total.", time.monotonic() - pipeline_start)
        return

    # --- Gemini-only: load pos_words.json, slice, run Gemini, exit ---
    if args.gemini_only:
        logger.info("Gemini-only run: loading %s", args.pos_json)
        if not Path(args.pos_json).exists():
            raise FileNotFoundError("pos_words.json not found. Run POS first.")
        with open(args.pos_json, encoding="utf-8") as f:
            raw = json.load(f)
        word_dict = {
            k: {"pos": v.get("pos", "X"), "word_forms": v.get("word_forms", [])}
            for k, v in raw.items()
        }
        word_dict = _slice_word_dict(word_dict, args.gemini_start_row, args.gemini_end_row)
        logger.info("Gemini: using %s words (gemini_start_row=%s, gemini_end_row=%s)", len(word_dict), args.gemini_start_row, args.gemini_end_row)
        with stage_timer(logger, "Stage 3: Gemini analysis"):
            gemini = GeminiAnalyzer(api_key=gemini_key)
            gemini_result = gemini.identify_non_colloquial(
                word_dict,
                persist_path=args.gemini_json,
            )
        gemini_rows = [
            {"word": w, "is_non_colloquial": d.get("is_non_colloquial", False), "simplified": d.get("simplified", w), "pos": d.get("pos", "X")}
            for w, d in gemini_result.items()
        ]
        pd.DataFrame(gemini_rows).to_csv(args.gemini_csv, index=False)
        logger.info("Wrote Stage 3 output to %s", args.gemini_csv)
        for i, (w, d) in enumerate(list(gemini_result.items())[:SAMPLE_SIZE]):
            logger.info(
                "[Stage 3 sample] word: \"%s\", is_non_colloquial: %s, simplified: \"%s\"",
                w[:MAX_SAMPLE_LEN],
                d.get("is_non_colloquial"),
                (d.get("simplified") or w)[:MAX_SAMPLE_LEN],
            )
        logger.info("Pipeline completed (Gemini only) in %.2fs total.", time.monotonic() - pipeline_start)
        return

    # --- Full pipeline (or translation-only) ---
    logger.info("Loading input: %s", args.input)
    df = pd.read_csv(args.input)
    validate_input(df, args.text_column)
    logger.info(
        "Pipeline started at %s. Input: %s rows.",
        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        len(df),
    )

    # Step 1: Translation
    with stage_timer(logger, "Stage 1: Translation"):
        if not args.skip_translation and bhashini_key:
            translator = BhashiniTranslator(api_key=bhashini_key)
            df = translator.translate_all(
                df,
                text_column=args.text_column,
                checkpoint_file=args.checkpoint,
                output_column=hindi_column,
                start_row=args.start_row,
                end_row=args.end_row,
            )
        else:
            if Path(args.checkpoint).exists():
                df = pd.read_csv(args.checkpoint)
                logger.info("Loaded checkpoint: %s", args.checkpoint)
            else:
                raise FileNotFoundError("No checkpoint and --skip-translation: run translation first.")
        if hindi_column not in df.columns:
            raise ValueError("No hindi_translation column; run translation step.")
        stage1_df = df[[args.text_column, hindi_column]].copy()
        stage1_df.columns = ["grievance_text", "hindi_translation"]
        stage1_df.to_csv(args.translation_csv, index=False)
        logger.info("Wrote Stage 1 output to %s", args.translation_csv)
        non_empty = stage1_df[hindi_column].astype(str).str.strip() != ""
        for _, row in stage1_df.loc[non_empty].head(SAMPLE_SIZE).iterrows():
            logger.info(
                "[Stage 1 sample] english: \"%s\", hindi: \"%s\"",
                (row["grievance_text"] or "")[:MAX_SAMPLE_LEN],
                (row["hindi_translation"] or "")[:MAX_SAMPLE_LEN],
            )

    if args.translation_only:
        logger.info("Pipeline completed (translation only) in %.2fs total.", time.monotonic() - pipeline_start)
        return

    # Apply POS row range for full pipeline
    df = _slice_df_rows(df, args.pos_start_row, args.pos_end_row)
    if args.pos_start_row is not None or args.pos_end_row is not None:
        logger.info("POS: using %s rows (pos_start_row=%s, pos_end_row=%s)", len(df), args.pos_start_row, args.pos_end_row)

    # Step 2: POS
    with stage_timer(logger, "Stage 2: POS extraction"):
        if not args.skip_pos:
            analyzer = StanzaPOSAnalyzer()
            word_dict = analyzer.extract_words(
                df[hindi_column].astype(str).tolist(),
                persist_path=args.pos_json,
            )
        else:
            if Path(args.pos_json).exists():
                with open(args.pos_json, encoding="utf-8") as f:
                    raw = json.load(f)
                word_dict = {
                    k: {"pos": v.get("pos", "X"), "word_forms": v.get("word_forms", [])}
                    for k, v in raw.items()
                }
                logger.info("Loaded POS from %s: %s words", args.pos_json, len(word_dict))
            else:
                raise FileNotFoundError("No --pos-json and --skip-pos: run POS step first.")
        pos_rows = [
            {"lemma": k, "pos": v.get("pos", "X"), "word_forms": "|".join(v.get("word_forms", []))}
            for k, v in word_dict.items()
        ]
        pd.DataFrame(pos_rows).to_csv(args.pos_csv, index=False)
        logger.info("Wrote Stage 2 output to %s", args.pos_csv)
        for i, (lemma, data) in enumerate(list(word_dict.items())[:SAMPLE_SIZE]):
            logger.info(
                "[Stage 2 sample] lemma: \"%s\", pos: \"%s\", word_forms: %s",
                lemma[:MAX_SAMPLE_LEN],
                data.get("pos", "X"),
                data.get("word_forms", [])[:5],
            )

    # Step 3: Gemini (optionally on a slice of the word list)
    word_dict_gemini = _slice_word_dict(word_dict, args.gemini_start_row, args.gemini_end_row)
    if args.gemini_start_row is not None or args.gemini_end_row is not None:
        logger.info("Gemini: using %s words (gemini_start_row=%s, gemini_end_row=%s)", len(word_dict_gemini), args.gemini_start_row, args.gemini_end_row)
    with stage_timer(logger, "Stage 3: Gemini analysis"):
        if not args.skip_gemini and gemini_key:
            gemini = GeminiAnalyzer(api_key=gemini_key)
            gemini_result = gemini.identify_non_colloquial(
                word_dict_gemini,
                persist_path=args.gemini_json,
            )
        else:
            if Path(args.gemini_json).exists():
                with open(args.gemini_json, encoding="utf-8") as f:
                    gemini_result = json.load(f)
                logger.info("Loaded Gemini results from %s", args.gemini_json)
            else:
                raise FileNotFoundError("No --gemini-json and --skip-gemini: run Gemini step first.")
        gemini_rows = [
            {"word": w, "is_non_colloquial": d.get("is_non_colloquial", False), "simplified": d.get("simplified", w), "pos": d.get("pos", "X")}
            for w, d in gemini_result.items()
        ]
        pd.DataFrame(gemini_rows).to_csv(args.gemini_csv, index=False)
        logger.info("Wrote Stage 3 output to %s", args.gemini_csv)
        for i, (w, d) in enumerate(list(gemini_result.items())[:SAMPLE_SIZE]):
            logger.info(
                "[Stage 3 sample] word: \"%s\", is_non_colloquial: %s, simplified: \"%s\"",
                w[:MAX_SAMPLE_LEN],
                d.get("is_non_colloquial"),
                (d.get("simplified") or w)[:MAX_SAMPLE_LEN],
            )

    # Step 4: Final output
    with stage_timer(logger, "Stage 4: Output generation"):
        final_df = create_final_output(
            df,
            text_column=args.text_column,
            hindi_column=hindi_column,
            word_dict=word_dict,
            gemini_result=gemini_result,
            summary_path=args.summary,
        )
        final_df.to_csv(args.output, index=False)
        logger.info("Wrote %s rows to %s", len(final_df), args.output)
        for i in range(min(SAMPLE_SIZE, len(final_df))):
            row = final_df.iloc[i]
            logger.info(
                "[Stage 4 sample] original_english_text=%s, non_colloquial_hindi_word=%s, simplified_hindi_alternative=%s, pos_tag=%s, full_hindi_translation=%s",
                (str(row.get("original_english_text", "")) or "")[:MAX_SAMPLE_LEN],
                (str(row.get("non_colloquial_hindi_word", "")) or "")[:MAX_SAMPLE_LEN],
                (str(row.get("simplified_hindi_alternative", "")) or "")[:MAX_SAMPLE_LEN],
                str(row.get("pos_tag", "")),
                (str(row.get("full_hindi_translation", "")) or "")[:MAX_SAMPLE_LEN],
            )

    logger.info("Pipeline completed in %.2fs total.", time.monotonic() - pipeline_start)


if __name__ == "__main__":
    main()
