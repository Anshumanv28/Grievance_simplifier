"""
Stanza-based POS tagging and lemma extraction for Hindi text.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import stanza
from tqdm import tqdm

try:
    import torch
    _use_gpu = torch.cuda.is_available()
except Exception:
    _use_gpu = False

logger = logging.getLogger(__name__)

# Stanza Hindi model
LANG = "hi"
PROCESSORS = "tokenize,pos,lemma"


class StanzaPOSAnalyzer:
    """Extract unique words with POS tags and lemmas from Hindi text using Stanza."""

    def __init__(self, lang: str = LANG, processors: str = PROCESSORS):
        self.lang = lang
        self.processors = processors
        self._pipeline: Optional[stanza.Pipeline] = None

    @property
    def nlp(self) -> stanza.Pipeline:
        if self._pipeline is None:
            logger.info("Downloading Stanza Hindi model if needed...")
            stanza.download(self.lang)
            logger.info("Building Stanza pipeline (use_gpu=%s)...", _use_gpu)
            self._pipeline = stanza.Pipeline(
                self.lang,
                processors=self.processors,
                use_gpu=_use_gpu,
            )
        return self._pipeline

    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Return list of token info: [{word, lemma, upos}, ...].
        Skips empty or whitespace-only text.
        """
        if text is None or (isinstance(text, float) and text != text):  # NaN
            text = ""
        else:
            text = str(text).strip()
        if not text:
            return []
        try:
            doc = self.nlp(text)
            out = []
            for sent in doc.sentences:
                for word in sent.words:
                    out.append({
                        "word": word.text,
                        "lemma": word.lemma or word.text,
                        "upos": word.upos or "X",
                    })
            return out
        except Exception as e:
            logger.debug("Stanza failed for text %r: %s", text[:50], e)
            return []

    def extract_words(
        self,
        texts: List[str],
        persist_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract unique words with POS and lemma from all texts.
        Keys are lemmas (or word if lemma missing); value has pos_list (list of POS seen),
        word_forms (set of surface forms), and a canonical pos (most frequent).
        """
        t_start = time.monotonic()
        logger.info("POS extraction started (texts=%s)", len(texts))
        unique: Dict[str, Dict[str, Any]] = {}
        for text in tqdm(texts, desc="POS", unit="text"):
            tokens = self.process_text(text)
            for t in tokens:
                lemma = (t.get("lemma") or t.get("word") or "").strip()
                if not lemma:
                    continue
                word = (t.get("word") or "").strip()
                upos = (t.get("upos") or "X").strip()
                if lemma not in unique:
                    unique[lemma] = {
                        "pos_list": [],
                        "word_forms": set(),
                        "pos": upos,
                    }
                unique[lemma]["word_forms"].add(word)
                unique[lemma]["pos_list"].append(upos)
        # Set canonical pos as most frequent
        for lemma, data in unique.items():
            pos_list = data["pos_list"]
            data["pos"] = max(set(pos_list), key=pos_list.count) if pos_list else "X"
            data["word_forms"] = list(data["word_forms"])
            del data["pos_list"]
        if persist_path:
            path = Path(persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # JSON-serializable
            out = {
                k: {
                    "pos": v["pos"],
                    "word_forms": v["word_forms"],
                }
                for k, v in unique.items()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info("POS word list saved to %s (%s unique)", persist_path, len(unique))
        elapsed = time.monotonic() - t_start
        logger.info("POS extraction completed (unique_words=%s, elapsed=%.2fs)", len(unique), elapsed)
        return unique
