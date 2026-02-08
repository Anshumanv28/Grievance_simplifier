# Grievance Simplification

Pipeline to translate English grievance texts to Hindi (Bhashini), run POS analysis (Stanza), and identify non-colloquial Hindi words with simplified alternatives (Gemini).

## Setup

### 1. Clone and enter the repo

```bash
cd Grievance_simplification
```

### 2. Create a virtual environment (optional but recommended)

With **uv**:

```bash
uv venv "grievance venv"
# Activate (Windows PowerShell):
.\grievance venv\Scripts\Activate.ps1
# Activate (Windows cmd): .\grievance venv\Scripts\activate.bat
# Activate (Linux/macOS): source grievance\ venv/bin/activate
```

With **venv**:

```bash
python -m venv "grievance venv"
# Then activate as above for your OS
```

### 3. Install dependencies

With **uv**:

```bash
uv pip install -r requirements.txt
# Or with lock file: uv pip sync requirements.lock
```

With **pip**:

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example env file and add your keys:

```bash
copy .env.example .env   # Windows
# cp .env.example .env  # Linux/macOS
```

Edit `.env` and set:

- `BHASHINI_API_KEY` – from Bhashini (for translation)
- `GEMINI_API_KEY` – Google Gemini (for non-colloquial word analysis)

### 5. Input data

Place your input CSV in the project root with a column named `grievance_text` (or pass `--text-column`). Default input file: `final_clean.csv`.

---

## Scripts to run

### Full pipeline

Runs all four stages: translation → POS → Gemini → final output.

```bash
python run_pipeline.py
```

- **Input:** `final_clean.csv` (or `--input path/to/file.csv`)
- **Outputs:**  
  `translation_output.csv`, `pos_output.csv`, `gemini_output.csv`, `final_output.csv`, plus checkpoint and JSON intermediates.

**Useful options:**

| Option | Description |
|--------|-------------|
| `--translation-only` | Run only Stage 1 (translation), then exit |
| `--start-row N` | Start translation from row N (1-based); earlier rows from checkpoint |
| `--end-row N` | Stop translation after row N (1-based, inclusive) |
| `--skip-translation` | Use existing checkpoint; run POS, Gemini, output only |
| `--skip-pos` | Load POS from existing `pos_words.json` |
| `--skip-gemini` | Load Gemini results from existing `gemini_words.json` |

**Examples:**

```bash
# Translation only, rows 66001 to end
python run_pipeline.py --start-row 66001 --translation-only

# Translation only, rows 66001 to 70000
python run_pipeline.py --start-row 66001 --end-row 70000 --translation-only

# Full pipeline but skip translation (use existing checkpoint)
python run_pipeline.py --skip-translation

# Custom input and output paths
python run_pipeline.py --input my_data.csv --output my_final.csv
```

### Tests

```bash
python -m pytest tests/ -v
# Or with unittest:
python -m unittest tests.test_pipeline -v
```

---

## Pipeline stages

| Stage | Script / module | Output |
|-------|------------------|--------|
| 1. Translation | `run_pipeline.py` → `translation.py` | `translation_checkpoint.csv`, `translation_output.csv` |
| 2. POS | `run_pipeline.py` → `pos_analysis.py` | `pos_words.json`, `pos_output.csv` |
| 3. Gemini | `run_pipeline.py` → `gemini_analysis.py` | `gemini_words.json`, `gemini_output.csv` |
| 4. Final output | `run_pipeline.py` → `output_builder.py` | `final_output.csv`, `summary.json` |

Progress and checkpoints:

- Translation saves a checkpoint every 1000 rows; re-run the same command to resume after a crash.
- Rows that already have a translation in the checkpoint are skipped.

Logs go to the terminal and to `logs/pipeline.log`.
