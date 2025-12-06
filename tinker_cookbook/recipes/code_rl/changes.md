# Changes Log

## 2025-12-05: Add Claude Code CLI for Code Quality Grading

**Files:**
- `claude_code_qual.py` (new)
- `claude_cli_code_env.py` (new)
- `claude_train.py` (new)

### Changes:

1. **`claude_code_qual.py`** - Core grading module using Claude Code CLI:
   - `build_claude_prompt(code)` - Creates prompt asking Claude to grade code quality (0.0-1.0)
   - `call_claude_cli(prompt)` - Calls `claude -p "<prompt>" --output-format text`
   - `parse_score_from_response(raw)` - Robust JSON parsing with fallbacks
   - `grade_code_with_claude(code)` - Main function returning clamped score

2. **`claude_cli_code_env.py`** - Environment with Claude code quality grading:
   - `CodeEnv_Claude` - Extends `ProblemEnv` with `code_qual_weight` parameter
   - `check_sandbox_correctness()` returns `tuple[bool, str | None]` (consistent API)
   - `step()` includes `log_summary()` and collapsible sections
   - `DeepcoderDataset_Claude` and `DeepcoderDatasetBuilder_Claude` with `code_qual_weight`

3. **`claude_train.py`** - Training CLI entry point:
   - Uses `DeepcoderDatasetBuilder_Claude`
   - Adds `code_qual_weight` CLI parameter (default 0.1)
   - Default model: `Qwen/Qwen3-4B-Instruct-2507`

### Motivation:
- Provide alternative to Gemini for code quality grading
- Use Claude Code CLI which may already be installed/configured

## 2025-12-05: Fix Gemini Code Quality Grading Bugs

**Files:**
- `gemini_code_qual.py`
- `gemini_cli_code_env.py`

### Changes:

1. **`gemini_code_qual.py`**:
   - Fixed unreachable code: removed `check=True` from subprocess.run (was conflicting with manual returncode check)
   - Removed debug print statement
   - Simplified score clamping to `max(0.0, min(1.0, score))`

2. **`gemini_cli_code_env.py`**:
   - Fixed `check_sandbox_correctness()` to return `tuple[bool, str | None]` (was returning just `bool`)
   - Added `log_summary()` call for summary banner at top of HTML logs
   - Added collapsible sections for Problem, Model Response, Submitted Code, Results
   - Now passes `question` to `grade_code_with_gemini()` for context-aware grading

### Motivation:
- Fix bugs that would cause runtime errors or unreachable code
- Consistent API across CodeEnv variants
- Better HTML log readability

## 2025-12-05: Add Summary Banner to CodeEnv Rollout Logs

**File:** `code_env.py`

### Changes:
1. **Added `log_summary()` call in `step()`** - Now logs a summary banner with Correct, Format, and Reward metrics at the top of each rollout log. The CSS ordering in logtree automatically positions `.lt-summary` elements at the top of the page (after title/subtitle).

### Motivation:
- Provides quick at-a-glance results without scrolling through the problem/response/code sections

## 2025-12-05: Dark Mode and Summary Banner for HTML Logs

**Files:**
- `tinker_cookbook/utils/logtree.py`
- `tinker_cookbook/utils/logtree_formatters.py`
- `tinker_cookbook/rl/train.py`

### Changes:

1. **Dark Mode Support** - HTML logs now support dark mode:
   - Automatically respects system preference (`prefers-color-scheme: dark`)
   - Added theme toggle button (sun/moon icon) in top-right corner
   - Three-state toggle: auto → dark → light → auto
   - User preference persisted to localStorage
   - All colors defined as CSS custom properties for consistent theming
   - Updated conversation formatter colors for dark mode readability

2. **Summary Banner** - New `log_summary()` API in logtree:
   - Displays key metrics in a card-based banner
   - Progress bars with color coding (green/yellow/red based on thresholds)
   - Automatically appears at top of page (after title) via CSS ordering
   - Shows: Pass Rate, Format Rate, Mean Reward, All Good, All Bad, Mixed

3. **Training Integration** - Extended logtree scope in `do_sync_training`:
   - Logtree scope now includes the train step (was previously outside)
   - Calls `log_summary()` after metrics are computed
   - Summary banner added to all training iteration HTML logs

### CSS Custom Properties Added:
- Theme colors: `--lt-bg`, `--lt-text`, `--lt-card`, `--lt-accent`, `--lt-border`, `--lt-sub`
- Message role colors: `--lt-user-*`, `--lt-assistant-*`, `--lt-system-*`, `--lt-tool-*`
- Summary colors: `--lt-success`, `--lt-warning`, `--lt-danger`, `--lt-progress-bg`

### Motivation:
- Dark mode reduces eye strain when reviewing logs in low-light environments
- Summary banner provides quick at-a-glance performance metrics without scrolling through details
- Consistent theming infrastructure makes future styling changes easier

## 2025-12-05: Wandb Run Resumption Support

**File:** `tinker_cookbook/utils/ml_log.py`

### Changes:
1. **Modified `WandbLogger.__init__()`** - Now automatically saves and resumes wandb runs:
   - On first run, saves the wandb run ID to `{log_dir}/wandb_run_id.txt`
   - On subsequent runs, reads the saved run ID and passes `resume="must"` to `wandb.init()`
   - This ensures restarted training runs continue logging to the same wandb run

### Motivation:
- When training runs are restarted (e.g., after a crash or manual stop), metrics should continue in the same wandb run rather than creating a new one
- This makes it easier to track the full training history in a single wandb dashboard

## 2025-12-04: Improved HTML Eval Logs

**File:** `code_env.py`

### Changes:
1. **Modified `check_sandbox_correctness()`** - Now returns a tuple `(bool, str | None)` containing both the success status and the extracted code, so it can be logged separately.

2. **Improved `step()` logging** - Restructured the logtree output with clear sections:
   - **Problem**: Collapsible section showing the problem statement
   - **Model Response**: Collapsible section showing the full model output
   - **Submitted Code**: Collapsible section showing just the extracted code that was sent to the sandbox
   - **Results**: Summary with format validity, correctness, and reward

### Motivation:
- Make it easier to distinguish between problem and response in HTML logs
- Allow quick inspection of the actual code being evaluated without wading through the full response
- Use collapsible sections to keep logs manageable while preserving full details
