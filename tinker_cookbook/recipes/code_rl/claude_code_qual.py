# claude_code_qual.py
"""Code quality grading using Claude Code CLI."""

import json
import re
import subprocess
import textwrap


def build_claude_prompt(code: str) -> str:
    """
    Build a strict grading prompt for Claude.
    Forces response to be pure JSON for easy parsing.
    """
    return textwrap.dedent(f"""
    You are a strict code-quality grader.

    You will be given a code snippet. Your task is to grade how well the code
    is designed on a continuous scale from 0.0 to 1.0.

    Score criteria (equally weighted):
    - Readability (clear structure, meaningful names, comments/docstrings)
    - Modularity (functions, separation of concerns, avoid duplication)
    - Robustness (basic error handling, sanity checks when appropriate)
    - Maintainability (easy to extend, minimal hard-coding, avoids hacks)

    Output format requirements:
    - Respond with *only* a single JSON object.
    - The JSON must be exactly of the form:
      {{
        "score": <float between 0.0 and 1.0>
      }}
    - Do not include any explanations, markdown, or extra keys.

    Code to grade:
    ```python
    {code}
    ```
    """).strip()


def call_claude_cli(prompt: str) -> str:
    """
    Call the Claude Code CLI with the given prompt and return stdout as a string.

    Uses `claude -p "prompt" --output-format text` for non-interactive single-shot mode.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "text",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI failed.\n"
            f"Return code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def parse_score_from_response(raw: str) -> float:
    """
    Parse the score from Claude's response.
    Handles both clean JSON and JSON embedded in text.
    """
    # First try direct JSON parse
    try:
        data = json.loads(raw)
        if "score" in data:
            return float(data["score"])
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from response
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "score" in data:
                return float(data["score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Try to find a bare float that looks like a score
    score_match = re.search(r"\"score\"\s*:\s*([\d.]+)", raw)
    if score_match:
        return float(score_match.group(1))

    raise RuntimeError(f"Could not parse score from Claude response:\n{raw}")


def grade_code_with_claude(code: str) -> float:
    """
    Builds the grading prompt, calls Claude via CLI, parses the response,
    and returns the score float in [0, 1].
    """
    prompt = build_claude_prompt(code)
    raw = call_claude_cli(prompt)

    score = parse_score_from_response(raw)

    # Clamp score to valid range
    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    # Test to see if this works
    code_snippet = textwrap.dedent("""
    def fibonacci(n):
        if n == 0:
            return n
        elif n == 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """)

    print("Grading code quality with Claude CLI...")
    code_qual_score = grade_code_with_claude(code_snippet)
    print("Finished grading")

    print("")
    print("-" * 80)
    print(f"Code quality score from Claude CLI: {code_qual_score}")
    print("-" * 80)
