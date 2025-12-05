# gemini_grader.py
import json
import subprocess
import textwrap


def build_gemini_prompt(code: str) -> str:
    """
    Build a strict grading prompt for Gemini.
    We force it to respond with pure JSON so we can parse it easily.
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
    """)


def call_gemini_cli(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Call the Gemini CLI with the given prompt and return stdout as a string.
    """
    # make sure gemini cli has deterministic score
    cmd = [
        "gemini",
        "--model", model,
        "--prompt", prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,  # raises CalledProcessError if CLI fails
    )
    print("This is the result after calling the subprocessing: ", result)
    if result.returncode != 0:
        raise RuntimeError(
            "Gemini CLI failed.\n"
            f"Return code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()

def grade_code_with_gemini(code: str, model: str = "gemini-2.5-flash") -> float:
    """
    Builds the grading prompt, calls gemini via cli, and parses the json and returns the score float in [0,1]
    """

    prompt = build_gemini_prompt(code)
    raw = call_gemini_cli(prompt, model=model)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        
        # just gets the text from prompt that corresponds to the code quality score
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise RuntimeError(f"Gemini response not valid JSON:\n{raw}")
        data = json.loads(match.group(0))

    if "score" not in data:
        raise KeyError(f"Gemini JSON missing 'score' key: {data}")

    score = float(data["score"])

    # keeps score in range of 0 to 1
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score

if __name__ == "__main__":

    # test to see if this might work
    
    code_snippet = textwrap.dedent("""
    def fibonacci(n):
        if n == 0:
            return n
        elif n == 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """)

    # make sure to make a gemini api key and authenticate before running the following
    #  export GEMINI_API_KEY="{api key}"

    print("Grading code quality with gemini cli...")
    code_qual_score = grade_code_with_gemini(code_snippet)
    print("Finished grading")

    print("")
    print("--------------------------------------------------------------------------------")
    print(f"This is the code quality score produced by gemini cli: {code_qual_score}")
    print("--------------------------------------------------------------------------------")