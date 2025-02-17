#!/usr/bin/env python3
"""
Example script that analyzes a Git repository and outputs a basic README draft to stdout,
using an LLM for text generation.

Usage:
    python analyze_git_repo.py <repo_path>

Environment Variables (same as the advanced script):
    OPENAI_API_KEY: Your OpenAI API key (optional)
    GEMINI_API_KEY: Your Google PaLM (Gemini) API key (optional)
    DEEPSEEK_API_KEY: Your DeepSeek API key (optional)
"""

import os
import sys
import subprocess
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# LLM / Model Configuration
# (Identical or very similar to your original script)
# ----------------------------------------------------------------------
OPENAI_MODEL = "o1-preview"  # or your preferred model
GEMINI_MODEL = "gemini-2.0-flash-exp"
DEEPSEEK_MODEL = "deepseek-reasoner"
VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

# Try importing OpenAI
try:
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
except ImportError:
    print("[ERROR] Please install openai: pip install openai")
    client = None

# Try importing Gemini
try:
    import google.generativeai as genai
except ImportError:
    print(
        "[WARN] google-generativeai module not installed; Gemini calls will not work."
    )
    genai = None

if gemini_api_key and genai:
    genai.configure(api_key=gemini_api_key)
    gemini_model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL)
else:
    gemini_model_instance = None

# Try initializing vLLM
vllm_client = None
try:
    from vllm import LLMEngine
    from vllm.executors import UniProcExecutor
    from vllm import SamplingParams

    vllm_client = LLMEngine(VLLM_MODEL, executor_class=UniProcExecutor, log_stats=False)
    print("[INFO] Initialized local vLLM engine.")
except ImportError:
    print("[WARN] vllm module not installed; vLLM calls will not work.")
except Exception as e:
    print(f"[WARN] vLLM initialization failed: {e}")

# Set max tokens
if gemini_api_key:
    MAX_TOKENS = 900000
elif deepseek_api_key:
    MAX_TOKENS = 5000
elif openai_api_key:
    MAX_TOKENS = 5000
else:
    MAX_TOKENS = 20000


# ----------------------------------------------------------------------
# LLM Utility Functions
# ----------------------------------------------------------------------
def call_llm_system_user(
    system_prompt: str, user_prompt: str, temperature=0.2, max_tokens=2000
) -> str:
    """
    Calls the LLM with a system prompt and a user prompt using one of the configured providers.
    """
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    # 1) Try OpenAI
    if openai_api_key:
        # For 'o1-preview' style, you might only supply a user message.
        if "o1" in OPENAI_MODEL:
            messages = [
                {"role": "user", "content": combined_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return call_openai_chat_completion(
            OPENAI_MODEL,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 2) Try Gemini
    elif gemini_api_key and gemini_model_instance:
        return call_gemini(combined_prompt, temperature=temperature)

    # 3) Try DeepSeek (or fallback to openai style prompt)
    elif deepseek_api_key:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_openai_chat_completion(
            DEEPSEEK_MODEL,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 4) Local vLLM
    elif vllm_client is not None:
        return call_vllm(
            combined_prompt, temperature=temperature, max_tokens=max_tokens
        )

    else:
        print("[ERROR] No valid API key provided or no LLM client available.")
        sys.exit(1)


def call_openai_chat_completion(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = None,
) -> str:
    """
    Call the OpenAI Chat Completion endpoint.
    """
    if not client:
        print("[ERROR] OpenAI client unavailable.")
        return ""

    print("[INFO] Contacting OpenAI Chat Completion...")
    try:
        # For demonstration, these parameters might differ for your environment:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] OpenAI Chat Completion failed: {e}")
        return ""


def call_gemini(prompt: str, temperature=0.0) -> str:
    """
    Call the Google PaLM (Gemini) API.
    """
    if not gemini_model_instance:
        print("[ERROR] Gemini model is not initialized.")
        return ""
    try:
        print("[INFO] Contacting Gemini (Google PaLM) API...")
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=temperature,
            ),
            stream=False,
        )
        # Non-stream approach: response is a single object
        return response.text if response else ""
    except Exception as e:
        print(f"[ERROR] Gemini call failed: {e}")
        return ""


def call_vllm(prompt: str, temperature=0.0, max_tokens=3000) -> str:
    """
    Call a local vLLM model.
    """
    if not vllm_client:
        print("[ERROR] vLLM client unavailable.")
        return ""
    try:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        print("[INFO] Contacting local vLLM model ...")
        results = vllm_client.infer(prompt, sampling_params)
        # vLLM returns an iterator of results, each with .text
        return "".join([r.text for r in results])
    except Exception as e:
        print(f"[ERROR] vLLM inference error: {e}")
        return ""


# ----------------------------------------------------------------------
# Git Repo Analysis Helpers
# ----------------------------------------------------------------------
def get_repo_name(repo_path: str) -> str:
    """Simple heuristic: take the directory name as the repo name."""
    return os.path.basename(os.path.abspath(repo_path))


def run_git_command(repo_path: str, args: List[str]) -> str:
    """
    Run a git command in `repo_path` and return its output as a string.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() + e.stderr.strip()


def get_dates_and_authors(repo_path: str) -> Dict[str, Any]:
    """
    Retrieve earliest commit date, latest commit date, list of authors, and whether
    there's recent activity in the last 1-2 years.
    """
    info = {
        "earliest_commit_date": None,
        "latest_commit_date": None,
        "authors": [],
        "active_recently": False,
    }
    # Get earliest commit date
    log_oldest = run_git_command(
        repo_path, ["log", "--reverse", "--pretty=%cd", "--date=short", "-1"]
    )
    info["earliest_commit_date"] = log_oldest

    # Get latest commit date
    log_latest = run_git_command(
        repo_path, ["log", "-1", "--pretty=%cd", "--date=short"]
    )
    info["latest_commit_date"] = log_latest

    # Grab all authors and store them. (This can be large on big repos.)
    all_authors = run_git_command(repo_path, ["log", "--format=%an"])
    authors_set = set(all_authors.splitlines())
    info["authors"] = sorted(authors_set)

    # Check if there's a commit in the last 1-2 years
    # For simplicity, let's just check if there's a commit more recent than 365 days:
    log_recent = run_git_command(
        repo_path, ["log", f"--since=1.year.ago", "--pretty=oneline", "-1"]
    )
    info["active_recently"] = bool(log_recent)

    return info


def guess_build_run_instructions(repo_path: str) -> str:
    """
    A naive guess for how the project might be built or run, based on file inspection.
    """
    # If there's a requirements.txt or a setup.py or pyproject.toml => Python?
    # If there's a package.json => Node?
    # If there's a Makefile => maybe `make build`?
    # Etc. We'll just do a quick check for some known files.
    known_files = os.listdir(repo_path)
    instructions = []

    if "requirements.txt" in known_files or "pyproject.toml" in known_files:
        instructions.append(
            "Likely a Python-based project. You might run `pip install -r requirements.txt` or `pip install .`, then run `python main.py`."
        )
    if "package.json" in known_files:
        instructions.append(
            "Likely a Node.js project. You might run `npm install` or `yarn install`, then `npm run start` or `yarn start`."
        )
    if "Makefile" in known_files:
        instructions.append("Contains a Makefile. Try `make build` or `make run`.")
    if not instructions:
        instructions.append(
            "No common build files found. Fill in your instructions here."
        )

    return "\n".join(instructions)


def guess_deployment(repo_path: str) -> str:
    """
    Check for typical deployment indicators: Dockerfile, .deploy folder, Jenkinsfile, etc.
    """
    found = []
    files_in_root = set(os.listdir(repo_path))

    # .deploy or deploy folder
    if ".deploy" in files_in_root:
        found.append("A `.deploy` folder suggests some custom deployment scripts.")
    if "deploy" in files_in_root:
        found.append("A `deploy` folder suggests custom deployment scripts.")

    # Docker
    if "Dockerfile" in files_in_root:
        found.append("A `Dockerfile` is present (Docker-based deployment).")
    if "docker-compose.yml" in files_in_root:
        found.append("A `docker-compose.yml` file is present.")
    if "Jenkinsfile" in files_in_root:
        found.append("A `Jenkinsfile` is present (Jenkins-based CI/CD).")

    if found:
        return "\n".join(found)
    return "No obvious deployment config found. (Check for other CI/CD systems.)"


# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_git_repo.py <path_to_git_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.isdir(repo_path):
        print(f"[ERROR] '{repo_path}' is not a directory or does not exist.")
        sys.exit(1)

    # Make sure this is a valid Git repository by checking for .git folder
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(
            f"[ERROR] '{repo_path}' does not appear to be a Git repo (no .git folder)."
        )
        sys.exit(1)

    # Gather info
    repo_name = get_repo_name(repo_path)
    date_author_info = get_dates_and_authors(repo_path)
    build_instructions = guess_build_run_instructions(repo_path)
    deploy_info = guess_deployment(repo_path)

    # Compose user prompt for the LLM
    # We'll ask the LLM to produce a README.
    user_prompt = f"""
Please generate a draft README for this repository with the following information:

- **Repo Name**: {repo_name}
- **Earliest Commit Date**: {date_author_info['earliest_commit_date']}
- **Latest Commit Date**: {date_author_info['latest_commit_date']}
- **Has it been active recently?**: {date_author_info['active_recently']}
- **Contributors**: {", ".join(date_author_info['authors'])}

We guessed how to build/run the project:
{build_instructions}

We guessed how it's deployed:
{deploy_info}

We also want placeholders or sections for:
- **Ownership** (the team or individual who owns this repo)
- **High-level Architecture** or design details
- **Roadmap / Future Plans**

Please produce a **markdown** README that includes:
1. A short description of the project
2. A summary of interesting commit or activity trends
3. Steps for building/running the project
4. Notes on deployment
5. Clearly labeled placeholders or prompts for ownership and architecture details
6. Any other relevant notes or disclaimers

If some information is not detected or uncertain, just make a note that a human should fill it in.
"""

    # (Optional) We can have a simple system prompt for style or clarity
    system_prompt = "You are a helpful assistant for generating README content. Please be concise and helpful."

    # Call the LLM
    readme_content = call_llm_system_user(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=2000,
    )

    # Print the README to stdout
    print(readme_content)


if __name__ == "__main__":
    main()
