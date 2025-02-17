#!/usr/bin/env python3
"""
Enhanced script that analyzes a Git repository and outputs a basic README draft to stdout,
using an LLM for text generation. This version provides richer file-structure insights,
including last-commit dates and a highlight of recently updated files.

Usage:
    python analyze_git_repo.py <path_to_git_repo>

Environment Variables (same as the advanced script):
    OPENAI_API_KEY: Your OpenAI API key (optional)
    GEMINI_API_KEY: Your Google PaLM (Gemini) API key (optional)
    DEEPSEEK_API_KEY: Your DeepSeek API key (optional)
"""

import os
import sys
import subprocess
import datetime
import time
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# LLM / Model Configuration
# (Same approach as before)
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
    print("[WARN] google-generativeai module not installed; Gemini calls will not work.")
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
        if "o1" in OPENAI_MODEL:  # Some internal or special model naming
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

    # 3) Try DeepSeek (treated like an OpenAI-compatible model name)
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
        return call_vllm(combined_prompt, temperature=temperature, max_tokens=max_tokens)

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
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
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
        return "".join([r.text for r in results])
    except Exception as e:
        print(f"[ERROR] vLLM inference error: {e}")
        return ""

# ----------------------------------------------------------------------
# Simple Utility: Chunk large text to avoid token limits
# ----------------------------------------------------------------------
def chunk_text(text: str, max_chunk_size: int = 12000) -> List[str]:
    """
    Splits text into chunks not exceeding max_chunk_size characters.
    This helps avoid overly large prompts for the LLM.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chunk_size])
        start += max_chunk_size
    return chunks

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
    # Get earliest commit date (first commit in the repo)
    log_oldest = run_git_command(repo_path, ["log", "--reverse", "--pretty=%cd", "--date=short", "-1"])
    info["earliest_commit_date"] = log_oldest

    # Get latest commit date (most recent commit)
    log_latest = run_git_command(repo_path, ["log", "-1", "--pretty=%cd", "--date=short"])
    info["latest_commit_date"] = log_latest

    # Grab all authors
    all_authors = run_git_command(repo_path, ["log", "--format=%an"])
    authors_set = set(all_authors.splitlines())
    info["authors"] = sorted(authors_set)

    # Check if there's a commit in the last 1-2 years
    log_recent = run_git_command(repo_path, ["log", '--since=1.year.ago', "--pretty=oneline", "-1"])
    info["active_recently"] = bool(log_recent)

    return info

def guess_build_run_instructions(repo_path: str) -> str:
    """
    A naive guess for how the project might be built or run, based on file inspection.
    """
    known_files = os.listdir(repo_path)
    instructions = []

    if "requirements.txt" in known_files or "pyproject.toml" in known_files:
        instructions.append(
            "Likely a Python-based project. Try `pip install -r requirements.txt` or `pip install .`, then `python main.py`."
        )
    if "package.json" in known_files:
        instructions.append(
            "Likely a Node.js project. Try `npm install` or `yarn install`, then `npm run start` or `yarn start`."
        )
    if "Makefile" in known_files:
        instructions.append(
            "Contains a Makefile. Try `make build` or `make run`."
        )
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

    if ".deploy" in files_in_root:
        found.append("A `.deploy` folder suggests some custom deployment scripts.")
    if "deploy" in files_in_root:
        found.append("A `deploy` folder suggests custom deployment scripts.")
    if "Dockerfile" in files_in_root:
        found.append("A `Dockerfile` is present (Docker-based deployment).")
    if "docker-compose.yml" in files_in_root:
        found.append("A `docker-compose.yml` file is present.")
    if "Jenkinsfile" in files_in_root:
        found.append("A `Jenkinsfile` is present (Jenkins-based CI/CD).")

    if found:
        return "\n".join(found)
    return "No obvious deployment config found. (Check for other CI/CD systems.)"

# New function: retrieve file list + last commit dates
def get_tracked_files_with_dates(repo_path: str) -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries with each file's name and last commit date.
    Example: [{'filename': 'src/main.py', 'last_commit_date': '2025-01-15 10:24:32 +0200'}, ...]
    """
    # 1) Get all tracked files
    raw_files = run_git_command(repo_path, ["ls-files"])
    files_list = raw_files.splitlines()

    file_info_list = []
    for f in files_list:
        # For each file, run `git log -1 --pretty=format:%ci <file>` to get the last commit date
        last_date_str = run_git_command(repo_path, ["log", "-1", "--pretty=format:%ci", f])
        file_info_list.append({
            "filename": f,
            "last_commit_date": last_date_str
        })

    return file_info_list

def sort_files_by_last_commit(file_info_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Given a list of file info dicts (with 'filename' and 'last_commit_date'),
    return them sorted by last_commit_date descending (most recent first).
    """
    def parse_date(date_str: str):
        # date_str looks like '2025-01-15 10:24:32 +0200'
        # We'll parse it to a datetime for sorting. If parse fails, return minimal date.
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    return sorted(file_info_list, key=lambda x: parse_date(x["last_commit_date"]), reverse=True)

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
        print(f"[ERROR] '{repo_path}' does not appear to be a Git repo (no .git folder).")
        sys.exit(1)

    # Gather info
    repo_name = get_repo_name(repo_path)
    date_author_info = get_dates_and_authors(repo_path)
    build_instructions = guess_build_run_instructions(repo_path)
    deploy_info = guess_deployment(repo_path)

    # Gather file structure + last commit dates
    file_info_list = get_tracked_files_with_dates(repo_path)
    if not file_info_list:
        file_info_text = "No tracked files found (repo is empty or something unexpected)."
    else:
        # Sort them by last modified date (descending)
        sorted_files = sort_files_by_last_commit(file_info_list)

        # We'll create a textual summary of all files + last commit date.
        # If the repo has a huge number of files, we might chunk or limit to avoid a massive prompt.
        lines = []
        for info in sorted_files:
            lines.append(f"{info['filename']} (last commit: {info['last_commit_date']})")

        file_info_text_raw = "\n".join(lines)
        # If it's huge, we chunk it. We'll pass the chunked content to the LLM in separate sections.
        # For demonstration, let's just store it in `file_info_text` and do chunking in the LLM prompt logic.
        file_info_text = file_info_text_raw

    # We'll highlight the top ~5 most recently modified files:
    top_recent = sorted_files[:5] if sorted_files else []
    top_recent_text = "\n".join([f"{f['filename']} ({f['last_commit_date']})" for f in top_recent])

    # Compose user prompt for the LLM
    # We'll chunk the big file info if needed, but let's keep it simpler:
    # We'll just embed it in a single string. If it goes over token limit, you may need more advanced chunking.
    file_info_chunks = chunk_text(file_info_text, max_chunk_size=6000)

    # Combine them into a single big string (or multiple if you prefer).
    # We'll do a naive approach: just put each chunk after a separator.
    # A more robust approach might call the LLM multiple times or do additional summarization.
    combined_file_info_text = ""
    for i, chunk in enumerate(file_info_chunks, start=1):
        combined_file_info_text += f"\n--- Files Chunk {i}/{len(file_info_chunks)} ---\n{chunk}\n"

    user_prompt = f"""
Please generate a draft README for this repository with **especially** robust details for a brand new engineer joining the team. They might need:

1. **Repo Name**: {repo_name}
2. **Earliest Commit Date**: {date_author_info['earliest_commit_date']}
3. **Latest Commit Date**: {date_author_info['latest_commit_date']}
4. **Recent Activity?**: {date_author_info['active_recently']}
5. **Contributors**: {", ".join(date_author_info['authors'])}
6. **Most Recently Modified Files** (top 5 by last commit):
{top_recent_text}

---
**All Tracked Files** (with last commit dates) for further signal:
{combined_file_info_text}

---
**Guessed Build/Run Instructions**:
{build_instructions}

**Guessed Deployment**:
{deploy_info}

We also want placeholders or sections for:
- **Ownership** (the team or individual who owns this repo)
- **High-level Architecture** or design details
- **Major historical or present-day events** in the repo that new folks should know (e.g. big rewrites, forks, merges)
- **Roadmap / Future Plans**

Please produce a **markdown** README that covers:
1. A short description of the project
2. Interesting commit or activity trends (including insights from the file structure and last commit dates)
3. Steps for building/running the project
4. Notes on deployment
5. Clearly labeled placeholders for ownership, architecture, major events, etc.
6. Any other relevant disclaimers or notes for a new engineer.
"""

    # (Optional) A minimal system prompt
    system_prompt = (
        "You are a helpful assistant for generating README content. Provide detailed, thoughtful guidance."
    )

    # Call the LLM
    readme_content = call_llm_system_user(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    # Print the README to stdout
    print(readme_content)


if __name__ == "__main__":
    main()
