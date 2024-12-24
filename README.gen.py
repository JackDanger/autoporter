#!/usr/bin/env python3
"""
Analyze a code repository using a single LLM pass per file.

Goals:
- For each file that looks like source code, send its entire contents to the LLM once.
- The LLM returns a structured JSON with fields:
  {
    "endpoints": [... or "No relevant info"],
    "db_schema": [... or "No relevant info"],
    "external_calls": [... or "No relevant info"],
    "language": "..." or "unknown",
    "summary": "..." or "No relevant info"
  }

We then aggregate these results:
- endpoints to http_endpoints.txt
- db schema info to db_schema.dot (one combined graph)
- third-party calls to third_party_calls.txt
- a final README.gen summarizing all.

We implement a heuristic function identify_structure() that determines if a file is code or not.
We do not attempt directory filtering. We analyze all code files.

We add exponential backoff for rate-limit errors as before.
"""

import os
import sys
import traceback
import time
import json
import re
from tqdm import tqdm
from google import genai
from openai import OpenAI

MAX_RETRIES = 3


def print_error_and_exit(message):
    print(f"Error: {message}")
    sys.exit(1)


def init_clients():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_token = os.environ.get("OPENAI_API_TOKEN")

    if not gemini_key and not openai_token:
        print_error_and_exit(
            "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. "
            "Please set one before running this script."
        )

    gemini_client = None
    #gemini_model_name = 'gemini-2.0-flash-exp'
    gemini_model_name = 'gemini-1.5-pro'
    if gemini_key:
        print("[DEBUG] Initializing Gemini client...")
        try:
            gemini_client = genai.Client(api_key=gemini_key)
        except Exception as e:
            print_error_and_exit(f"Failed to initialize Gemini client: {str(e)}")

    return gemini_client, gemini_model_name


def infer(prompt, gemini_client=None, gemini_model_name=None):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_token = os.environ.get("OPENAI_API_TOKEN")

    if not gemini_key and not openai_token:
        print_error_and_exit("Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. Cannot proceed.")

    use_gemini = bool(gemini_key)
    backend_name = "Gemini" if use_gemini else "OpenAI gpt-4o"

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"[DEBUG] Attempt {attempt}/{MAX_RETRIES} to call {backend_name} API.")
        try:
            if use_gemini:
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt.strip()
                )
                return response.text.strip()
            else:
                client = OpenAI(api_key=openai_token)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt.strip()}],
                    response_format={"type": "text"},
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0)
                return completion.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e)
            print(f"[DEBUG] {backend_name} API call attempt {attempt} failed: {err_str}")
            traceback.print_exc()

            if attempt == MAX_RETRIES:
                print_error_and_exit(
                    f"Failed to get a valid response from {backend_name} after multiple attempts."
                )

            if "rate limit" in err_str.lower():
                wait_time = 2 ** (attempt - 1)
                print(f"[DEBUG] Rate limit encountered. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            # Otherwise just retry immediately

    return ""  # Should not reach here


def identify_structure(file_path):
    """
    Determine if a file likely contains code and should be analyzed.

    Heuristics:
    1. Check file extension. If it's a common code extension (e.g. py, js, java, php, cs, rb, go, ts),
       consider it likely code.
    2. If not a known extension but the file is small (<1MB) and contains code-like patterns:
       - Matches something like function definitions, imports, class definitions.
    3. If binary or too large (>5MB), skip it.
    """

    # File size check
    if os.path.getsize(file_path) > 5 * 1024 * 1024:  # 5MB limit
        return False

    known_extensions = {
        ".py", ".js", ".java", ".php", ".cs", ".rb", ".go", ".ts", ".c", ".cpp", ".swift",
        ".rs", ".scala", ".vb", ".sql"
    }
    _, ext = os.path.splitext(file_path.lower())
    if ext in known_extensions:
        return True

    # If extension isn't known, try content heuristics
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(5000)  # read first 5000 chars
    except:
        return False

    # Simple heuristics: check for common code patterns
    code_patterns = [
        r"(def |function |func |public |private |class )",
        r"(import |using |#include )",
        r"(=>|->|\{.*\})"
    ]

    for pattern in code_patterns:
        if re.search(pattern, sample):
            return True

    return False


def analyze_file_once(file_path, gemini_client, gemini_model_name):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    prompt = """
You are an AI assistant analyzing a single code file.
The file contents are between triple backticks:
```
""" + content + """
```
Produce a JSON only, with the following keys:
- "endpoints": a list of HTTP endpoints found (like [{"method":"GET","path":"/api/user"}]) or "No relevant info"
- "db_schema": a representation of database entities and relationships; if none, "No relevant info"
- "external_calls": a list of external service calls, or "No relevant info"
- "language": best guess of the programming language used, or "unknown"
- "summary": a brief summary of what this file does, or "No relevant info"

No extra text outside of JSON. Make sure the JSON is well-formed and uses double quotes.
If you cannot determine something, put "No relevant info" for that key.
If endpoints/db_schema/external_calls are found, output them in a structured way (arrays or objects).
If no endpoints, db_schema, or external_calls are found, return the string "No relevant info" for that key.
"""

    response = infer(prompt, gemini_client=gemini_client, gemini_model_name=gemini_model_name)

    # Try parsing JSON. If fails, we will return a default structure.
    try:
        # Remove Markdown code fences if present
        cleaned_response = re.sub(r"```(?:json)?\s*", "", response.strip())
        cleaned_response = re.sub(r"```", "", cleaned_response).strip()

        data = json.loads(cleaned_response)
        return data
    except json.JSONDecodeError:
        print(cleaned_response)
        print("[DEBUG] Failed to parse JSON from LLM response. Returning default.")
        return {
            "endpoints": "No relevant info",
            "db_schema": "No relevant info",
            "external_calls": "No relevant info",
            "language": "unknown",
            "summary": "No relevant info"
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_repo.py <path_to_git_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        print_error_and_exit("The provided path does not exist or is not a directory.")

    gemini_client, gemini_model_name = init_clients()

    # Enumerate files
    print("[DEBUG] Enumerating files in the repository...")
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        if '.git' in root:
            continue
        for filename in files:
            if filename.startswith('.git'):
                continue
            full_path = os.path.join(root, filename)
            file_list.append(full_path)

    if not file_list:
        print("No files found in the provided repository path. Exiting.")
        sys.exit(0)

    # Write out the file list for traceability
    file_list_path = os.path.join(repo_path, "file_list.txt")
    with open(file_list_path, "w", encoding="utf-8") as f:
        for fp in file_list:
            f.write(os.path.relpath(fp, repo_path) + "\n")
    print("[DEBUG] Full file list saved to file_list.txt")

    # Before analyzing each file:
    # Create a directory for partial results if not exists
    partial_dir = os.path.join(repo_path, "partial_results")
    if not os.path.exists(partial_dir):
        os.makedirs(partial_dir, exist_ok=True)

    for fp in tqdm(file_list, desc="Analyzing files"):
        if identify_structure(fp):
            # Check if a partial result already exists for this file:
            rel_path = os.path.relpath(fp, repo_path)
            result_file_path = os.path.join(partial_dir, rel_path.replace(os.sep, '_') + ".json")
            if os.path.exists(result_file_path):
                # Already analyzed this file in a previous run
                with open(result_file_path, "r", encoding="utf-8") as rf:
                    file_result = json.load(rf)
            else:
                # Analyze file and store partial result
                file_result = analyze_file_once(fp, gemini_client, gemini_model_name)
                file_result["file"] = rel_path
                # Ensure directory exists for partial results (created above)
                with open(result_file_path, "w", encoding="utf-8") as rf:
                    json.dump(file_result, rf, indent=2)
        else:
            # Not code or too large, skip
            continue

    results.append(file_result)

    # Aggregate results
    all_endpoints = []
    all_nodes = set()
    all_edges = set()
    all_calls = []

    for r in results:
        endpoints = r.get("endpoints", "No relevant info")
        if isinstance(endpoints, list) and endpoints:
            # Expecting [{"method":"GET","path":"/api"}]
            for ep in endpoints:
                method = ep.get("method", "UNKNOWN").upper()
                path = ep.get("path", "/unknown")
                all_endpoints.append(f"{method} {path}")
        elif endpoints != "No relevant info" and endpoints:
            # If LLM returns something unexpected
            if isinstance(endpoints, str):
                all_endpoints.append(endpoints)
        # db_schema: can be complex. Expecting arrays or something similar
        db_schema = r.get("db_schema", "No relevant info")
        if db_schema != "No relevant info":
            # Heuristics: if db_schema might contain something like "ENTITY: desc"
            # or "A -> B". If it's a list, parse lines. If string, split by lines.
            lines = db_schema if isinstance(db_schema, list) else db_schema.split("\n")
            for line in lines:
                line = line.strip()
                if "->" in line:
                    all_edges.add(line)
                elif ":" in line:
                    entity = line.split(":", 1)[0].strip()
                    all_nodes.add(entity)

        external_calls = r.get("external_calls", "No relevant info")
        if isinstance(external_calls, list):
            all_calls.extend(external_calls)
        elif external_calls != "No relevant info":
            all_calls.append(external_calls)

    # Write endpoints
    http_endpoints_file = os.path.join(repo_path, "http_endpoints.txt")
    if all_endpoints:
        with open(http_endpoints_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_endpoints) + "\n")
    else:
        with open(http_endpoints_file, "w", encoding="utf-8") as f:
            f.write("No HTTP endpoints found.\n")

    # Write db_schema
    db_schema_file = os.path.join(repo_path, "db_schema.dot")
    if all_nodes or all_edges:
        dot_lines = ["digraph schema {"]
        for node in all_nodes:
            dot_lines.append(f'  "{node}" [shape=box];')
        for edge in all_edges:
            dot_lines.append(f'  {edge};')
        dot_lines.append("}")
        with open(db_schema_file, "w", encoding="utf-8") as f:
            f.write("\n".join(dot_lines) + "\n")
    else:
        with open(db_schema_file, "w", encoding="utf-8") as f:
            f.write("digraph schema {\n// No database schema found.\n}\n")

    # Write external calls
    third_party_calls_file = os.path.join(repo_path, "third_party_calls.txt")
    if all_calls:
        with open(third_party_calls_file, "w", encoding="utf-8") as f:
            for c in all_calls:
                if isinstance(c, dict):
                    # If structured call info is given
                    f.write(json.dumps(c) + "\n")
                else:
                    f.write(str(c) + "\n")
    else:
        with open(third_party_calls_file, "w", encoding="utf-8") as f:
            f.write("No external calls found.\n")

    # Produce README
    try:
        with open(http_endpoints_file, "r", encoding="utf-8") as f:
            http_endpoints_content = f.read().strip()
    except IOError:
        http_endpoints_content = "No HTTP endpoints found."

    try:
        with open(db_schema_file, "r", encoding="utf-8") as f:
            db_schema_content = f.read().strip()
    except IOError:
        db_schema_content = "digraph schema {\n// No database schema found.\n}"

    try:
        with open(third_party_calls_file, "r", encoding="utf-8") as f:
            third_party_calls_content = f.read().strip()
    except IOError:
        third_party_calls_content = "No external calls found."

    # Summarize into README.gen
    # Including language and summary was per-file. For final summary, just mention we have analyzed them.
    # We could incorporate per-file summaries, but let's keep final README similar.
    step4_prompt = f"""
You are an AI assistant. You have these analyses from multiple files:

HTTP endpoints:
{http_endpoints_content}

Database schema:
{db_schema_content}

Third-party calls:
{third_party_calls_content}

Produce a README as follows:

# Analysis Summary
A one-paragraph summary synthesizing the information from the analyses.

## HTTP Endpoints
{http_endpoints_content}

## Database Schema
{db_schema_content}

## Third-Party Calls
{third_party_calls_content}
"""

    readme_file = os.path.join(repo_path, "README.gen")
    step4_output = infer(step4_prompt, gemini_client=gemini_client, gemini_model_name=gemini_model_name)
    try:
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(step4_output + "\n")
    except IOError as e:
        print_error_and_exit(
            f"Error writing README.gen: {str(e)}. "
            "Check write permissions to the repository directory."
        )
    print(f"[DEBUG] Analysis complete. The final README is at: {readme_file}")


if __name__ == "__main__":
    main()
