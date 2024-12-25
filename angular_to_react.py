#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python migrate_angular_to_react.py --source_dir <source_dir> --output_dir <output_dir> [--force]

description:
    This script converts a legacy AngularJS application to a React application,
    storing intermediate states in an SQLite database, providing a progress bar
    for file processing, and allowing interruption/resumption.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import traceback

from bs4 import BeautifulSoup
from google import genai
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def print_error_and_exit(message: str) -> None:
    """
    Prints an error message and exits the script.
    """
    print(f"Error: {message}")
    sys.exit(1)

gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_model_name = 'gemini-2.0-flash-exp'
openai_token = os.environ.get("OPENAI_API_TOKEN")
openai_model_name = 'gpt-4o-mini'

if not gemini_key and not openai_token:
    print_error_and_exit(
        "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. "
        "Please set one before running this script."
    )

use_gemini = bool(gemini_key)
if use_gemini:
    gemini_client = genai.Client(api_key=gemini_key)

# ---------------------------------------------------------------------------
# LLM Infer Function
# ---------------------------------------------------------------------------
MAX_RETRIES = 10

def infer(prompt: str) -> str:
    """
    Calls an LLM backend (Gemini or OpenAI) to get a response.
    Retries on transient errors (e.g., rate limits).
    Returns the LLM text output only.
    """
    backend_name = "Gemini" if use_gemini else "OpenAI"

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            logging.debug(f"[DEBUG] Attempt {attempt}/{MAX_RETRIES} to call {backend_name} API.")

        try:
            if use_gemini:
                # Gemini backend
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt.strip()
                )
                return response.text.strip()
            else:
                # OpenAI backend
                client = OpenAI(api_key=openai_token)
                completion = client.chat.completions.create(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    response_format={"type": "text"},
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                return completion.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e)
            logging.debug(f"[DEBUG] {backend_name} API call attempt {attempt} failed: {err_str}")
            traceback.print_exc()

            if attempt == MAX_RETRIES:
                print_error_and_exit(
                    f"Failed to get a valid response from {backend_name} after multiple attempts."
                )

            if "rate limit" in err_str.lower():
                wait_time = 2 ** (attempt - 1)
                logging.debug(
                    f"[DEBUG] Rate limit encountered. Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)

    return ""

# ---------------------------------------------------------------------------
# Remove Code Fences
# ---------------------------------------------------------------------------
def strip_code_fences(text: str) -> str:
    """
    Strips out code fences (like ```jsx or ``` or ```javascript).
    """
    # Remove any lines starting with triple backticks and optional language spec
    text = re.sub(r"```[\w-]*", '', text)
    # Remove any remaining triple backticks
    text = re.sub(r"```", '', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------------
db_name = "migration_state.db"


def init_db() -> None:
    """
    Initialize the SQLite database if it doesn't already exist.
    Adds an iteration_count column to track how many times improvements ran.
    """
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS converted_files (
                file_path TEXT PRIMARY KEY,
                status TEXT,
                output_file_path TEXT,
                error_msg TEXT,
                iteration_count INTEGER
            )
            """
        )
    finally:
        conn.close()


def get_conversion_status(file_path: str):
    """
    Returns a tuple of (status, output_file_path, error_msg, iteration_count)
    for the given file_path, or None if it doesn't exist in the DB.
    """
    conn = sqlite3.connect(db_name)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT status, output_file_path, error_msg, iteration_count
            FROM converted_files
            WHERE file_path=?
            """,
            (file_path,),
        )
        row = cur.fetchone()
        return row if row else None
    finally:
        conn.close()


def update_conversion_status(
    file_path: str,
    status: str,
    output_file_path: str = None,
    error_msg: str = None,
    iteration_count: int = None
) -> None:
    """
    Insert or update the conversion status for a given file_path, including iteration_count.
    """
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            """
            INSERT INTO converted_files
            (file_path, status, output_file_path, error_msg, iteration_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path)
            DO UPDATE SET
              status=excluded.status,
              output_file_path=excluded.output_file_path,
              error_msg=excluded.error_msg,
              iteration_count=excluded.iteration_count
            """,
            (file_path, status, output_file_path, error_msg, iteration_count),
        )
        conn.commit()
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Parsing / Conversion Helpers
# ---------------------------------------------------------------------------
def parse_html(html_content: str) -> BeautifulSoup:
    """
    Parses HTML content using BeautifulSoup.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    except Exception as exc:
        logging.error("Error parsing HTML: %s", exc)
        return None


def parse_js(js_filename: str) -> dict:
    """
    Parses JavaScript content using esprima via subprocess.
    Assumes parse_js_esprima.js is in the same directory
    and that Node is installed.
    """
    try:
        result = subprocess.run(
            ["node", "parse_js_esprima.js", js_filename],
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            logging.error(
                "Error parsing JavaScript with esprima. Stderr: %s",
                result.stderr
            )
            return {}
    except FileNotFoundError:
        logging.error(
            "Error: parse_js_esprima.js not found. Make sure it's in the same directory."
        )
        return {}
    except Exception as exc:
        logging.error("Error parsing JavaScript %s: %s", js_filename, exc)
        return {}


def is_angular_module_file(file_path: str) -> bool:
    """
    Simple check for Angular module files, e.g., ends with '-module.js'.
    """
    return file_path.endswith('-module.js')


def extract_file_content(file_path: str) -> str:
    """
    Reads a file from disk and returns its contents as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            return file_obj.read()
    except Exception as exc:
        logging.error("Error reading file %s: %s", file_path, exc)
        return ""

# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------
def generate_react_component_from_controller(
    controller_content: str,
    dependencies: str,
    ast: dict
) -> str:
    """
    Converts an AngularJS controller to a React component using the LLM.
    Returns ONLY the LLM's result (the React code).
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting "
        "AngularJS components to ReactJS.\n"
        "You'll be provided with an AngularJS controller, relevant "
        "dependencies, and a parsed AST tree.\n"
        "Convert the following AngularJS controller to a functional React "
        "component that uses hooks.\n"
        "Follow these rules:\n"
        "1. Convert $scope properties to useState/useRef.\n"
        "2. Convert Angular services to plain JS objects with import statements.\n"
        "3. Convert Angular event listeners to React equivalents.\n"
        "4. Convert Angular dependency injection to props or import statements.\n"
        "5. Use functional components with React Hooks for state and lifecycle.\n"
        "6. Use JS (no TypeScript).\n"
        "7. Use proper JS import statements for all dependencies.\n"
        "8. For shared state, recommend React context.\n"
        "9. Comment all code for clarity.\n"
        "10. Return only the converted React component code (with imports).\n"
        "11. Translate any 'this' references appropriately.\n"
        "12. No external state management library; if state is shared, mention createContext.\n"
        "13. Output in ES6 JavaScript (no TypeScript).\n\n"
        "AngularJS Controller:\n"
        f"```javascript\n{controller_content}\n```\n\n"
        "JavaScript AST:\n"
        f"```json\n{json.dumps(ast)}\n```\n\n"
        "Dependencies:\n"
        f"```javascript\n{dependencies}\n```\n"
    )
    response = infer(prompt)
    return strip_code_fences(response)


def generate_react_component_from_html_template(
    template_content: str,
    file_path: str
) -> str:
    """
    Converts an AngularJS HTML template to a React component using the LLM.
    Returns ONLY the LLM's result (the React code).
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS "
        "HTML templates to ReactJS.\n"
        "Convert the following AngularJS HTML template to a React component.\n"
        "Follow these rules:\n"
        "1. Translate Angular directives (ng-show, ng-repeat, ng-if, ng-click, ng-model) "
        "   to their React equivalents using React’s JSX syntax and hooks.\n"
        "2. Use JSX syntax.\n"
        "3. Use descriptive class names.\n"
        "4. Return a valid functional React component with necessary imports, ensuring "
        "   any state is managed with hooks.\n"
        "5. Convert Angular event handlers like ng-click to React's onClick, etc.\n"
        "6. Convert any binding like {{property}} to JSX syntax.\n"
        "7. Use valid React attributes.\n"
        "8. Do not assume any particular styling library; use inline styles only if necessary.\n"
        "9. Handle conditional rendering with JSX.\n"
        "10. Use Fragments (<></>) when needed.\n"
        "11. Output ES6 JavaScript, not TypeScript.\n"
        "12. If the HTML template is a gridCellTemplate, create a separate React component "
        "   and import it.\n"
        "13. If the file_path contains gridHeaderTemplate, generate JSX only for the table "
        "   headers.\n\n"
        "HTML Template:\n"
        f"```html\n{template_content}\n```\n"
    )
    response = infer(prompt)
    return strip_code_fences(response)


def convert_angular_module(module_content: str) -> str:
    """
    Converts an AngularJS module to a React equivalent or plain JS file using the LLM.
    Returns ONLY the LLM's result (the React code).
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS modules "
        "to ReactJS.\n"
        "Convert the following AngularJS module to either a React context or a simple JS "
        "file:\n"
        "1. Translate AngularJS services/values to JS objects or contexts.\n"
        "2. Use ES6 import and export statements.\n"
        "3. If services are used throughout, convert them into a React context.\n"
        "4. Include explanatory comments.\n"
        "5. Output ES6 JavaScript (no TypeScript).\n\n"
        "AngularJS Module:\n"
        f"```javascript\n{module_content}\n```\n"
    )
    response = infer(prompt)
    return strip_code_fences(response)

# ---------------------------------------------------------------------------
# Feedback Loop Helpers
# ---------------------------------------------------------------------------
def score_react_component(component_code: str) -> int:
    """
    Scores the React component based on common mistakes. Negative score if we see Angular artifacts,
    positive if we detect features typical of a good migration.
    """
    if not component_code:
        return 0  # no score if empty

    score = 0
    # Penalize leftover AngularJS directives
    if 'ng-show' in component_code:
        score -= 1
    if 'ng-if' in component_code:
        score -= 1
    if 'ng-repeat' in component_code:
        score -= 1
    if 'ng-click' in component_code:
        score -= 1
    if 'ng-model' in component_code:
        score -= 1
    if '{{' in component_code:
        score -= 1

    # If using React standard, we expect className for styling
    # If we don't see it, penalize
    if 'className=' not in component_code:
        score -= 1

    # If we see at least one useState, reward a point
    if 'useState(' in component_code:
        score += 1

    return score


def improve_react_component(
    file_path: str,
    original_content: str,
    converted_code: str,
    iteration_count: int
) -> str:
    """
    Improves a React component via a feedback loop with the LLM prompt.
    Looks at the existing converted code, compares with the original HTML,
    and tries to remove leftover Angular artifacts.
    Returns ONLY the improved React code.
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS to ReactJS.\n"
        "You'll be provided with an original AngularJS HTML template and the current converted ReactJS component.\n"
        "Review the converted React component, identify any mistakes, fix them, and produce a correct React code.\n"
        "Follow these rules:\n"
        "1. Translate any Angular-specific directives (e.g., ng-show, ng-repeat, ng-if, ng-click, ng-model)\n"
        "   to their React equivalents using React’s JSX syntax and React’s hooks (e.g., useState, conditional rendering, etc.).\n"
        "2. Use JSX syntax.\n"
        "3. Use descriptive class names.\n"
        "4. Return a valid functional React component with necessary import statements, ensuring any state used is managed with hooks.\n"
        "5. Convert Angular event handlers like ng-click into React’s onClick syntax, ensuring that events are handled correctly.\n"
        "6. Convert any binding like {{property}} to JSX syntax. Make sure to handle both one-way and two-way binding.\n"
        "7. All attributes must be valid React attributes.\n"
        "8. Do not assume any particular styling library; use inline styles only if necessary.\n"
        "9. Handle conditional rendering with a JSX ternary operator or logical &&.\n"
        "10. Use Fragments (<></>) when needed to avoid extra wrapping divs.\n"
        "11. Output in ES6 JavaScript (no TypeScript).\n"
        "12. If the HTML template is a gridCellTemplate, create a separate React component that can be used as a template.\n"
        "13. If the HTML template contains a table, ensure the table is rendered properly with headers and cells.\n"
        "14. If the file_path contains gridHeaderTemplate, generate JSX only for the table headers.\n"
        "15. Remove any extraneous code, leftover Angular bits, or irrelevant comments.\n"
        "16. Only return the final, improved ReactJS component code with all import statements.\n"
        f"\nOriginal AngularJS HTML Template:\n```html\n{original_content}\n```\n"
        f"\nCurrent React Component:\n```javascript\n{converted_code}\n```\n"
        f"\nIteration {iteration_count}\n"
    )

    try:
        response = infer(prompt)
        return strip_code_fences(response)
    except Exception as e:
        logging.error("Error improving react component: %s", e)
        return converted_code

# ---------------------------------------------------------------------------
# Main Processing Logic
# ---------------------------------------------------------------------------
def write_converted_file(output_file_path: str, content: str) -> None:
    """
    Writes `content` to `output_file_path`, creating directories if needed.
    """
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write(content)


def handle_html_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    force: bool
) -> None:
    """
    Handles conversion of HTML files to React components.
    """
    record = get_conversion_status(file_path)
    # If status is success and not forcing, skip
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path,
            "error",
            error_msg="empty or unreadable file"
        )
        return

    try:
        converted_code = generate_react_component_from_html_template(file_content, file_path)
        if not converted_code:
            update_conversion_status(file_path, "error", error_msg="no initial code")
            return

        # Improvement feedback loop
        max_iterations = 5
        iteration_count = 0
        score = score_react_component(converted_code)

        while score < 0 and iteration_count < max_iterations:
            iteration_count += 1
            converted_code = improve_react_component(
                file_path,
                file_content,
                converted_code,
                iteration_count
            )
            score = score_react_component(converted_code)

        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(
            output_dir,
            rel_path.replace('.html', '.js')
        )
        write_converted_file(output_file_path, converted_code)

        update_conversion_status(
            file_path,
            "success",
            output_file_path=output_file_path,
            iteration_count=iteration_count
        )

    except Exception as exc:
        logging.error("Error processing HTML file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_module_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: dict,
    force: bool
) -> None:
    """
    Handles conversion of Angular module files to React equivalents.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path,
            "error",
            error_msg="empty or unreadable file"
        )
        return

    try:
        react_module_code = convert_angular_module(file_content)
        if react_module_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_module_code)

            update_conversion_status(
                file_path,
                "success",
                output_file_path=output_file_path
            )

            # Store the original module content for potential dependencies
            module_name = os.path.basename(file_path).replace('-module.js', '')
            angular_modules[module_name] = file_content

    except Exception as exc:
        logging.error("Error processing module file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_controller_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: dict,
    force: bool
) -> None:
    """
    Handles conversion of Angular controller files to React components.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path,
            "error",
            error_msg="empty or unreadable file"
        )
        return

    try:
        ast = parse_js(file_path)
        if not ast:
            update_conversion_status(file_path, "error", error_msg="ast is empty")
            return

        dependencies = [
            dep['value'] for dep in ast.get('dependencies', [])
            if 'value' in dep
        ]

        # Gather content from relevant modules
        dependency_files_content = ""
        for dep in dependencies:
            if dep in angular_modules:
                dependency_files_content += angular_modules[dep] + "\n"

        react_component_code = generate_react_component_from_controller(
            file_content,
            dependency_files_content,
            ast
        )
        if react_component_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_component_code)

            update_conversion_status(
                file_path,
                "success",
                output_file_path=output_file_path
            )

    except Exception as exc:
        logging.error("Error processing controller file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_generic_js_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    force: bool
) -> None:
    """
    Handles copying over generic JS files that don't match Angular patterns.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path,
            "error",
            error_msg="empty or unreadable file"
        )
        return

    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        # Just copy the original. Not run through LLM.
        write_converted_file(output_file_path, file_content)

        update_conversion_status(
            file_path,
            "success",
            output_file_path=output_file_path
        )
    except Exception as exc:
        logging.error("Error processing generic JS file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_other_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    force: bool
) -> None:
    """
    Handles copying over other file types unchanged.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path,
            "error",
            error_msg="empty or unreadable file"
        )
        return

    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        # Just copy the original.
        write_converted_file(output_file_path, file_content)

        update_conversion_status(
            file_path,
            "success",
            output_file_path=output_file_path
        )
    except Exception as exc:
        logging.error("Error processing file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))

def process_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: dict,
    force: bool = False
) -> None:
    """
    Routes files to the correct handler based on file extension
    and Angular usage.
    """
    if file_path.endswith('.html'):
        handle_html_file(file_path, source_dir, output_dir, force)
    elif file_path.endswith('.js'):
        if is_angular_module_file(file_path):
            handle_angular_module_file(
                file_path, source_dir, output_dir, angular_modules, force
            )
        elif 'controller' in file_path.lower():
            handle_angular_controller_file(
                file_path, source_dir, output_dir, angular_modules, force
            )
        else:
            handle_generic_js_file(file_path, source_dir, output_dir, force)
    else:
        handle_other_file(file_path, source_dir, output_dir, force)

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert AngularJS to React.")
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Path to the source directory containing AngularJS code."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for React code."
    )
    parser.add_argument(
        "--force",
        help="Force re-conversion even if the file is already marked 'success'.",
        action="store_true"
    )
    args = parser.parse_args()

    # 1. Initialize DB
    init_db()

    # 2. Prepare to store Angular modules
    angular_modules = {}

    # 3. Gather all .html and .js files from the source directory
    file_paths = []
    for root, _, files in os.walk(args.source_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            # Skip DB file or the script itself if present
            if full_path.endswith('.html') or full_path.endswith('.js'):
                file_paths.append(full_path)

    # 4. Process each file with a progress bar
    for file_path in tqdm(file_paths, desc="Converting files"):
        tqdm.write(f"Processing: {file_path}")
        process_file(
            file_path,
            args.source_dir,
            args.output_dir,
            angular_modules,
            force=args.force
        )

    # 5. Done
    logging.info("Conversion complete. See '%s' for details.", db_name)


if __name__ == "__main__":
    main()
