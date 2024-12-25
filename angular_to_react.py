#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
    python migrate_angular_to_react.py --source_dir <source_dir> --output_dir <output_dir> [--force]

description:
    This script converts a legacy AngularJS application to a React application,
    intelligently identifying and filling gaps in the translation process.
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
from typing import Dict, Optional, Tuple, List

from bs4 import BeautifulSoup
from google import genai
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def print_error_and_exit(message: str) -> None:
    """Prints an error message and exits the script."""
    print(f"Error: {message}")
    sys.exit(1)


gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_model_name = "gemini-2.0-flash-exp"
openai_token = os.environ.get("OPENAI_API_TOKEN")
openai_model_name = "gpt-4o-mini"

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
    """Calls an LLM backend to get a response."""
    backend_name = "Gemini" if use_gemini else "OpenAI"
    # ... (Implementation remains the same)
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            logging.debug(
                f"[DEBUG] Attempt {attempt}/{MAX_RETRIES} to call {backend_name} API."
            )
        try:
            if use_gemini:
                response = gemini_client.models.generate_content(
                    model=gemini_model_name, contents=prompt.strip()
                )
                return response.text.strip()
            else:
                client = OpenAI(api_key=openai_token)
                completion = client.chat.completions.create(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    response_format={"type": "text"},
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return completion.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            logging.debug(
                f"[DEBUG] {backend_name} API call attempt {attempt} failed: {err_str}"
            )
            traceback.print_exc()
            if attempt == MAX_RETRIES:
                print_error_and_exit(
                    f"Failed to get a valid response from {backend_name} after multiple attempts."
                )
            if "rate limit" in err_str.lower() or '429' in err_str.lower():
                wait_time = 2 ** (attempt - 1)
                logging.debug(
                    "[DEBUG] Rate limit encountered. Waiting %s seconds before retry...",
                    wait_time,
                )
                time.sleep(wait_time)
    return ""

# ---------------------------------------------------------------------------
# Remove Code Fences
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Strips out code fences from LLM responses."""
    text = re.sub(r"```[\w-]*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------------
db_name = "migration_state.db"


def init_db() -> None:
    """Initializes the SQLite database."""
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


def get_conversion_status(file_path: str) -> Optional[Tuple[str, str, str, int]]:
    """Gets the conversion status of a file."""
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
    output_file_path: Optional[str] = None,
    error_msg: Optional[str] = None,
    iteration_count: Optional[int] = None,
) -> None:
    """Updates the conversion status of a file."""
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


def parse_html(html_content: str) -> Optional[BeautifulSoup]:
    """Parses HTML content using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup
    except Exception as exc:
        logging.error("Error parsing HTML: %s", exc)
        return None


def parse_js(js_filename: str) -> Dict:
    """Parses JavaScript content using esprima via subprocess."""
    try:
        result = subprocess.run(
            ["node", "parse_js_esprima.js", js_filename],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            logging.error(
                "Error parsing JavaScript with esprima. Stderr: %s", result.stderr
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
    """Checks if a file is an Angular module file."""
    return file_path.endswith("-module.js")


def extract_file_content(file_path: str) -> str:
    """Reads a file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            return file_obj.read()
    except Exception as exc:
        logging.error("Error reading file %s: %s", file_path, exc)
        return ""

# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------


def generate_react_component_from_controller(
    controller_content: str, dependencies: str, ast: Dict
) -> str:
    """Converts an AngularJS controller to a React component."""
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
    template_content: str, file_path: str
) -> str:
    """Converts an AngularJS HTML template to a React component."""
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
    """Converts an AngularJS module to a React equivalent or plain JS file."""
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
# Feedback Loop Helpers (Modified for Gap Identification)
# ---------------------------------------------------------------------------


def identify_missing_logic(original_html: str, react_component_code: str) -> List[str]:
    """
    Identifies potential gaps in the React component by comparing it to the original HTML.
    Returns a list of strings describing the missing logic.
    """
    missing_logic = []
    original_soup = parse_html(original_html)
    if not original_soup:
        return missing_logic

    # Basic checks for common directives - you can expand this
    if original_soup.find(attrs={"ng-click": True}) and "onClick" not in react_component_code:
        missing_logic.append("Missing 'onClick' handler for elements with 'ng-click'.")
    if original_soup.find(attrs={"ng-show": True}) and "useState" not in react_component_code and "style={{display:" not in react_component_code:
        missing_logic.append("Missing conditional rendering (e.g., using 'useState') for elements with 'ng-show'.")
    if original_soup.find(attrs={"ng-if": True}) and not any(op in react_component_code for op in ["&&", "?"]):
        missing_logic.append("Missing conditional rendering for elements with 'ng-if'.")
    if original_soup.find(attrs={"ng-repeat": True}) and ".map(" not in react_component_code:
        missing_logic.append("Missing list rendering (e.g., using '.map()') for elements with 'ng-repeat'.")
    if original_soup.find(attrs={"ng-model": True}) and "useState" not in react_component_code:
        missing_logic.append("Missing state management (e.g., using 'useState') for elements with 'ng-model'.")
    if original_soup.find("{{") and "{/*" not in react_component_code: # Ignore commented out angular bindings
        missing_logic.append("AngularJS bindings '{{}}' not fully translated to JSX.")

    return missing_logic


def improve_react_component_with_gap_filling(
    file_path: str, original_content: str, converted_code: str, missing_logic: List[str]
) -> str:
    """
    Improves a React component by specifically addressing identified missing logic.
    """
    if not missing_logic:
        return converted_code

    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS to ReactJS.\n"
        "You'll be provided with an original AngularJS HTML template and the current converted ReactJS component.\n"
        f"The following potential issues were identified in the React component:\n{chr(10).join([f'- {issue}' for issue in missing_logic])}\n"
        "Review the converted React component, and the original HTML, fix the identified issues, and produce correct React code.\n"
        "Follow these rules:\n"
        "1. Translate any Angular-specific directives (e.g., ng-show, ng-repeat, ng-if, ng-click, ng-model)\n"
        "   to their React equivalents using React’s JSX syntax and React’s hooks (e.g., useState, conditional rendering, etc.).\n"
        "2. Use JSX syntax.\n"
        # ... (rest of the rules from the previous improve_react_component function)
        "16. Only return the final, improved ReactJS component code with all import statements.\n"
        f"\nOriginal AngularJS HTML Template:\n```html\n{original_content}\n```\n"
        f"\nCurrent React Component:\n```javascript\n{converted_code}\n```\n"
    )

    try:
        response = infer(prompt)
        return strip_code_fences(response)
    except Exception as e:
        logging.error("Error improving react component with gap filling: %s", e)
        return converted_code

# ---------------------------------------------------------------------------
# Main Processing Logic (Modified for Gap Filling)
# ---------------------------------------------------------------------------


def write_converted_file(output_file_path: str, content: str) -> None:
    """Writes content to the output file."""
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        f_out.write(content)


def handle_html_file(
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """Handles conversion of HTML files to React components with gap filling."""
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path, "error", error_msg="empty or unreadable file"
        )
        return

    try:
        converted_code = generate_react_component_from_html_template(
            file_content, file_path
        )
        if not converted_code:
            update_conversion_status(file_path, "error", error_msg="no initial code")
            return

        max_iterations = 3  # Limit iterations for gap filling
        iteration_count = 0

        while iteration_count < max_iterations:
            iteration_count += 1
            missing_logic = identify_missing_logic(file_content, converted_code)
            if not missing_logic:
                break  # No missing logic found, move on

            converted_code = improve_react_component_with_gap_filling(
                file_path, file_content, converted_code, missing_logic
            )

        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(
            output_dir, rel_path.replace(".html", ".js")
        )
        write_converted_file(output_file_path, converted_code)

        update_conversion_status(
            file_path,
            "success",
            output_file_path=output_file_path,
            iteration_count=iteration_count,
        )

    except Exception as exc:
        logging.error("Error processing HTML file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_module_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: Dict[str, str],
    force: bool,
) -> None:
    """Handles conversion of Angular module files."""
    # ... (Implementation remains similar, no direct gap analysis here yet)
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path, "error", error_msg="empty or unreadable file"
        )
        return
    try:
        react_module_code = convert_angular_module(file_content)
        if react_module_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_module_code)
            update_conversion_status(
                file_path, "success", output_file_path=output_file_path
            )
            module_name = os.path.basename(file_path).replace("-module.js", "")
            angular_modules[module_name] = file_content
    except Exception as exc:
        logging.error("Error processing module file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_controller_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: Dict[str, str],
    force: bool,
) -> None:
    """Handles conversion of Angular controller files."""
    # ... (Implementation remains similar)
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path, "error", error_msg="empty or unreadable file"
        )
        return
    try:
        ast = parse_js(file_path)
        if not ast:
            update_conversion_status(file_path, "error", error_msg="ast is empty")
            return
        dependencies = [
            dep["value"] for dep in ast.get("dependencies", []) if "value" in dep
        ]
        dependency_files_content = ""
        for dep in dependencies:
            if dep in angular_modules:
                dependency_files_content += angular_modules[dep] + "\n"
        react_component_code = generate_react_component_from_controller(
            file_content, dependency_files_content, ast
        )
        if react_component_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_component_code)
            update_conversion_status(
                file_path, "success", output_file_path=output_file_path
            )
    except Exception as exc:
        logging.error("Error processing controller file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_generic_js_file(
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """Handles copying over generic JS files."""
    # ... (Implementation remains the same)
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path, "error", error_msg="empty or unreadable file"
        )
        return
    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        write_converted_file(output_file_path, file_content)
        update_conversion_status(
            file_path, "success", output_file_path=output_file_path
        )
    except Exception as exc:
        logging.error("Error processing generic JS file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_other_file(
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """Handles copying over other file types."""
    # ... (Implementation remains the same)
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(
            file_path, "error", error_msg="empty or unreadable file"
        )
        return
    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        write_converted_file(output_file_path, file_content)
        update_conversion_status(
            file_path, "success", output_file_path=output_file_path
        )
    except Exception as exc:
        logging.error("Error processing file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def process_file(
    file_path: str,
    source_dir: str,
    output_dir: str,
    angular_modules: Dict[str, str],
    force: bool = False,
) -> None:
    """Routes files to the correct handler."""
    if file_path.endswith(".html"):
        handle_html_file(file_path, source_dir, output_dir, force)
    elif file_path.endswith(".js"):
        if is_angular_module_file(file_path):
            handle_angular_module_file(
                file_path, source_dir, output_dir, angular_modules, force
            )
        elif "controller" in file_path.lower():
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
        help="Path to the source directory containing AngularJS code.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for React code.",
    )
    parser.add_argument(
        "--force",
        help="Force re-conversion even if the file is already marked 'success'.",
        action="store_true",
    )
    args = parser.parse_args()

    init_db()
    angular_modules: Dict[str, str] = {}
    file_paths = []
    for root, _, files in os.walk(args.source_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if full_path.endswith(".html") or full_path.endswith(".js"):
                file_paths.append(full_path)

    for file_path in tqdm(file_paths, desc="Converting files"):
        tqdm.write(f"Processing: {file_path}")
        process_file(
            file_path,
            args.source_dir,
            args.output_dir,
            angular_modules,
            force=args.force,
        )

    logging.info("Conversion complete. See '%s' for details.", db_name)

if __name__ == "__main__":
    main()
