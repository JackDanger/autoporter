#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    python migrate_angular_to_react.py --source_dir <SOURCE_DIR> --output_dir <OUTPUT_DIR> [--force]

Description:
    This script converts a legacy AngularJS application to a React application,
    storing intermediate states in an SQLite database, providing a progress bar
    for file processing, and allowing interruption/resumption.
"""

import os
import ast
import sqlite3
import argparse
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------------------------
# LLM Infer Function (Placeholder)
# ---------------------------------------------------------------------------
def infer(prompt: str) -> str:
    """
    This is a placeholder for the LLM interaction.
    Replace this with your actual LLM API call.
    """
    # Example: Replace with your real LLM call
    return f"LLM Response for prompt: {prompt}"


# ---------------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------------
DB_NAME = "migration_state.db"


def init_db() -> None:
    """
    Initialize the SQLite database if it doesn't already exist.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS converted_files (
                file_path TEXT PRIMARY KEY,
                status TEXT,
                output_file_path TEXT,
                error_msg TEXT
            )
            """
        )
    finally:
        conn.close()


def get_conversion_status(file_path: str):
    """
    Return a tuple of (status, output_file_path, error_msg) for the given
    file_path, or None if it doesn't exist in the DB.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT status, output_file_path, error_msg
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
    error_msg: str = None
) -> None:
    """
    Insert or update the conversion status for a given file_path.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            INSERT INTO converted_files
            (file_path, status, output_file_path, error_msg)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path)
            DO UPDATE SET
              status=excluded.status,
              output_file_path=excluded.output_file_path,
              error_msg=excluded.error_msg
            """,
            (file_path, status, output_file_path, error_msg),
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


def parse_js(js_content: str):
    """
    Parses JavaScript content into an AST.
    """
    try:
        return ast.parse(js_content)
    except SyntaxError as err:
        logging.error("SyntaxError parsing JS: %s", err)
        return None
    except Exception as exc:
        logging.error("Error parsing JS: %s", exc)
        return None


def is_angular_module_file(file_path: str) -> bool:
    """
    Simple check for Angular module files, e.g. ends with '-module.js'.
    """
    return file_path.endswith('-module.js')


def extract_file_content(file_path: str) -> str:
    """
    Reads file and returns its content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            return file_obj.read()
    except Exception as exc:
        logging.error("Error reading file %s: %s", file_path, exc)
        return None


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------
def generate_react_component_from_controller(
    controller_content: str, dependencies: str
) -> str:
    """
    Converts an Angular controller to a React component using the LLM.
    """
    prompt = f"""
You are an expert JavaScript developer skilled at converting AngularJS components to ReactJS.
You'll be provided with an AngularJS controller and relevant dependencies.
Convert the following AngularJS controller to a functional React component that uses hooks.
Follow these rules:
1. Convert $scope properties to useState/useRef.
2. Convert Angular services to plain JS objects with import statements.
3. Convert Angular event listeners to React equivalents.
4. Convert Angular dependency injection to props or import statements.
5. Use functional components with React Hooks for state and lifecycle.
6. Use JS (no TypeScript).
7. Use proper JS import statements for all dependencies.
8. For shared state, recommend React Context.
9. Comment all code for clarity.
10. Return only the converted React component code (with imports).
11. Translate any 'this' references appropriately.
12. No external state management library; if state is shared, mention createContext.
13. Output in ES6 JavaScript (no TypeScript).

AngularJS Controller:
```javascript
{controller_content}
```

Dependencies:
```javascript
{dependencies}
```
    """
    return infer(prompt)


def generate_react_component_from_html_template(template_content: str) -> str:
    """
    Converts an HTML template to a React component using the LLM.
    """
    prompt = f"""
You are an expert JavaScript developer skilled at converting AngularJS HTML templates to ReactJS.
Convert the following AngularJS HTML template to a React component:
1. Translate angular directives (e.g. ng-show, ng-repeat) to React equivalents.
2. Use JSX syntax.
3. Use descriptive classNames.
4. Return a valid React component with necessary import statements.
5. All attributes must be valid React attributes.
6. Do not assume a styling library; inline if needed.
7. Output in ES6 JS, not TypeScript.

HTML Template:
```html
{template_content}
```
    """
    return infer(prompt)


def convert_angular_module(module_content: str) -> str:
    """
    Converts an Angular module to a React equivalent or plain JS file using the LLM.
    """
    prompt = f"""
You are an expert JavaScript developer skilled at converting AngularJS modules to ReactJS.
Convert the following AngularJS module to either a React context or a simple JS file:
1. Translate AngularJS services/values to JS objects or contexts.
2. Use ES6 import and export statements.
3. If services are used throughout, convert them into a React Context.
4. Include explanatory comments.
5. Output ES6 JavaScript (no TypeScript).

AngularJS Module:
```javascript
{module_content}
```
    """
    return infer(prompt)


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
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """
    Handle conversion of HTML files to React components.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="Empty or unreadable file")
        return

    try:
        react_component_code = generate_react_component_from_html_template(file_content)
        if react_component_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(
                output_dir, rel_path.replace('.html', '.js')
            )
            write_converted_file(output_file_path, react_component_code)
            update_conversion_status(file_path, "success",
                                     output_file_path=output_file_path)
    except Exception as exc:
        logging.error("Error processing HTML file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_module_file(
    file_path: str, source_dir: str, output_dir: str,
    angular_modules: dict, force: bool
) -> None:
    """
    Handle conversion of Angular module files to React equivalents.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="Empty or unreadable file")
        return

    try:
        react_module_code = convert_angular_module(file_content)
        if react_module_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_module_code)
            update_conversion_status(file_path, "success",
                                     output_file_path=output_file_path)
            # Store the original module content for potential dependencies
            module_name = os.path.basename(file_path).replace('-module.js', '')
            angular_modules[module_name] = file_content
    except Exception as exc:
        logging.error("Error processing module file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_controller_file(
    file_path: str, source_dir: str, output_dir: str,
    angular_modules: dict, force: bool
) -> None:
    """
    Handle conversion of Angular controller files to React components.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="Empty or unreadable file")
        return

    try:
        tree = parse_js(file_content)
        dependencies = set()
        if tree:
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and
                        isinstance(node.func, ast.Attribute) and
                        node.func.attr == 'module'):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            dependencies.add(arg.value)

        # Gather content from relevant modules
        dependency_files_content = ""
        for dep in dependencies:
            if dep in angular_modules:
                dependency_files_content += angular_modules[dep] + "\n"

        # Convert Angular controller
        react_component_code = generate_react_component_from_controller(
            file_content, dependency_files_content
        )
        if react_component_code:
            rel_path = os.path.relpath(file_path, source_dir)
            output_file_path = os.path.join(output_dir, rel_path)
            write_converted_file(output_file_path, react_component_code)
            update_conversion_status(file_path, "success",
                                     output_file_path=output_file_path)
    except Exception as exc:
        logging.error("Error processing controller file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_generic_js_file(
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """
    Handle copying over generic JS files that don't match Angular patterns.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="Empty or unreadable file")
        return

    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        write_converted_file(output_file_path, file_content)
        update_conversion_status(file_path, "success",
                                 output_file_path=output_file_path)
    except Exception as exc:
        logging.error("Error processing generic JS file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_other_file(
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """
    Handle copying over other file types unchanged.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="Empty or unreadable file")
        return

    try:
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        write_converted_file(output_file_path, file_content)
        update_conversion_status(file_path, "success",
                                 output_file_path=output_file_path)
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
    Route files to the correct handler based on file extension and Angular usage.
    """
    if file_path.endswith('.html'):
        handle_html_file(file_path, source_dir, output_dir, force)
    elif file_path.endswith('.js'):
        if is_angular_module_file(file_path):
            handle_angular_module_file(
                file_path, source_dir, output_dir, angular_modules, force
            )
        elif 'controller' in file_path:
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
        help="Force re-conversion even if the file is already in the DB as success",
        action="store_true"
    )
    args = parser.parse_args()

    # 1. Initialize DB
    init_db()

    # 2. Prepare to store Angular modules, etc.
    angular_modules = {}

    # 3. Gather all files from the source directory
    file_paths = []
    for root, _, files in os.walk(args.source_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            # Skip our own script or the DB file
            if full_path.endswith('.html'):
                file_paths.append(full_path)
            elif full_path.endswith('.js'):
                file_paths.append(full_path)
            else:
                continue

    # 4. Process each file with a progress bar
    for file_path in tqdm(file_paths, desc="Converting Files"):
        process_file(
            file_path,
            args.source_dir,
            args.output_dir,
            angular_modules,
            force=args.force
        )

    # 5. Done
    logging.info("Conversion complete. See '%s' for details.", DB_NAME)


if __name__ == "__main__":
    main()
