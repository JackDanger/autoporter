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

import os
import re
import sqlite3
import argparse
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString
import subprocess
import json

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
db_name = "migration_state.db"


def init_db() -> None:
    """
    Initialize the SQLite database if it doesn't already exist.
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
                iteration_count INTEGER DEFAULT 0
            )
            """
        )
    finally:
        conn.close()


def get_conversion_status(file_path: str):
    """
    Return a tuple of (status, output_file_path, error_msg, iteration_count) for the given
    file_path, or None if it doesn't exist in the DB.
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
    iteration_count:int = 0
) -> None:
    """
    Insert or update the conversion status for a given file_path.
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
              iteration_count = excluded.iteration_count
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


def parse_js(js_file: str) -> dict:
    """
    Parses JavaScript content using esprima via subprocess.
    """
    try:
        result = subprocess.run(
            ["node", "parse_js_esprima.js", js_file],
            capture_output=True, text=True, check=True
        )
        if result.returncode == 0:
          return json.loads(result.stdout)
        else:
          logging.error("Error parsing Javascript with esprima, stderr: %s", result.stderr)
          return None
    except FileNotFoundError:
        logging.error("Error: parse_js_esprima.js not found. Make sure it's in the same directory")
        return None
    except Exception as e:
        logging.error(f"Error parsing JavaScript: {e}")
        return None


def is_angular_module_file(file_path: str) -> bool:
    """
    Simple check for Angular module files, e.g., ends with '-module.js'.
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
    controller_content: str, dependencies: str, ast:dict
) -> str:
    """
    Converts an Angular controller to a React component using the LLM.
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS components to ReactJS.\n"
        "You'll be provided with an AngularJS controller, relevant dependencies, and a parsed AST tree. \n"
        "Convert the following AngularJS controller to a functional React component that uses hooks.\n"
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
        "13. Output in ES6 JavaScript (no TypeScript).\n"
        "\n"
        "AngularJS Controller:\n"
        "```javascript\n"
        f"{controller_content}\n"
        "```\n"
        "\n"
         "Javascript AST:\n"
        "```json\n"
        f"{json.dumps(ast)}\n"
        "```\n"
        "\n"
        "Dependencies:\n"
        "```javascript\n"
        f"{dependencies}\n"
        "```\n"
    )
    return infer(prompt)


def generate_react_component_from_html_template(template_content: str, file_path:str) -> str:
    """
    Converts an HTML template to a React component using the LLM.
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS HTML templates to ReactJS.\n"
        "Convert the following AngularJS HTML template to a React component.\n"
        "Follow these rules:\n"
        "1. Translate any Angular-specific directives (e.g., ng-show, ng-repeat, ng-if, ng-click, ng-model)\n"
        "   to their React equivalents using React’s JSX syntax and React’s hooks (e.g., useState, conditional rendering, etc.).\n"
        "2. Use JSX syntax.\n"
        "3. Use descriptive class names.\n"
        "4. Return a valid functional React component with necessary import statements, making sure that any state used is managed using hooks.\n"
        "5. Convert Angular event handlers like ng-click into React’s onClick syntax, ensuring that events are handled correctly within the React Component.\n"
        "6. Convert any binding like {{property}} to JSX syntax.  Make sure to handle both one-way and two-way binding.\n"
        "7. All attributes must be valid React attributes.\n"
        "8. Do not assume any particular styling library; use inline styles only if necessary.\n"
        "9. Handle conditional rendering with JSX ternary operator or logical &&.\n"
        "10. Use Fragments (<></>) when needed to avoid extra divs.\n"
        "11. Output in ES6 JavaScript, not TypeScript.\n"
        "12. If the HTML template is a gridCellTemplate, create a separate React component that can be used as a template, and then import it into the parent component using props.\n"
        "13. If the HTML template contains a table, make sure the html table is rendered properly with proper headers and cells.  Use map if needed to dynamically render table rows.\n"
        "14. If the file_path contains gridHeaderTemplate, then make sure to generate JSX for the table headers and not the whole table.\n"
        "\n"
        "HTML Template:\n"
        "```html\n"
        f"{template_content}\n"
        "```\n"
    )
    try:
      response =  infer(prompt)
      # Remove surrounding backticks if present
      response = response.strip('`')
      response = response.replace('```javascript','')
      response = response.replace('```','')
      return response
    except Exception as e:
      logging.error("error generating react component from html template: %s", e)
      return None

def convert_angular_module(module_content: str) -> str:
    """
    Converts an Angular module to a React equivalent or plain JS file using the LLM.
    """
    prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS modules to ReactJS.\n"
        "Convert the following AngularJS module to either a React context or a simple JS file:\n"
        "1. Translate AngularJS services/values to JS objects or contexts.\n"
        "2. Use ES6 import and export statements.\n"
        "3. If services are used throughout, convert them into a React context.\n"
        "4. Include explanatory comments.\n"
        "5. Output ES6 JavaScript (no TypeScript).\n"
        "\n"
        "AngularJS Module:\n"
        "```javascript\n"
        f"{module_content}\n"
        "```\n"
    )
    return infer(prompt)


# ---------------------------------------------------------------------------
# Feedback Loop Helper
# ---------------------------------------------------------------------------

def score_react_component(component_code: str) -> int:
  """Scores the React component based on common mistakes"""
  score = 0
  if not component_code:
    return 0 # no score if empty
  
  # common errors
  if 'ng-show' in component_code:
     score -=1
  if 'ng-if' in component_code:
     score -=1
  if 'ng-repeat' in component_code:
     score -=1
  if 'ng-click' in component_code:
    score -=1
  if 'ng-model' in component_code:
    score -=1
  if '{{' in component_code:
    score -=1
  if 'className=' not in component_code:
    score -=1

  if 'useState(' in component_code:
    score+=1

  return score



def improve_react_component(file_path: str, original_content: str, converted_code: str, iteration: int) -> str:
  """Improves a react component via LLM feedback"""
  prompt = (
        "You are an expert JavaScript developer skilled at converting AngularJS to ReactJS.\n"
        "You'll be provided with an original AngularJS HTML template, and the current converted ReactJS component. \n"
        "You are to review the converted React component, identify any mistakes, fix them, and produce a perfect React conversion.\n"
        "Follow these rules:\n"
        "1.  Translate any Angular-specific directives (e.g., ng-show, ng-repeat, ng-if, ng-click, ng-model)\n"
        "   to their React equivalents using React’s JSX syntax and React’s hooks (e.g., useState, conditional rendering, etc.).\n"
         "2. Use JSX syntax.\n"
        "3. Use descriptive class names.\n"
        "4. Return a valid functional React component with necessary import statements, making sure that any state used is managed using hooks.\n"
        "5. Convert Angular event handlers like ng-click into React’s onClick syntax, ensuring that events are handled correctly within the React Component.\n"
        "6. Convert any binding like {{property}} to JSX syntax.  Make sure to handle both one-way and two-way binding.\n"
        "7. All attributes must be valid React attributes.\n"
        "8. Do not assume any particular styling library; use inline styles only if necessary.\n"
         "9. Handle conditional rendering with JSX ternary operator or logical &&.\n"
        "10. Use Fragments (<></>) when needed to avoid extra divs.\n"
        "11. Output in ES6 JavaScript, not TypeScript.\n"
        "12. If the HTML template is a gridCellTemplate, create a separate React component that can be used as a template, and then import it into the parent component using props.\n"
        "13. If the HTML template contains a table, make sure the html table is rendered properly with proper headers and cells.  Use map if needed to dynamically render table rows.\n"
        "14. If the file_path contains gridHeaderTemplate, then make sure to generate JSX for the table headers and not the whole table.\n"
        "15. Analyze the React component, and fix any syntax errors, remove any extraneous code or comments, and remove any usage of Angular within the React component.\n"
          "16. Only return the ReactJS component code, with all import statements.\n"
        "\n"
        f"Original AngularJS HTML Template:\n ```html\n{original_content}\n```"
        f"\n\nCurrent React Component:\n ```javascript\n{converted_code}\n```"
        f"\n\n Iteration {iteration} \n"
    )
  try:
        response =  infer(prompt)
        # Remove surrounding backticks if present
        response = response.strip('`')
        response = response.replace('```javascript','')
        response = response.replace('```','')
        return response
  except Exception as e:
    logging.error(f"error improving react component: {e}")
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
    file_path: str, source_dir: str, output_dir: str, force: bool
) -> None:
    """
    Handles conversion of HTML files to React components.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="empty or unreadable file")
        return

    try:
        
        converted_code = generate_react_component_from_html_template(file_content, file_path)
        if not converted_code:
            update_conversion_status(file_path, "error", error_msg="no initial code")
            return
        
        
        
        # Improve via feedback loop
        max_iterations = 5
        score = score_react_component(converted_code)
        iteration_count = 0
        
        while score < 0 and iteration_count < max_iterations:
          iteration_count+=1
          converted_code = improve_react_component(file_path, file_content, converted_code, iteration_count)
          score = score_react_component(converted_code)
        
        
        rel_path = os.path.relpath(file_path, source_dir)
        output_file_path = os.path.join(
            output_dir, rel_path.replace('.html', '.js')
        )
        write_converted_file(output_file_path, converted_code)
        update_conversion_status(file_path, "success",
                                  output_file_path=output_file_path, iteration_count = iteration_count)
    except Exception as exc:
        logging.error("Error processing HTML file %s: %s", file_path, exc)
        update_conversion_status(file_path, "error", error_msg=str(exc))


def handle_angular_module_file(
    file_path: str, source_dir: str, output_dir: str,
    angular_modules: dict, force: bool
) -> None:
    """
    Handles conversion of Angular module files to React equivalents.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="empty or unreadable file")
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
    Handles conversion of Angular controller files to React components.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return
    
    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="empty or unreadable file")
        return

    try:
        ast = parse_js(file_path)
        if not ast:
            update_conversion_status(file_path, "error", error_msg="ast is empty")
            return
        
        dependencies = [dep['value'] for dep in ast.get('dependencies', [])]


        # Gather content from relevant modules
        dependency_files_content = ""
        for dep in dependencies:
            if dep in angular_modules:
                dependency_files_content += angular_modules[dep] + "\n"

        # Convert Angular controller
        react_component_code = generate_react_component_from_controller(
            file_content, dependency_files_content, ast
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
    Handles copying over generic JS files that don't match Angular patterns.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="empty or unreadable file")
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
    Handles copying over other file types unchanged.
    """
    record = get_conversion_status(file_path)
    if record and record[0] == "success" and not force:
        return

    file_content = extract_file_content(file_path)
    if not file_content:
        update_conversion_status(file_path, "error", error_msg="empty or unreadable file")
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
    Routes files to the correct handler based on file extension and Angular usage.
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
    for file_path in tqdm(file_paths, desc="Converting files"):
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
