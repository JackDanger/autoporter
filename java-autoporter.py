#!/usr/bin/env python3

"""
autoporter.py

A revolutionary, single-file script that ports legacy Java codebases directly to modern Python,
leveraging an LLM to produce beautiful, fully-documented, and test-covered Python code.

Features:
- Uses FastAPI, SQLAlchemy, Alembic migrations, and pytest as the standard stack.
- Automatically translates Java classes (models, services, controllers, etc.) into Python equivalents.
- Updates and corrects comments and docstrings to follow Python best practices.
- Generates thoughtful and comprehensive unit tests.
- Manages database migrations and models through SQLAlchemy and Alembic.
- Splits the process into well-defined steps, feeding the LLM with rich context and instructions.
- Uses a single environment variable `OPENAI_API_KEY` to authenticate with the OpenAI API.

Prerequisites:
- Python 3.9+
- openai (pip install openai)
- pydantic (for validation and prompt structuring)
- tqdm (for progress bars: pip install tqdm)
- A directory of legacy Java code that you want to convert.

Usage:
    export OPENAI_API_KEY="sk-...your-token..."
    python autoporter.py --input legacy_java_src --output new_python_app
"""

import os
import sys
import argparse
from pathlib import Path
import google.generativeai as genai
from openai import OpenAI
from typing import List, Dict
from pydantic import BaseModel
from tqdm import tqdm

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel(model_name='gemini-1.5-flash-8b')

# ------------------------------------------------------------------------------------
# Prompt engineering and data structures
# ------------------------------------------------------------------------------------


class FileContext(BaseModel):
    relative_path: str
    content: str


class TransformStep(BaseModel):
    name: str
    description: str
    instructions: str


# ------------------------------------------------------------------------------------
# Prompt templates and instructions
# ------------------------------------------------------------------------------------

INTRO_PROMPT = """You are a world-class expert in converting legacy Java codebases to modern Python web backends using FastAPI, SQLAlchemy, and Alembic for migrations. You also produce comprehensive unit tests using pytest, and ensure that all code is fully documented, idiomatic, and clean. You use Python best practices, type hints, and docstrings in Google style.

You will be given a list of files from a legacy Java codebase and asked to:
1. Convert these files into a single coherent Python FastAPI application using SQLAlchemy models and Alembic migrations.
2. Ensure that the application structure follows a modern, well-organized layout:
    - `app/` directory containing `main.py` (FastAPI startup), `models.py` (SQLAlchemy models), `schemas.py` (Pydantic schemas), `routers/`, etc.
    - `tests/` directory containing pytest-based tests for all major components.
    - `alembic/` directory for migrations.
3. Port logic from Java classes (controllers, services, DAOs, entities, etc.) into Python equivalents with proper layering. For instance:
    - Java entities and DTOs become SQLAlchemy models and Pydantic schemas.
    - Java controllers become FastAPI routers.
    - Java services become Python modules with business logic.
4. Update and improve comments, docstrings, and overall code clarity. If the original code has unclear logic or poor structure, improve it gracefully.
5. Generate a requirements.txt or pyproject.toml if needed.
6. Add Alembic migrations as needed.
7. Provide at least one comprehensive pytest test file that covers key aspects of the application.
8. After completing the conversion, output a structured plan and the final Python files.

You may receive multiple steps and instructions. At the end, you will produce:
- A directory structure with all the needed Python files.
- A summary of changes and improvements.

Follow best practices strictly. Utilize Python 3.9+ features and type hints.
"""

TRANSFORM_PROMPT_TEMPLATE = """You are given a set of Java source files and their contents. You have already read the instructions above.
Now, analyze the provided files and produce a modern Python FastAPI application as described.

Below are the Java files:

{file_list}

Step-by-step, do the following:
1. Understand the domain model from the Java files.
2. Identify all entities, services, and controllers.
3. Devise a Python module structure (app/main.py, app/models.py, app/schemas.py, app/routers/*, etc.).
4. Convert Java entities to SQLAlchemy models and Alembic migrations.
5. Convert Java controllers to FastAPI routers.
6. Convert services and utilities into Python modules and classes.
7. Create Pydantic schemas for request/response models.
8. Write a sample Alembic migration script based on the discovered models.
9. Generate unit tests using pytest in a `tests/` directory.
10. Include docstrings and comments that explain what the code does. Be clear and Pythonic.
11. Generate a requirements.txt or pyproject.toml file including necessary dependencies.
12. Finally, provide a structured output with all the resulting files as a directory tree and their contents.

Make sure the resulting code is self-contained, understandable, and runs as a standalone Python application once dependencies are installed and Alembic migrations are applied.
"""

# ------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------


def gather_java_files(input_dir: Path) -> List[FileContext]:
    """Recursively gather all .java files from the given directory."""
    print("Step 1: Gathering Java files...")
    java_files = []
    all_files = list(input_dir.rglob("*.java"))
    for f in tqdm(all_files, desc="Reading Java files", unit="file"):
        with f.open("r", encoding="utf-8") as file:
            content = file.read()
        rel_path = str(f.relative_to(input_dir))
        java_files.append(FileContext(relative_path=rel_path, content=content))
    return java_files


def call_openai_chat_completion(messages: List[Dict[str, str]], model: str = "gpt-4", temperature: float = 0.0, max_tokens: int = 8000) -> str:
    """Call the OpenAI Chat API and return the response content."""
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Since this is a single request that may take some time, we can show a simple message:
    print("Contacting the LLM to transform your code... Please wait.")

    response = openai_client.chat.completions.create(model=model,
                                                     messages=messages,
                                                     temperature=temperature,
                                                     max_tokens=max_tokens)
    return response.choices[0].message.content


def call_gemini(prompt):
    max_retries = 15
    retry_delay = 1  # Start with 1-second delay

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.8,
                ),
                stream=True,
            )
            chunks = ""
            for chunk in response:
                print(chunk.text, end='', flush=True)
                chunks += chunk.text
            return chunks
        except Exception as e:
            print_step(
                f"Error generating response: {e}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print_step("Failed to get a valid response from the Google PaLM API.")
    return ''


def transform_code(java_files: List[FileContext]) -> str:
    """Send the Java files to the LLM and get back the Python code."""
    print("Step 2: Transforming code with the LLM...")
    file_list_str = ""
    for jf in java_files:
        file_list_str += f"File: {jf.relative_path}\n```\n{jf.content}\n```\n\n"

    prompt = TRANSFORM_PROMPT_TEMPLATE.format(file_list=file_list_str)
    output = call_llm(prompt)
    return output


def write_output_structure(output_str: str, output_dir: Path) -> None:
    """
    The LLM output is expected to contain a structured representation of all files,
    possibly as a directory tree. We will parse that structure and write files accordingly.
    """
    print("Step 3: Writing output files...")

    lines = output_str.splitlines()
    current_file = None
    current_content = []
    file_map = {}  # filename -> content

    # Parse the LLM output to extract files
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<BEGIN FILE:"):
            # Save current if any
            if current_file is not None:
                file_map[current_file] = "\n".join(current_content)
                current_content = []
            # Extract filename
            filename = stripped[len("<BEGIN FILE:"):].rstrip(">").strip()
            current_file = filename
            current_content = []
        elif stripped.startswith("<END FILE>"):
            if current_file is not None:
                file_map[current_file] = "\n".join(current_content)
                current_file = None
                current_content = []
        else:
            if current_file is not None:
                current_content.append(line)

    # If there's a dangling file
    if current_file is not None:
        file_map[current_file] = "\n".join(current_content)

    # Write files with a progress bar
    file_items = list(file_map.items())
    for fname, fcontent in tqdm(file_items, desc="Writing files", unit="file"):
        fpath = output_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with fpath.open("w", encoding="utf-8") as f:
            f.write(fcontent)


def call_llm(prompt):
    if openai_client.api_key is not None:
        messages = [
            {"role": "system", "content": INTRO_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_openai_chat_completion(messages)
    else:
        return call_gemini(f"{INTRO_PROMPT}\n\n{prompt}")


def main():
    parser = argparse.ArgumentParser(description="Auto-port a legacy Java codebase to Python (FastAPI + SQLAlchemy) using LLMs.")
    parser.add_argument("--input", required=True, help="Path to the input directory containing Java source code.")
    parser.add_argument("--output", required=True, help="Path to output the new Python application.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    java_files = gather_java_files(input_dir)
    if not java_files:
        print(f"No Java files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    output_str = transform_code(java_files)
    write_output_structure(output_str, output_dir)

    print(f"Done! Python application is now in {output_dir}.")


if __name__ == "__main__":
    main()
