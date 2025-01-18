#!/usr/bin/env python3

"""
A script to automatically convert a legacy Java application into a modern Python web application
and then perform iterative review/fix passes to close gaps in HTTP endpoints, database schemas,
and business logic.

Usage:
    python convert_app.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key. (Required to use the OpenAI LLM pass)
    GEMINI_API_KEY: Your Google PaLM API key. (Used if OPENAI_API_KEY is not set)

Requirements:
    pip install openai google-generativeai
"""

import os
import sys
import re
import time
from typing import List, Dict
import google.generativeai as genai
from openai import OpenAI

# -------------------------------------------------------------------------
# LLM / Model Config
# -------------------------------------------------------------------------
OPENAI_MODEL = "o1-preview"
GEMINI_MODEL = 'gemini-2.0-flash-exp'

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)

# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------

# High-level system instructions for converting Java -> Python
INTRO_PROMPT = """You are a world-class expert in converting legacy Java codebases to modern Python web backends using FastAPI, SQLAlchemy, and Alembic for migrations.
You also produce comprehensive unit tests using pytest, and ensure that all code is fully documented, idiomatic, and clean.
You use Python best practices, type hints, and docstrings in Google style.

You will be given a list of files from a legacy Java codebase and asked to:
1. Convert these files into a single coherent Python FastAPI application using SQLAlchemy models and Alembic migrations.
2. Ensure that the application structure follows a modern, well-organized layout:
    - `app/` directory containing `main.py` (FastAPI startup), `models.py` (SQLAlchemy models), `schemas.py` (Pydantic schemas), `routers/`, etc.
    - `tests/` directory containing pytest-based tests for all major components.
    - `alembic/` directory for migrations.
3. Port logic from Java classes (controllers, services, DAOs, entities, etc.) into Python equivalents with proper layering.
4. Update and improve comments, docstrings, and overall code clarity.
5. Generate a requirements.txt or pyproject.toml if needed.
6. Add Alembic migrations as needed.
7. Provide a comprehensive pytest test file that covers key aspects of the application.
8. At the end, produce a directory structure with all the needed Python files plus a summary of changes and improvements.

Follow best practices strictly. Utilize Python 3.9+ features and type hints.
"""

# Template for transforming Java -> Python
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

# Prompt for iterative passes that look for missing or incorrect pieces
REVIEW_PROMPT_TEMPLATE = """You are given the Python code for a newly migrated application.
Now, carefully review and improve it with respect to potential gaps or mistakes in:
1. Missing or incorrect {category}.

Provide updated files if anything requires changing or adding.
Follow best practices in FastAPI, SQLAlchemy, migrations, and business logic.
If something is already correct, keep it as is.

Output your revised code using the pattern:
# filename: ...
<code here>

Include only the updated or new content for each file. If a file needs no changes, provide it anyway for completeness.
"""


# -------------------------------------------------------------------------
# LLM Utility Functions
# -------------------------------------------------------------------------
def call_llm_system_user(system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=8000) -> str:
    """
    Calls the LLM with a system prompt and a user prompt using OpenAI if available,
    otherwise uses Gemini.
    """
    # If we have an OpenAI key, use OpenAI
    if openai_client.api_key:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_openai_chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
    else:
        # Otherwise, fallback to Gemini PaLM
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return call_gemini(combined_prompt)


def call_openai_chat_completion(
    messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 8000
) -> str:
    """Call the OpenAI Chat API and return the response content."""
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    print("[INFO] Contacting OpenAI Chat Completion API... Please wait.")
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_gemini(prompt):
    """Call the Gemini (Google PaLM) API for generation."""
    max_retries = 15
    retry_delay = 1  # Start with 1-second delay

    for attempt in range(max_retries):
        try:
            print("[INFO] Contacting Google PaLM API (Gemini)... Please wait.")
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.8,
                ),
                stream=True,
            )
            chunks = ""
            for chunk in response:
                chunks += chunk.text
            return chunks
        except Exception as e:
            print(f"[WARN] Error generating response: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print("[ERROR] Failed to get a valid response from Google PaLM API.")
    return ""

# -------------------------------------------------------------------------
# File Handling Utilities
# -------------------------------------------------------------------------


def gather_java_files_content(start_path: str) -> str:
    """
    Recursively walks the input directory, reading the contents of each .java file
    and returning them in a single string, separated by markers indicating filename.
    """
    java_files_content = []
    for root, dirs, files in os.walk(start_path):
        for filename in files:
            if filename.endswith(".java"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                rel_path = os.path.relpath(full_path, start_path)
                java_files_content.append(f"# filename: {rel_path}\n{content}\n\n")
    return "".join(java_files_content)


def create_transform_prompt(java_files_str: str) -> str:
    """
    Creates a carefully crafted prompt for the LLM, instructing it to:
    - Convert the Java code into a modern Python web app using FastAPI, SQLAlchemy, Alembic.
    - Use a best-practice project structure.
    - Provide a README, requirements.txt, and at least one test in `tests/`.
    - Follow best practices and PEP8.
    """
    # We can simply wrap the input Java code in the TRANSFORM_PROMPT_TEMPLATE
    return TRANSFORM_PROMPT_TEMPLATE.format(file_list=java_files_str)


def split_and_write_files(llm_output: str, output_dir: str):
    """
    Splits the LLM output based on `# filename: some_path` lines,
    then writes each file to the correct path under output_dir.
    Supports .py, .md, .txt, etc.
    """
    lines = llm_output.splitlines()
    current_file = None
    content = []

    file_pattern = re.compile(r"^# filename:\s+[\w.\-/\\]+(\.py|\.md|\.txt)$")

    def write_to_file(path: str, lines_content: List[str]):
        out_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write("".join(lines_content))
        print(f"[INFO] Written to {out_path}")

    for line in lines:
        # Detect lines that start with '# filename:' followed by a valid path
        if line.startswith("# filename: ") and file_pattern.match(line):
            # If we were tracking another file, write it before starting a new one
            if current_file and content:
                write_to_file(current_file, content)
                content = []

            current_file = line[len("# filename: "):].strip()
        else:
            # If it's a line of content (and we have a current file) gather it
            if current_file is not None:
                content.append(line + "\n")

    # Write the last file's content if any remains
    if current_file and content:
        write_to_file(current_file, content)


def read_entire_python_code(base_dir: str) -> str:
    """
    Recursively read all .py (and optionally .md, .txt) files from the
    output directory to feed back into the LLM for the review steps.
    """
    aggregated_content = []
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            # Optionally read just .py or also .md, .txt. Here we read .py for clarity
            if filename.endswith((".py", ".md", ".txt")):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, base_dir)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                aggregated_content.append(f"# filename: {rel_path}\n{content}\n\n")
    return "".join(aggregated_content)


def perform_review_and_fix(
    base_dir: str,
    category: str,
    output_dir: str,
    temperature: float = 0.0,
    max_tokens: int = 4000
):
    """
    Perform a single pass of reviewing the code for either:
     - Missing/incorrect HTTP endpoints
     - Missing/incorrect DB schemas
     - Missing/incorrect business logic
    by prompting the LLM with the entire codebase as context.
    """
    # 1. Read the entire generated code
    codebase_str = read_entire_python_code(base_dir)
    # 2. Create the review prompt
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(category=category)
    combined_prompt = f"{review_prompt}\n\nHere is the current codebase:\n\n{codebase_str}"
    # 3. Call the LLM
    review_response = call_llm_system_user(INTRO_PROMPT, combined_prompt, temperature=temperature, max_tokens=max_tokens)

    if not review_response.strip():
        print(f"[WARN] No response from LLM for {category} pass. Skipping write.")
        return

    # 4. Split and write the updated files
    print(f"[INFO] Writing {category} pass changes into {output_dir}...")
    split_and_write_files(review_response, output_dir)


# -------------------------------------------------------------------------
# Main Script Flow
# -------------------------------------------------------------------------
def main():
    """
    Main driver function:
    1. Parse arguments.
    2. Gather Java files content.
    3. Create the prompt and call LLM to produce the initial Python code.
    4. Split & write the generated files into the output directory.
    5. Perform additional passes to review/fix missing or incorrect:
         a. HTTP endpoints
         b. Database schemas
         c. Business logic
    """
    if len(sys.argv) != 3:
        print("Usage: python convert_app.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1) Gather all .java files
    java_files_str = gather_java_files_content(input_dir)
    if not java_files_str.strip():
        print("[ERROR] No Java files found in input directory. Exiting.")
        sys.exit(1)

    # 2) Create the transformation prompt
    transform_prompt = create_transform_prompt(java_files_str)

    # 3) Call the LLM for the initial Java->Python conversion
    print("[INFO] Generating initial Python code from Java sources...")
    llm_output = call_llm_system_user(INTRO_PROMPT, transform_prompt, temperature=0.0, max_tokens=8000)

    if not llm_output.strip():
        print("[ERROR] LLM returned an empty response for the transformation step.")
        sys.exit(1)

    # 4) Split and write the files into the output directory
    print("[INFO] Writing initial pass files to output directory...")
    split_and_write_files(llm_output, output_dir)

    # 5) Additional passes:
    passes = [
        "HTTP endpoints",
        "database schemas",
        "business logic",
    ]
    for p in passes:
        print(f"\n[INFO] Starting review/fix pass for {p}...\n")
        perform_review_and_fix(
            base_dir=output_dir,
            category=p,
            output_dir=output_dir,
            temperature=0.0,
            max_tokens=4000
        )
        print(f"[INFO] Completed {p} pass.\n")

    print("[INFO] All passes completed successfully!")
    print("[INFO] Your updated Python codebase is located in:", output_dir)


if __name__ == "__main__":
    main()
