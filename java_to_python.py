#!/usr/bin/env python3

"""
A script to automatically convert a legacy Java application into a modern Python web application,
complete with unit tests, a README, and best-practice structures (FastAPI, SQLAlchemy, Alembic).

Usage:
    python convert_app.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key.

Requirements:
    pip install openai
"""

import os
import sys
import re
import time
from typing import List, Dict
import google.generativeai as genai
from openai import OpenAI

OPENAI_MODEL = "o1-preview"
GEMINI_MODEL = 'gemini-2.0-flash-exp'
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)


INTRO_PROMPT = """You are a world-class expert in converting legacy Java codebases to modern Python web backends using FastAPI, SQLAlchemy, and Alembic for migrations.
You also produce comprehensive unit tests using pytest, and ensure that all code is fully documented, idiomatic, and clean.
You use Python best practices, type hints, and docstrings in Google style.

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


def gather_java_files_content(start_path):
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


def create_prompt(java_files_str):
    """
    Creates a carefully crafted prompt for the LLM, instructing it to:
    - Convert the Java code into a modern Python web app using FastAPI, SQLAlchemy, Alembic.
    - Use a blueprint/module pattern.
    - Provide a README, requirements.txt, and at least one test in the tests/ folder.
    - Follow best practices and PEP8.
    """
    initial_instructions = (
        "Below is the entire source code of a legacy Java application. "
        "You are a veteran application migration engineer and a principal-level Python developer.\n\n"
        "Carefully analyze the Java code and produce a fully modern, production-grade Python web application.\n"
        "Use these technologies and patterns:\n"
        "- FastAPI for the web framework.\n"
        "- SQLAlchemy for database interactions.\n"
        "- Alembic for migrations.\n"
        "- A blueprint or module-based project layout.\n\n"
        "Additionally, please:\n"
        "- Remove unnecessary Java ceremony.\n"
        "- Include a README.md with instructions on how to install, run, and test the application.\n"
        "- Include a requirements.txt listing any needed Python libraries.\n"
        "- Provide a unit test suite in a tests/ directory (at least one test file is required).\n"
        "- Ensure the Python code is PEP8-compliant.\n"
        "- Use type hints, docstrings, and follow Python best practices.\n"
        "Avoid partial or incomplete files; produce only fully working code.\n\n"
        "Here is the Java source:\n\n"
    )

    final_instructions = (
        "\n\n"
        "Now, convert all of this Java code into a Python codebase. \n"
        "Include all necessary Python files with correct filenames. \n"
        "Separate files with the comment pattern:\n"
        "`# filename: path/to/file.py`\n\n"
        "Your response should be a self-contained solution with:\n"
        "1. The main FastAPI application.\n"
        "2. A README.md.\n"
        "3. A requirements.txt.\n"
        "4. A tests/ folder containing at least one test file.\n"
        "5. Alembic migration scripts or placeholders for them.\n\n"
        "Make sure the code runs as-is when placed in the correct file structure.\n"
        "Thank you!"
    )

    return initial_instructions + java_files_str + final_instructions


def call_llm(prompt):
    if openai_client.api_key:
        messages = [
            {"role": "system", "content": INTRO_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_openai_chat_completion(messages)
    else:
        return call_gemini(f"{INTRO_PROMPT}\n\n{prompt}")


def call_openai_chat_completion(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 8000) -> str:
    """Call the OpenAI Chat API and return the response content."""
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Since this is a single request that may take some time, we can show a simple message:
    print("Contacting the LLM to transform your code... Please wait.")

    response = openai_client.chat.completions.create(model=OPENAI_MODEL,
                                                     messages=messages,
                                                     temperature=temperature,
                                                     max_tokens=max_tokens)
    return response.choices[0].message.content


def call_gemini(prompt):
    max_retries = 15
    retry_delay = 1  # Start with 1-second delay

    for attempt in range(max_retries):
        try:
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
            print_step(
                f"Error generating response: {e}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print_step("Failed to get a valid response from the Google PaLM API.")
    return ''


def split_and_write_files(llm_output, output_dir):
    """
    Splits the LLM output based on `# filename: some_path.py` lines,
    then writes each file to the correct path under output_dir.
    """
    lines = llm_output.splitlines()
    current_file = None
    content = []

    def write_to_file(path, lines_content):
        out_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write("".join(lines_content))
        print(f"[INFO] Written to {out_path}")

    file_pattern = re.compile(r"^# filename:\s+[\w.\-/\\]+\.py$|^# filename:\s+[\w.\-/\\]+\.md$|^# filename:\s+[\w.\-/\\]+\.txt$")

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


def print_step(step):
    print(f"[Step] {step}")


def main():
    """
    Main driver function:
    1. Parse arguments.
    2. Gather Java files content.
    3. Create the final prompt.
    4. Call the LLM once to get the new Python code.
    5. Split and write the files into the output directory.
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

    # Gather all .java files
    java_files_str = gather_java_files_content(input_dir)

    # Create the prompt
    prompt = create_prompt(java_files_str)
    print(prompt)

    # Call the LLM
    llm_output = call_llm(prompt)
    print(llm_output)

    # Write the results to the output directory
    split_and_write_files(llm_output, output_dir)


if __name__ == "__main__":
    main()
