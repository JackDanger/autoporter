#!/usr/bin/env python3
"""
A conversion tool to transform legacy 2018-era .NET (C#) applications into a modern Python codebase
using FastAPI, SQLAlchemy, Alembic, pytest, and the blueprint architecture. In addition, it generates
a working Dockerfile.

This tool runs multiple conversion passes to capture:
  - Database models and migrations
  - HTTP endpoints and controllers
  - Business logic including authentication, credentials, HTML/text templating, email delivery,
    service-to-service calls, and database queries
  - Comprehensive unit tests (pytest)
  - Docker containerization

High-level strategies include:
  - Building an advanced IR (with rudimentary dependency graph analysis)
  - Iterative multi-pass LLM conversion (each pass focusing on one area)
  - Chunking large inputs to avoid token limits
  - Merging and consolidating code over many iterations

Usage:
    python convert_dotnet_to_python.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (optional)
    GEMINI_API_KEY: Your Google PaLM (Gemini) API key (optional)
    DEEPSEEK_API_KEY: Your DeepSeek API key (optional)

Requirements:
    pip install openai google-generativeai
"""

import os
import sys
import re
import time
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# LLM / Model Configuration
# ----------------------------------------------------------------------
OPENAI_MODEL = "o1-preview"  # or your preferred OpenAI model name
GEMINI_MODEL = "gemini-2.0-flash-exp"  # example model name
DEEPSEEK_MODEL = "deepseek-reasoner"

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

# Import OpenAI
try:
    import openai
except ImportError:
    print("[ERROR] Please install openai: pip install openai")
    sys.exit(1)
openai.api_key = openai_api_key

# Import Gemini (Google Generative AI) if available
try:
    import google.generativeai as genai
except ImportError:
    print(
        "[WARN] google-generativeai module not installed; Gemini calls will not work."
    )
genai.configure(api_key=gemini_api_key)
gemini_model_instance = None
if gemini_api_key:
    gemini_model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL)

# ----------------------------------------------------------------------
# System Prompt and Multi-Pass Objectives
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert software engineer with deep experience converting legacy .NET (C#) applications "
    "into modern Python applications using FastAPI, SQLAlchemy, Alembic, pytest, and the blueprint architecture. \n"
    "Your output must capture all key aspects including database models/migrations, HTTP endpoints, business logic, "
    "authentication and credentials, HTML/text templating, email delivery, service-to-service calls, and database queries. \n"
    "Additionally, you must produce a working Dockerfile and container configuration. \n"
    "Adhere strictly to best practices, PEP8 standards, and produce code that is modular, maintainable, and cohesive. \n"
    "Include inline comments where necessary and produce each output file prefixed by '# filename: <relative/path>'."
)

# Define a list of conversion passes with detailed objectives.
CONVERSION_PASSES = [
    (
        "Base Conversion",
        "Convert the legacy .NET application into a basic Python codebase using FastAPI with a blueprint architecture. "
        "Extract general business logic and create a preliminary structure including initial controllers and modules.",
    ),
    (
        "Database Models & Migrations",
        "Analyze and extract all database-related logic from the .NET code. Convert database models into SQLAlchemy models "
        "and generate Alembic migration scripts capturing schema changes.",
    ),
    (
        "HTTP Endpoints & Controllers",
        "Identify and convert all HTTP endpoints and controllers from the legacy code into FastAPI routers. "
        "Ensure proper separation of concerns between routes and business logic.",
    ),
    (
        "Authentication, Templating & Email Delivery",
        "Extract logic related to authentication, credential management, HTML and text templating, and email delivery. "
        "Convert these into idiomatic Python code using standard libraries and best practices.",
    ),
    (
        "Service-to-Service Calls & Database Queries",
        "Convert any service-to-service calls and complex database queries from the .NET application into asynchronous "
        "HTTP calls and robust SQLAlchemy query logic.",
    ),
    (
        "Unit Tests",
        "Generate a comprehensive suite of unit tests using pytest that covers all the functionality converted so far. "
        "Ensure tests are organized per module and cover edge cases and business logic.",
    ),
    (
        "Dockerization",
        "Generate a working Dockerfile (and docker-compose.yml if necessary) that containerizes the entire Python application. "
        "Ensure that environment variables and dependency installations are properly configured.",
    ),
    (
        "Final Consolidation",
        "Perform a final pass that consolidates all previous changes into a cohesive, consistent codebase. "
        "Refine the structure, resolve any dependency issues, and ensure overall code quality and adherence to best practices.",
    ),
]


# ----------------------------------------------------------------------
# LLM Utility Functions
# ----------------------------------------------------------------------
def call_llm_system_user(
    system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=8000
) -> str:
    """
    Calls the LLM with a system and a user prompt using OpenAI if available,
    otherwise falls back to Gemini or DeepSeek.
    """
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    if openai_api_key:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_openai_chat_completion(
            OPENAI_MODEL, messages, temperature=temperature, max_tokens=max_tokens
        )
    elif gemini_api_key and gemini_model_instance:
        return call_gemini(combined_prompt, temperature=temperature)
    elif deepseek_api_key:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_openai_chat_completion(
            DEEPSEEK_MODEL, messages, temperature=temperature, max_tokens=max_tokens
        )
    else:
        print("[ERROR] No valid API key provided for any supported LLM provider.")
        sys.exit(1)


def call_openai_chat_completion(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int = None,
) -> str:
    """
    Call the OpenAI Chat Completion endpoint.
    """
    print("[INFO] Contacting Chat Completion API... Please wait.")
    try:
        if model_name in ["o1-preview", "deepseek-reasoner"]:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return ""


def call_gemini(prompt: str, temperature=0.0) -> str:
    """
    Call the Gemini (Google PaLM) API for generation.
    """
    max_retries = 15
    retry_delay = 1  # seconds
    for attempt in range(max_retries):
        try:
            print("[INFO] Contacting Google PaLM API (Gemini)... Please wait.")
            response = gemini_model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                ),
                stream=True,
            )
            chunks = ""
            for chunk in response:
                chunks += chunk.text
            return chunks
        except Exception as e:
            print(f"[WARN] Gemini error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff
    print("[ERROR] Failed to get a valid response from Gemini.")
    return ""


def chunk_text(text: str, max_chunk_size: int = 12000) -> List[str]:
    """
    Splits a large text into chunks not exceeding max_chunk_size.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chunk_size, length)
        chunks.append(text[start:end])
        start = end
    return chunks


# ----------------------------------------------------------------------
# Multi-Pass Conversion Functionality
# ----------------------------------------------------------------------
def multi_pass_conversion(
    ir_data: dict, raw_dotnet: str, temperature=0.0, max_tokens=3000
) -> str:
    """
    Performs multiple iterative passes over the legacy .NET code to incrementally convert it
    into a modern Python codebase. Each pass refines a different aspect of the conversion.
    """
    # Start with an empty accumulated code base.
    accumulated_code = ""

    # For very large raw sources, break into chunks for context.
    dotnet_chunks = chunk_text(raw_dotnet, max_chunk_size=10000)
    context_source = "\n".join(
        dotnet_chunks
    )  # You might choose to refine per-pass chunking

    # Iterate over the defined conversion passes.
    for pass_index, (pass_name, pass_objective) in enumerate(
        CONVERSION_PASSES, start=1
    ):
        user_prompt = (
            f"=== Conversion Pass {pass_index}: {pass_name} ===\n\n"
            f"Objective: {pass_objective}\n\n"
            "Below is the Intermediate Representation (IR) of the legacy .NET codebase:\n"
            f"{repr(ir_data)}\n\n"
            "Below is the complete raw .NET source code (with file markers):\n"
            f"{context_source}\n\n"
            "The Python code generated so far is as follows:\n"
            "-------------------------\n"
            f"{accumulated_code}\n"
            "-------------------------\n\n"
            "Please update and extend the code to address the above objective for this pass. "
            "If necessary, refine previously generated logic, add new modules, endpoints, models, "
            "tests, or configuration files. Output all of your updated code using the format:\n"
            "# filename: relative/path/to/file\n"
            "<file contents>\n\n"
            "Do not omit any functionality; ensure that the final result includes proper handling "
            "for database models, migrations, HTTP endpoints, authentication, templating, email delivery, "
            "service-to-service calls, database queries, unit tests, and a working Dockerfile."
        )
        print(f"[INFO] Starting pass {pass_index}: {pass_name} ...")
        # Call the LLM with the system prompt and current pass objective
        pass_result = call_llm_system_user(
            SYSTEM_PROMPT, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        if pass_result.strip():
            accumulated_code = pass_result  # Replace previous code with refined version
        else:
            print(
                f"[WARN] Pass {pass_index} returned empty result; retaining previous code."
            )

    return accumulated_code


# ----------------------------------------------------------------------
# .NET (C#) Parsing / Advanced IR Construction
# ----------------------------------------------------------------------
def parse_dotnet_files(input_dir: str) -> Dict[str, Any]:
    """
    Parses all .cs files in the input directory to build an advanced IR.
    The IR includes classes, interfaces, methods, properties, fields, and uses simple heuristics
    to capture potential dependency information (for use in multi-pass conversion).

    The structure is:
      {
        "files": [
           {
             "filename": "relative/path/to/file.cs",
             "types": [
                {
                  "name": <name>,
                  "type": "class" | "interface" | "struct",
                  "methods": [<method names>],
                  "properties": [<property names>],
                  "fields": [<field names>],
                  "dependencies": [<other types referenced>]
                },
                ...
             ]
           },
           ...
        ]
      }
    """
    ir = {"files": []}
    # Simple regex patterns for types, methods, properties, and fields.
    type_pattern = re.compile(
        r"\b(public|internal|private|protected)?\s*(partial\s+)?(class|interface|struct)\s+(\w+)",
        re.MULTILINE,
    )
    method_pattern = re.compile(
        r"\b(public|internal|private|protected)\s+(static\s+)?([\w<>\[\]]+)\s+(\w+)\s*\(",
        re.MULTILINE,
    )
    property_pattern = re.compile(
        r"\b(public|internal|private|protected)\s+([\w<>\[\]]+)\s+(\w+)\s*\{\s*(get;|get\s*\{)",
        re.MULTILINE,
    )
    field_pattern = re.compile(
        r"\b(public|internal|private|protected)\s+([\w<>\[\]]+)\s+(\w+)\s*(=|;)",
        re.MULTILINE,
    )

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".cs"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                try:
                    with open(
                        full_path, "r", encoding="utf-8", errors="ignore"
                    ) as src_file:
                        content = src_file.read()
                except Exception as e:
                    content = f"// Error reading file: {e}"
                file_ir = {"filename": rel_path, "types": []}
                for match in type_pattern.finditer(content):
                    access, partial, typ, name = match.groups()
                    type_info = {
                        "name": name,
                        "type": typ,
                        "methods": [],
                        "properties": [],
                        "fields": [],
                        "dependencies": [],  # We could later use call graphs to populate this
                    }
                    # Search for methods, properties, and fields within the file (simple heuristic)
                    for m in method_pattern.finditer(content):
                        if m.start() > match.end():
                            type_info["methods"].append(m.group(4))
                    for p in property_pattern.finditer(content):
                        if p.start() > match.end():
                            type_info["properties"].append(p.group(3))
                    for fmatch in field_pattern.finditer(content):
                        if fmatch.start() > match.end():
                            type_info["fields"].append(fmatch.group(3))
                    file_ir["types"].append(type_info)
                ir["files"].append(file_ir)
    return ir


def read_raw_dotnet_sources(input_dir: str) -> str:
    """
    Concatenates the raw text of all .cs files with file markers.
    """
    blocks = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".cs"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                try:
                    with open(
                        full_path, "r", encoding="utf-8", errors="ignore"
                    ) as src_file:
                        content = src_file.read()
                except Exception as e:
                    content = f"// Error reading file: {e}"
                blocks.append(f"# filename: {rel_path}\n{content}\n\n")
    return "".join(blocks)


# ----------------------------------------------------------------------
# Output Splitting and Writing
# ----------------------------------------------------------------------
def split_and_write_files(llm_output: str, output_dir: str):
    """
    Splits the LLM output by lines starting with "# filename:" and writes each file to disk.
    """
    file_pattern = re.compile(r"^# filename:\s+(.+)$")
    current_file = None
    content_lines = []
    lines = llm_output.splitlines()

    def write_file(path: str, lines_content: List[str]):
        out_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out_file:
            out_file.write("".join(lines_content))
        print(f"[INFO] Written file: {out_path}")

    for line in lines:
        file_match = file_pattern.match(line)
        if file_match:
            if current_file and content_lines:
                write_file(current_file, content_lines)
                content_lines = []
            current_file = file_match.group(1).strip()
        else:
            if current_file is not None:
                content_lines.append(line + "\n")
    if current_file and content_lines:
        write_file(current_file, content_lines)


# ----------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_dotnet_to_python.py <input_directory> <output_directory>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"[ERROR] '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Parsing .cs files to build the IR...")
    ir_data = parse_dotnet_files(input_dir)

    print("[INFO] Reading raw .cs source files...")
    raw_dotnet = read_raw_dotnet_sources(input_dir)

    print("[INFO] Starting multi-pass conversion process. This may take some time...")
    final_code = multi_pass_conversion(
        ir_data, raw_dotnet, temperature=0.0, max_tokens=3000
    )

    if not final_code.strip():
        print("[ERROR] The LLM returned an empty result. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Splitting the final code into files and writing to output directory..."
    )
    split_and_write_files(final_code, output_dir)

    print(
        "\n[INFO] Conversion complete! Please check the output directory for your modern Python codebase."
    )


if __name__ == "__main__":
    main()
