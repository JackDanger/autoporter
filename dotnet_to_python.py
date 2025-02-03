#!/usr/bin/env python3
"""
A conversion tool to transform legacy 2018-era .NET (C#) applications into a modern Python codebase
using FastAPI, SQLAlchemy, Alembic, pytest, and a blueprint architecture. In addition, it generates
a working Dockerfile (and optionally docker-compose.yml).

This tool performs multiple passes over the project so as to capture:
  - Database models and migrations
  - HTTP endpoints and controllers
  - Business logic (including authentication, credentials, templating, email delivery)
  - Service-to-service calls and database queries
  - Comprehensive unit tests (pytest)
  - Critical configuration (including XML files with SQL queries and settings)
  - Containerization via Docker

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


if gemini_api_key:
    MAX_TOKENS = 800000
elif deepseek_api_key:
    MAX_TOKENS = 5000
elif openai_api_key:
    MAX_TOKENS = 5000

# ----------------------------------------------------------------------
# System Prompt and Multi-Pass Objectives
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert software engineer with deep experience converting legacy .NET (C#) applications "
    "into modern Python applications using FastAPI, SQLAlchemy, Alembic, pytest, and a modular blueprint architecture. \n"
    "The legacy projects include not only C# source code but also XML configuration files containing critical SQL queries, "
    "connection settings, and other configuration data. \n"
    "Your output must capture all key aspects including database models/migrations, HTTP endpoints, business logic, "
    "authentication and credentials, HTML/text templating, email delivery, service-to-service calls, and complex database queries. \n"
    "Additionally, you must produce a working Dockerfile (and docker-compose.yml if needed) to containerize the application. \n"
    "Follow best practices and PEP8 standards; produce modular, maintainable code with inline comments as needed. \n"
    "Output each file with a marker in the format: '# filename: relative/path/to/file'."
)

# Define conversion passes with explicit objectives
CONVERSION_PASSES = [
    (
        "Base Conversion",
        "Convert the legacy .NET application into a basic Python codebase using FastAPI with a modular blueprint architecture. "
        "Extract and translate the general business logic from the C# code.",
    ),
    (
        "Database Models & Migrations",
        "Identify and extract all database-related logic and SQL queries from both C# and XML files. "
        "Convert database models to SQLAlchemy models and generate Alembic migration scripts.",
    ),
    (
        "HTTP Endpoints & Controllers",
        "Extract HTTP endpoints and controllers from the legacy .NET project and convert them into FastAPI routers. "
        "Ensure a clear separation of concerns.",
    ),
    (
        "Authentication, Templating & Email Delivery",
        "Identify legacy authentication logic, HTML/text templating, and email delivery code (possibly spread across C# and XML). "
        "Convert these into idiomatic Python code with secure best practices.",
    ),
    (
        "Service-to-Service Calls & Database Queries",
        "Convert service-to-service communication and any complex database queries from the legacy code into asynchronous HTTP calls "
        "and robust SQLAlchemy query logic.",
    ),
    (
        "Unit Tests",
        "Generate a comprehensive suite of unit tests using pytest that covers all the converted functionality. "
        "Organize tests per module and include edge cases.",
    ),
    (
        "Dockerization",
        "Produce a complete Dockerfile (and docker-compose.yml if needed) that containerizes the entire Python application. "
        "Ensure that all dependencies, environment variables, and build instructions are correctly specified.",
    ),
    (
        "Final Consolidation",
        "Perform a final pass that consolidates all previous outputs into a cohesive, consistent, PEP8-compliant codebase. "
        "Resolve any dependency issues and ensure that configuration, business logic, and infrastructure files (like the Dockerfile) are present.",
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
    ir_data: dict, raw_project: str, temperature=0.0, max_tokens=3000
) -> str:
    """
    Performs multiple iterative passes over the legacy .NET project (including C# and XML files)
    to incrementally convert it into a modern Python codebase.
    """
    accumulated_code = ""
    project_chunks = chunk_text(raw_project, max_chunk_size=10000)
    context_source = "\n".join(project_chunks)

    for pass_index, (pass_name, pass_objective) in enumerate(
        CONVERSION_PASSES, start=1
    ):
        user_prompt = (
            f"=== Conversion Pass {pass_index}: {pass_name} ===\n\n"
            f"Objective: {pass_objective}\n\n"
            "Below is the Intermediate Representation (IR) of the legacy .NET project:\n"
            f"{repr(ir_data)}\n\n"
            "Below is the complete raw legacy project source code, including all .cs and XML files, with file markers:\n"
            f"{context_source}\n\n"
            "The Python code generated so far is as follows:\n"
            "-------------------------\n"
            f"{accumulated_code}\n"
            "-------------------------\n\n"
            "Please update and extend the code to address the above objective for this pass. "
            "If necessary, refine previously generated logic, add new modules, endpoints, models, tests, "
            "or configuration files (such as a Dockerfile). Output all of your updated code using the format:\n"
            "# filename: relative/path/to/file\n"
            "<file contents>\n\n"
            "Do not omit any functionality; ensure that configuration, SQL queries (from XML), business logic, "
            "and Docker containerization are all present."
        )
        print(f"[INFO] Starting pass {pass_index}: {pass_name} ...")
        pass_result = call_llm_system_user(
            SYSTEM_PROMPT, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        if pass_result.strip():
            accumulated_code = pass_result
        else:
            print(
                f"[WARN] Pass {pass_index} returned empty result; retaining previous code."
            )

    return accumulated_code


# ----------------------------------------------------------------------
# .NET (C#) and XML Parsing / Advanced IR Construction
# ----------------------------------------------------------------------
def parse_cs_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Parses all .cs files to build an IR of C# source code.
    Returns a list of file IRs.
    """
    cs_files = []
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
                        "dependencies": [],
                    }
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
                cs_files.append(file_ir)
    return cs_files


def parse_xml_files(input_dir: str) -> List[Dict[str, str]]:
    """
    Parses all XML files to build a brief IR for configuration and SQL query files.
    Returns a list of dictionaries with filename and a snippet of the content.
    """
    xml_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".xml"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                try:
                    with open(
                        full_path, "r", encoding="utf-8", errors="ignore"
                    ) as xml_file:
                        content = xml_file.read()
                except Exception as e:
                    content = f"<!-- Error reading file: {e} -->"
                snippet = content[:500]  # first 500 characters as a summary
                xml_files.append({"filename": rel_path, "snippet": snippet})
    return xml_files


def build_ir(input_dir: str) -> Dict[str, Any]:
    """
    Combines the IR from .cs files and XML files.
    """
    ir = {
        "cs_files": parse_cs_files(input_dir),
        "xml_files": parse_xml_files(input_dir),
    }
    return ir


def read_raw_project_sources(input_dir: str) -> str:
    """
    Concatenates the raw text of all .cs and .xml files with file markers.
    """
    blocks = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith((".cs", ".xml")):
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

    print("[INFO] Building Intermediate Representation (IR) from .cs and XML files...")
    ir_data = build_ir(input_dir)

    print("[INFO] Reading raw project source files (.cs and .xml)...")
    raw_project = read_raw_project_sources(input_dir)

    print("[INFO] Starting multi-pass conversion process. This may take some time...")
    final_code = multi_pass_conversion(
        ir_data, raw_project, temperature=0.0, max_tokens=MAX_TOKENS
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
