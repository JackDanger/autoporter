#!/usr/bin/env python3
"""
A script to parse a legacy Java application and faithfully convert it into
a modern Python FastAPI application using SQLAlchemy, Alembic, and Pytest.

Key Points:
1. We parse the .java files using javalang to build an intermediate representation (IR).
2. We gather the entire raw .java code as well, so the LLM has both structural and textual info.
3. We feed the IR + raw code to the LLM, asking it to produce a *complete* Python codebase.
4. We carefully split files from the LLM output and write them to the specified output directory.
5. We keep the final code as simple and PEP8-compliant as possible.

Usage:
    python convert_app.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: (optional) Your OpenAI API key if you want to call the OpenAI Chat API.
    GEMINI_API_KEY: (optional) Your Google PaLM (Gemini) API key if you want to call PaLM instead.
                    If both keys are present, we'll default to OpenAI for now.

Requirements:
    pip install openai google-generativeai javalang
"""

import os
import sys
import re
import time
from typing import List, Dict, Any
import javalang  # pip install javalang
import google.generativeai as genai
from openai import OpenAI

# ----------------------------------------------------------------------
# LLM / Model Config
# ----------------------------------------------------------------------
OPENAI_MODEL = "gpt-4"  # Or "o1-preview", or any model you prefer
GEMINI_MODEL = "gemini-2.0-bison"  # Example

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)

# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a world-class software engineer with deep expertise in both Java and Python.
You convert Java code (including classes, methods, fields, annotations, logic, etc.) to Python,
ensuring no important details are lost.

You produce Python applications using:
- FastAPI for web endpoints
- SQLAlchemy for ORM
- Alembic for migrations
- Pytest for tests
- PEP8 and best-practice architecture

All code must be as close as possible in functionality to the Java source. If there's unclear Java logic, 
comment it in Python for clarity. Preserve and translate inline comments where relevant. 
Use Python type hints, docstrings, and be as idiomatic as possible without losing the Java logic.
"""

USER_PROMPT_TEMPLATE = """Below is an intermediate representation (IR) of the entire Java codebase 
(parsed with javalang) and, following that, the *raw text* of each .java file.

Your goal:
1. Faithfully convert *all* logic, data structures, classes, methods, and usage into a single coherent 
   Python application using FastAPI, SQLAlchemy, Alembic, and Pytest.
2. Use a recommended file structure:
    app/
       main.py
       models.py
       schemas.py
       routers/
         ... (one file per major route/controller)
       ...
    tests/
       test_*.py
    alembic/
       versions/
       env.py (or equivalent)
    requirements.txt (or pyproject.toml)
    README.md
3. Avoid skipping or omitting code. If certain Java classes are not obviously connected, 
   still convert them as separate modules or routers.
4. Keep the code no more complex than necessary, but do not lose logic. 
5. The final output must be *all* necessary files, separated by the format:
   # filename: relative/path/to/file.py
   <contents>

First, here is the IR (JSON-like structure you can read) summarizing the Java project:
{ir}

Next, here is the entire raw Java code (with file markers):
{raw_java_sources}

Please now produce the final Python codebase, ensuring everything is included.
Do not skip anything.
"""

# ----------------------------------------------------------------------
# LLM Utility Functions
# ----------------------------------------------------------------------

def call_llm_system_user(system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=8000) -> str:
    """
    Calls the LLM with a system prompt and a user prompt using OpenAI if available,
    otherwise uses Gemini. Returns the LLM response text.
    """
    if openai_client.api_key:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_openai_chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
    else:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return call_gemini(combined_prompt, temperature=temperature)

def call_openai_chat_completion(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    """Call the OpenAI Chat API (gpt-4 or similar) and return the response content."""
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set or is empty.")

    print("[INFO] Contacting OpenAI Chat Completion API... Please wait.")
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def call_gemini(prompt: str, temperature=0.0) -> str:
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
                    temperature=temperature,
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

# ----------------------------------------------------------------------
# Java Parsing / IR Construction
# ----------------------------------------------------------------------

def parse_java_files(input_dir: str) -> Dict[str, Any]:
    """
    Parse all .java files using javalang, building an intermediate representation.

    Returns a dictionary of:
    {
      "files": [
        {
          "filename": "Relative path",
          "classes": [
             {
               "name": str,
               "extends": str or None,
               "implements": [...],
               "methods": [
                  {
                     "name": str,
                     "params": [...],
                     "return_type": str or None,
                     "is_static": bool,
                     "is_abstract": bool,
                     "annotations": [...],
                     "body_lines": (optional) int or snippet
                  },
                  ...
               ],
               "fields": [...],
               "annotations": [...]
             },
             ...
          ]
        },
        ...
      ]
    }
    """
    ir = {"files": []}

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".java"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                with open(full_path, "r", encoding="utf-8") as src_file:
                    src = src_file.read()

                try:
                    tree = javalang.parse.parse(src)
                except javalang.parser.JavaSyntaxError as e:
                    print(f"[WARN] Java syntax error in {rel_path}: {e}. Will store partial parse info.")
                    # Attempt partial parse or store minimal IR
                    # We'll just skip deeper parse but store raw text in IR
                    ir["files"].append({
                        "filename": rel_path,
                        "error": str(e),
                        "classes": [],
                    })
                    continue

                file_ir = {
                    "filename": rel_path,
                    "classes": []
                }

                # For top-level types in this file
                for t in tree.types:
                    if isinstance(t, javalang.tree.ClassDeclaration) or isinstance(t, javalang.tree.EnumDeclaration):
                        class_info = {
                            "name": t.name,
                            "extends": str(t.extends.name) if t.extends else None,
                            "implements": [i.name for i in t.implements] if t.implements else [],
                            "annotations": [a.name for a in t.annotations] if t.annotations else [],
                            "fields": [],
                            "methods": [],
                            "type": "enum" if isinstance(t, javalang.tree.EnumDeclaration) else "class"
                        }

                        # Fields
                        for field in t.fields:
                            # field.declarators might be multiple variables in one statement
                            for decl in field.declarators:
                                field_info = {
                                    "name": decl.name,
                                    "type": str(field.type) if field.type else None,
                                    "annotations": [a.name for a in field.annotations] if field.annotations else [],
                                }
                                class_info["fields"].append(field_info)

                        # Methods
                        for method in t.methods:
                            method_info = {
                                "name": method.name,
                                "params": [
                                    {
                                        "name": p.name,
                                        "type": str(p.type.name) if p.type else None,
                                    }
                                    for p in method.parameters
                                ],
                                "return_type": str(method.return_type.name) if method.return_type else None,
                                "is_static": 'static' in method.modifiers,
                                "is_abstract": 'abstract' in method.modifiers,
                                "annotations": [a.name for a in method.annotations] if method.annotations else [],
                                # We'll just store a line count as a basic measure of complexity
                                "body_line_count": (
                                    len(method.body) if method.body else 0
                                )
                            }
                            class_info["methods"].append(method_info)

                        file_ir["classes"].append(class_info)

                    elif isinstance(t, javalang.tree.InterfaceDeclaration):
                        # Similar structure
                        interface_info = {
                            "name": t.name,
                            "implements": [],  # Java doesn't do "implements" for interface
                            "extends": [ext.name for ext in t.extends] if t.extends else [],
                            "annotations": [a.name for a in t.annotations] if t.annotations else [],
                            "fields": [],
                            "methods": [],
                            "type": "interface"
                        }
                        for method in t.methods:
                            method_info = {
                                "name": method.name,
                                "params": [
                                    {
                                        "name": p.name,
                                        "type": str(p.type.name) if p.type else None,
                                    }
                                    for p in method.parameters
                                ],
                                "return_type": str(method.return_type.name) if method.return_type else None,
                                "is_static": 'static' in method.modifiers,
                                "is_abstract": 'abstract' in method.modifiers,
                                "annotations": [a.name for a in method.annotations] if method.annotations else [],
                                "body_line_count": (
                                    len(method.body) if method.body else 0
                                )
                            }
                            interface_info["methods"].append(method_info)
                        file_ir["classes"].append(interface_info)

                    # If there's more exotic Java structures, handle them if needed
                    # For brevity, we won't parse them in detail here

                ir["files"].append(file_ir)

    return ir


def read_raw_java_sources(input_dir: str) -> str:
    """
    Concatenate the raw text of all .java files, separated by markers.
    This ensures we pass the original text to the LLM.
    """
    blocks = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".java"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir)
                with open(full_path, "r", encoding="utf-8") as src_file:
                    content = src_file.read()
                blocks.append(f"# filename: {rel_path}\n{content}\n\n")
    return "".join(blocks)


# ----------------------------------------------------------------------
# Output Parsing / Writing
# ----------------------------------------------------------------------

def split_and_write_files(llm_output: str, output_dir: str):
    """
    Splits the LLM output based on `# filename: some_path.ext` lines,
    then writes each file to the correct path under output_dir.
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
        if line.startswith("# filename: ") and file_pattern.match(line):
            # If we were tracking another file, write it out first
            if current_file and content:
                write_to_file(current_file, content)
                content = []

            current_file = line[len("# filename: "):].strip()
        else:
            if current_file is not None:
                content.append(line + "\n")

    # Write the last file if any remains
    if current_file and content:
        write_to_file(current_file, content)


# ----------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_app.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 1) Build IR from the Java files
    print("[INFO] Parsing Java files to build IR...")
    ir_data = parse_java_files(input_dir)

    # 2) Read raw Java code
    print("[INFO] Reading raw Java sources...")
    raw_java = read_raw_java_sources(input_dir)

    # 3) Construct user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ir=repr(ir_data),  # or could use json.dumps, but repr is enough for the LLM
        raw_java_sources=raw_java
    )

    # 4) Call the LLM (system + user prompts)
    print("[INFO] Calling LLM to transform Java -> Python codebase...")
    llm_output = call_llm_system_user(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=8000)

    if not llm_output.strip():
        print("[ERROR] LLM returned an empty response. Exiting.")
        sys.exit(1)

    # 5) Write the resulting files
    print("[INFO] Writing output files to disk...")
    split_and_write_files(llm_output, output_dir)

    print("\n[INFO] Conversion complete! Check your output directory for the new Python codebase.")


if __name__ == "__main__":
    main()