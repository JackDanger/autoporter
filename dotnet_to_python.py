#!/usr/bin/env python3
"""
A script to parse a legacy 2018-era .NET (C#) application and faithfully convert it into
a modern, Dockerized ASP.NET Core application using:
  - ASP.NET Core for web endpoints
  - Entity Framework Core for ORM/data access
  - xUnit for testing

Key Points:
1. We parse the .cs files with a custom regex-based parser to build an intermediate representation (IR).
2. We gather the entire raw .cs source code (with file markers) so the LLM has both structural and textual info.
3. We feed the IR plus raw code to the LLM, asking it to produce a *complete* modern ASP.NET Core codebase.
4. We carefully split files from the LLM output and write them to the specified output directory.
5. The generated code must include a Dockerfile (and docker-compose.yml if needed) plus comprehensive unit tests.

Usage:
    python convert_dotnet_app.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: (optional) Your OpenAI API key if you want to call the OpenAI Chat API.
    GEMINI_API_KEY: (optional) Your Google PaLM (Gemini) API key if you want to call PaLM instead.
                    If both keys are present, we'll default to OpenAI for now.
    DEEPSEEK_API_KEY: (optional) Your DeepSeek API key if you want to use the DeepSeek model.

Requirements:
    pip install openai google-generativeai
"""

import os
import sys
import re
import time
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# LLM / Model Config
# ----------------------------------------------------------------------
OPENAI_MODEL = "o3-mini"  # or your chosen OpenAI model name
GEMINI_MODEL = "gemini-2.0-flash-exp"  # example
DEEPSEEK_MODEL = "deepseek-reasoner"

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

# Import OpenAI libraries if available
try:
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
except ImportError:
    print("[ERROR] Please install openai: pip install openai")
    sys.exit(1)

# Setup OpenAI client instance(s)

# Import Google Generative AI if available
try:
    import google.generativeai as genai
except ImportError:
    print(
        "[WARN] google-generativeai module not installed. Gemini calls will fail if selected."
    )
genai.configure(api_key=gemini_api_key)
gemini_model_instance = None
if gemini_api_key:
    gemini_model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL)

# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a world-class software engineer with deep expertise in both legacy .NET Framework (circa 2018)
and modern ASP.NET Core development. You convert legacy C# code (including classes, interfaces, structs, methods, properties, fields, etc.)
into a fully modern, Dockerized ASP.NET Core application. Your target application must include:
- A proper ASP.NET Core project (with Program.cs and Startup.cs or minimal hosting setup)
- A clean separation of Controllers, Models, Services, and Data (using Entity Framework Core)
- A Dockerfile (and docker-compose.yml if needed) to containerize the application
- Comprehensive unit tests using xUnit that carefully cover all discovered functionality
- All best practices for modern C# and ASP.NET Core development

Preserve all business logic, inline comments, and structure from the legacy source. If any logic is unclear, add clarifying comments.
Output all necessary files in the format specified below.
"""

USER_PROMPT_TEMPLATE = """Below is an intermediate representation (IR) of the entire legacy .NET (C#) codebase
(parsed from the .cs files) and, following that, the *raw text* of each .cs file.

Your goal:
1. Faithfully convert *all* logic, data structures, classes, interfaces, methods, properties, fields, etc.
   into a single coherent, modern ASP.NET Core application.
2. Use the recommended file structure:
    src/
      MyApp.csproj
      Program.cs
      Startup.cs (or equivalent)
      Controllers/
      Models/
      Data/
      Services/
    tests/
      MyApp.Tests.csproj (with unit tests using xUnit)
    Dockerfile
    docker-compose.yml (if needed)
    README.md
3. Include unit tests that thoroughly verify every piece of functionality discovered in the source app.
4. Do not skip any logic—even if parts seem disconnected—convert them into appropriate modules.
5. The final output must include *all* necessary files, separated by the format:
   # filename: relative/path/to/file
   <contents>

First, here is the IR (JSON-like structure):
{ir}

Next, here is the entire raw .NET source code (with file markers):
{raw_dotnet_sources}

Please now produce the complete modern ASP.NET Core codebase accordingly.
"""


# ----------------------------------------------------------------------
# LLM Utility Functions
# ----------------------------------------------------------------------
def call_llm_system_user(
    system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=8000
) -> str:
    """
    Calls the LLM with a system prompt and a user prompt using OpenAI if available,
    otherwise uses Gemini. Returns the LLM response text.
    """
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    if openai_api_key:
        # For OpenAI Chat API models that require system + user messages
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
    Call an OpenAI-compatible Chat Completion endpoint.
    """
    print("[INFO] Contacting Chat Completion API... Please wait.")
    try:
        if model_name in ["o1-preview", "deepseek-reasoner"]:
            response = client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature
            )
        else:
            response = client.chat.completions.create(
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
    Call the Gemini API for generation.
    """
    max_retries = 15
    retry_delay = 1  # in seconds
    for attempt in range(max_retries):
        try:
            print("[INFO] Contacting Google Gemini... Please wait.")
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
            print(
                f"[WARN] Error generating response: {e}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff
    print("[ERROR] Failed to get a valid response from Google Gemini.")
    return ""


def chunk_text(text: str, max_chunk_size: int = 12000) -> List[str]:
    """
    Split a large string into multiple pieces, each at most `max_chunk_size` characters.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chunk_size, length)
        chunks.append(text[start:end])
        start = end
    return chunks


def chunk_and_call_llm(
    ir_data: dict, raw_dotnet: str, system_prompt: str, temperature=0.0, max_tokens=3000
) -> str:
    """
    Break the raw .NET source into manageable chunks, then iteratively call the LLM,
    carrying forward partial code so everything is eventually addressed with proper context.
    """
    dotnet_chunks = chunk_text(raw_dotnet, max_chunk_size=10000)  # adjust if needed
    accumulated_code = ""  # this stores the code generated so far
    for i, chunk in enumerate(dotnet_chunks):
        user_prompt = f"""
You have the following Intermediate Representation (IR) of the .NET codebase:
{repr(ir_data)}

Below is the partial code you've generated so far (accumulated):
\"\"\"
{accumulated_code}
\"\"\"

Now, here is a chunk of the raw .NET source code we haven't processed yet:
\"\"\"
{chunk}
\"\"\"

Please update or extend the code so that it incorporates everything in this chunk without losing previously converted logic.
If classes, methods, or modules already converted need to be refined, refine them.
If new logic appears, incorporate it.
Output all of your updated code using the format:
# filename: relative/path/to/file
<contents>

Ensure that the final codebase is complete, cohesive, and follows the modern ASP.NET Core project structure with Dockerization and unit tests.
"""
        print(f"[INFO] Processing chunk {i+1} / {len(dotnet_chunks)}...")
        updated_code = call_llm_system_user(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        if updated_code.strip():
            accumulated_code = updated_code
        else:
            print(
                "[WARN] Received empty response for this chunk; retaining previously generated code."
            )
    return accumulated_code


# ----------------------------------------------------------------------
# .NET (C#) Parsing / IR Construction
# ----------------------------------------------------------------------
def parse_dotnet_files(input_dir: str) -> Dict[str, Any]:
    """
    Walks through the input directory to find all .cs files and builds an intermediate
    representation (IR) that summarizes classes, interfaces, structs, methods, properties, and fields.

    The IR will have the structure:

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
              },
              ...
           ]
         },
         ...
      ]
    }
    """
    ir = {"files": []}
    # Regex patterns (very simple; may not catch all edge cases in messy legacy code)
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
                with open(
                    full_path, "r", encoding="utf-8", errors="ignore"
                ) as src_file:
                    content = src_file.read()
                file_ir = {"filename": rel_path, "types": []}

                for match in type_pattern.finditer(content):
                    access, partial, typ, name = match.groups()
                    type_info = {
                        "name": name,
                        "type": typ,
                        "methods": [],
                        "properties": [],
                        "fields": [],
                    }
                    # Methods inside the type (simple scan of the entire file)
                    for m in method_pattern.finditer(content):
                        # A simple heuristic: if the method appears after the type definition
                        if m.start() > match.end():
                            type_info["methods"].append(m.group(4))
                    # Properties
                    for p in property_pattern.finditer(content):
                        if p.start() > match.end():
                            type_info["properties"].append(p.group(3))
                    # Fields
                    for fmatch in field_pattern.finditer(content):
                        if fmatch.start() > match.end():
                            type_info["fields"].append(fmatch.group(3))
                    file_ir["types"].append(type_info)
                ir["files"].append(file_ir)
    return ir


def read_raw_dotnet_sources(input_dir: str) -> str:
    """
    Concatenates the raw text of all .cs files, separated by markers.
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
# Output Parsing / Writing
# ----------------------------------------------------------------------
def split_and_write_files(llm_output: str, output_dir: str):
    """
    Splits the LLM output based on lines starting with "# filename: " and writes each file
    to the correct location under the output directory.
    """
    # We allow any file name (with or without extension) after "# filename: "
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
    # Write the last file if needed
    if current_file and content_lines:
        write_file(current_file, content_lines)


# ----------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python convert_dotnet_app.py <input_directory> <output_directory>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"[ERROR] '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build IR from the .cs files
    print("[INFO] Parsing .cs files to build IR...")
    ir_data = parse_dotnet_files(input_dir)

    # 2) Read raw .NET source code
    print("[INFO] Reading raw .cs sources...")
    raw_dotnet = read_raw_dotnet_sources(input_dir)

    # 3) Construct the user prompt using the updated template
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ir=repr(ir_data), raw_dotnet_sources=raw_dotnet
    )

    # 4) Call the LLM in a chunked fashion
    print(
        "[INFO] Generating modern ASP.NET Core codebase from legacy .NET sources (this may take a while)..."
    )
    llm_output = chunk_and_call_llm(
        ir_data=ir_data,
        raw_dotnet=raw_dotnet,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=3000,
    )
    if not llm_output.strip():
        print("[ERROR] LLM returned an empty response. Exiting.")
        sys.exit(1)

    # 5) Write the resulting files to disk
    print("[INFO] Writing output files to disk...")
    split_and_write_files(llm_output, output_dir)

    print(
        "\n[INFO] Conversion complete! Check the output directory for your modern ASP.NET Core codebase."
    )


if __name__ == "__main__":
    main()
