#!/usr/bin/env python3
"""
This tool transforms legacy .NET projects into a modern Python codebase using:
  - FastAPI for HTTP endpoints,
  - SQLAlchemy with Alembic for ORM and migrations,
  - pytest for unit tests,
  - A modular blueprint architecture, and
  - A working Dockerfile (plus docker-compose.yml if needed).

Advanced techniques used:
  - Building a comprehensive Intermediate Representation (IR) from C# and XML sources.
  - Analyzing a dependency graph of types (using NetworkX) to guide conversion ordering.
  - Measuring code quality metrics (line counts, complexity hints).
  - Multi-pass iterative conversion with LLM calls (each pass refines different aspects).
  - Chunking of large input sources.

Usage:
    python final_convert_dotnet_to_python.py <input_directory> <output_directory>

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (optional)
    GEMINI_API_KEY: Your Google PaLM (Gemini) API key (optional)
    DEEPSEEK_API_KEY: Your DeepSeek API key (optional)

Requirements:
    pip install openai google-generativeai networkx
"""

import os
import sys
import re
import time
import json
import networkx as nx
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# LLM / Model Configuration
# ----------------------------------------------------------------------
OPENAI_MODEL = "o1-preview"  # Use your preferred model name here
GEMINI_MODEL = "gemini-2.0-flash-exp"
DEEPSEEK_MODEL = "deepseek-reasoner"
VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

# Import OpenAI
try:
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
except ImportError:
    print("[ERROR] Please install openai: pip install openai")
    sys.exit(1)

# Import Gemini if available
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


# Import and initialize vLLM if available
vllm_client = None
try:
    from vllm import LLMEngine
    from vllm.executors import UniProcExecutor
    from vllm import SamplingParams

    # Initialize the vLLM engine with the local model named "ok"
    vllm_client = LLMEngine(VLLM_MODEL, executor_class=UniProcExecutor, log_stats=False)
    print(vllm_client)
except ImportError:
    print("[WARN] vllm module not installed; vLLM calls will not work.")
except Exception as e:
    print(f"[WARN] vLLM initialization failed: {e}")


if gemini_api_key:
    MAX_TOKENS = 900000
    MAX_CHUNK_SIZE = 800000
elif deepseek_api_key:
    MAX_TOKENS = 5000
    MAX_CHUNK_SIZE = 200000
elif openai_api_key:
    MAX_TOKENS = 5000
    MAX_CHUNK_SIZE = 200000
else:
    # If no API key is provided, vLLM may be the only option.
    MAX_TOKENS = 20000
    MAX_CHUNK_SIZE = 50000


# ----------------------------------------------------------------------
# Advanced IR Construction, Dependency Graph, & Quality Metrics
# ----------------------------------------------------------------------
def parse_cs_file(filepath: str) -> Dict[str, Any]:
    """
    Parses a single C# file to extract type declarations (classes, interfaces, structs),
    methods, properties, and fields.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        content = f"// Error reading file: {e}"

    file_ir = {
        "filename": os.path.relpath(filepath),
        "content": content,
        "types": [],
        "line_count": content.count("\n"),
    }
    # Regex patterns (heuristic)
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

    for match in type_pattern.finditer(content):
        access, partial, typ, name = match.groups()
        type_info = {
            "name": name,
            "kind": typ,
            "methods": [],
            "properties": [],
            "fields": [],
        }
        # Collect methods, properties, fields (naively assume they occur later in file)
        for m in method_pattern.finditer(content, match.end()):
            type_info["methods"].append(m.group(4))
        for p in property_pattern.finditer(content, match.end()):
            type_info["properties"].append(p.group(3))
        for fmatch in field_pattern.finditer(content, match.end()):
            type_info["fields"].append(fmatch.group(3))
        file_ir["types"].append(type_info)
    return file_ir


def parse_xml_file(filepath: str) -> Dict[str, Any]:
    """
    Parses an XML file and returns a snippet of its content along with its filename.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        content = f"<!-- Error reading file: {e} -->"
    return {
        "filename": os.path.relpath(filepath),
        "content": content,
        "snippet": content[:500],
    }


def build_advanced_ir(input_dir: str) -> Dict[str, Any]:
    """
    Walks the input directory and builds an advanced IR from all .cs and .xml files.
    Also builds a dependency graph (heuristically) based on type names.
    """
    ir = {"cs_files": [], "xml_files": []}
    dependency_graph = nx.DiGraph()
    # Process C# files
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".cs"):
                full_path = os.path.join(root, f)
                cs_ir = parse_cs_file(full_path)
                ir["cs_files"].append(cs_ir)
                # Add nodes for each type in this file.
                for typ in cs_ir["types"]:
                    node_id = f"{cs_ir['filename']}::{typ['name']}"
                    dependency_graph.add_node(
                        node_id, kind=typ["kind"], file=cs_ir["filename"]
                    )
                    # Heuristic: if a method or field references another type (by simple name search),
                    # add an edge. (This is rudimentary; real code would need proper parsing.)
                    for other in cs_ir["types"]:
                        if other["name"] != typ["name"]:
                            if re.search(
                                r"\b" + re.escape(other["name"]) + r"\b",
                                cs_ir["content"],
                            ):
                                dep_node = f"{cs_ir['filename']}::{other['name']}"
                                dependency_graph.add_edge(node_id, dep_node)
    # Process XML files
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".xml"):
                full_path = os.path.join(root, f)
                xml_ir = parse_xml_file(full_path)
                ir["xml_files"].append(xml_ir)
    # Attach a dependency summary to the IR
    ir["dependency_graph"] = nx.readwrite.json_graph.node_link_data(dependency_graph)
    # Also add overall metrics
    ir["metrics"] = {
        "cs_file_count": len(ir["cs_files"]),
        "xml_file_count": len(ir["xml_files"]),
        "total_lines": sum(f["line_count"] for f in ir["cs_files"]),
    }
    return ir


def read_raw_project_sources(input_dir: str) -> str:
    """
    Concatenates the raw text of all .cs and .xml files, with file markers.
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


def dependency_graph_summary(ir: Dict[str, Any]) -> str:
    """
    Returns a summary string of the dependency graph (e.g. number of nodes and edges, and key clusters)
    to be included in the prompt.
    """
    dg = ir.get("dependency_graph", {})
    num_nodes = len(dg.get("nodes", []))
    num_edges = len(dg.get("links", []))
    summary = f"Dependency Graph Summary: {num_nodes} nodes, {num_edges} edges."
    return summary


# ----------------------------------------------------------------------
# LLM Utility Functions & Chunking
# ----------------------------------------------------------------------
def call_llm_system_user(
    system_prompt: str, user_prompt: str, temperature, max_tokens
) -> str:
    """
    Calls the LLM with system and user prompts using OpenAI if available,
    otherwise falls back to Gemini or DeepSeek.
    """
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    if openai_api_key:
        if "o1" in OPENAI_MODEL:
            messages = [
                {"role": "user", "content": combined_prompt},
            ]
        else:
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
    elif vllm_client is not None:
        return call_vllm(
            combined_prompt, temperature=temperature, max_tokens=max_tokens
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
            response = client.chat.completions.create(
                model=model_name, messages=messages
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
    Call the Gemini (Google PaLM) API.
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
            retry_delay *= 2
    print("[ERROR] Failed to get a valid response from Gemini.")
    return ""


def call_vllm(prompt: str, temperature=0.0, max_tokens=3000) -> str:
    """
    Call the vLLM local model using the vLLM package.
    """
    if vllm_client is None:
        print("[ERROR] vLLM client not initialized.")
        return ""
    try:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        print("[INFO] Contacting local vLLM model ... Please wait.")
        results = vllm_client.infer(prompt, sampling_params)
        # Assuming results is an iterable of responses with an attribute 'text'
        full_output = ""
        for res in results:
            full_output += res.text
        return full_output
    except Exception as e:
        print(f"[ERROR] vLLM API call failed: {e}")
        return ""


def chunk_text(text: str, max_chunk_size: int = 12000) -> List[str]:
    """
    Splits text into chunks not exceeding max_chunk_size characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chunk_size])
        start += max_chunk_size
    return chunks


# ----------------------------------------------------------------------
# Multi-Pass Iterative Conversion
# ----------------------------------------------------------------------
CONVERSION_PASSES = [
    (
        "Base Conversion",
        "Convert the legacy .NET project into a preliminary Python codebase using FastAPI with a modular blueprint architecture. "
        "Extract general business logic and overall project structure from the C# and XML sources.",
    ),
    (
        "Database Models & Migrations",
        "Extract all database-related logic and SQL queries (from both C# and XML). Convert these into SQLAlchemy models and generate Alembic migration scripts.",
    ),
    (
        "HTTP Endpoints & Controllers",
        "Identify HTTP endpoints and controllers from the legacy code and convert them into FastAPI routers. Ensure proper separation of concerns.",
    ),
    (
        "Authentication, Templating & Email Delivery",
        "Convert legacy authentication logic, credential management, HTML/text templating, and email delivery code into idiomatic Python.",
    ),
    (
        "Service-to-Service Calls & Database Queries",
        "Translate service-to-service communication and complex database queries into robust Python code using asynchronous calls and SQLAlchemy.",
    ),
    (
        "Unit Tests",
        "Generate a comprehensive suite of unit tests using pytest that covers all aspects of the converted functionality.",
    ),
    (
        "Dockerization",
        "Produce a complete Dockerfile (and docker-compose.yml if needed) to containerize the Python application. Ensure environment variables and dependencies are configured correctly.",
    ),
    (
        "Final Consolidation",
        "Consolidate all previous outputs into a cohesive, PEP8-compliant codebase. Ensure that all configuration, business logic, and infrastructure files (including the Dockerfile) are present and integrated.",
    ),
]


def multi_pass_conversion(
    ir: Dict[str, Any],
    raw_project: str,
    intermediate_dir: str,
    temperature=0.5,
    max_tokens=3000,
) -> str:
    """
    Performs iterative conversion passes with robust chunking and intermediate file storage.
    The raw project source is split into manageable chunks and processed sequentially.
    After each pass (and after each chunk within a pass), the intermediate output is saved to the specified
    intermediate directory. This allows you to CTRL-C and resume later, as well as inspect all intermediate analyses.

    Parameters:
      - ir: The advanced intermediate representation (IR) of the legacy project.
      - raw_project: The complete raw source (concatenated .cs and .xml files with file markers).
      - intermediate_dir: Directory where all intermediate outputs and analyses will be stored.
      - temperature: LLM sampling temperature.
      - max_tokens: Maximum tokens for LLM responses.

    Returns:
      - The final accumulated Python codebase as a single string.
    """

    # Ensure intermediate directory exists.
    os.makedirs(intermediate_dir, exist_ok=True)

    # Write IR summary for inspection.
    ir_summary = json.dumps(
        {
            "metrics": ir.get("metrics", {}),
            "dependency_graph": dependency_graph_summary(ir),
            "cs_files": [
                {"filename": f["filename"], "types": [t["name"] for t in f["types"]]}
                for f in ir.get("cs_files", [])
            ],
            "xml_files": [
                {"filename": f["filename"], "snippet": f["snippet"]}
                for f in ir.get("xml_files", [])
            ],
        },
        indent=2,
    )
    with open(
        os.path.join(intermediate_dir, "ir_summary.json"), "w", encoding="utf-8"
    ) as f:
        f.write(ir_summary)

    # Split the raw project source into chunks.
    project_chunks = chunk_text(raw_project, max_chunk_size=MAX_CHUNK_SIZE)

    accumulated_code = ""
    # Iterate over each conversion pass.
    for pass_idx, (pass_name, pass_obj) in enumerate(CONVERSION_PASSES, start=1):
        print(f"[INFO] Starting pass {pass_idx}: {pass_name} ...")
        # Define an intermediate file for this pass.
        pass_filename = os.path.join(
            intermediate_dir, f"pass_{pass_idx}_{pass_name.replace(' ', '_')}.txt"
        )
        # If the intermediate file exists, load it to resume; otherwise, start with previous code.
        if os.path.exists(pass_filename):
            print(f"[INFO] Found intermediate file for pass {pass_idx}, loading...")
            with open(pass_filename, "r", encoding="utf-8") as f:
                pass_accumulated_code = f.read()
        else:
            pass_accumulated_code = accumulated_code
            # Process each chunk sequentially for the current pass.
            for chunk_idx, chunk in enumerate(project_chunks, start=1):
                user_prompt = (
                    f"=== Conversion Pass {pass_idx}: {pass_name} (Chunk {chunk_idx}/{len(project_chunks)}) ===\n\n"
                    f"Objective: {pass_obj}\n\n"
                    "Intermediate Representation (IR):\n"
                    f"{ir_summary}\n\n"
                    "Raw Project Source (current chunk):\n"
                    f"{chunk}\n\n"
                    "Python code generated so far for this pass:\n"
                    "-------------------------\n"
                    f"{pass_accumulated_code}\n"
                    "-------------------------\n\n"
                    "Please update and extend the code to meet the objective of this pass for this chunk. "
                    "Output the entire updated codebase using the following file marker format:\n"
                    "# filename: relative/path/to/file\n"
                    "<file contents>\n\n"
                    "Ensure nothing is omitted (configuration, SQL queries, business logic, Dockerfile, etc.)."
                )
                chunk_result = call_llm_system_user(
                    "", user_prompt, temperature=temperature, max_tokens=max_tokens
                )
                if chunk_result.strip():
                    pass_accumulated_code = chunk_result
                    # Save intermediate result after processing each chunk.
                    with open(pass_filename, "w", encoding="utf-8") as f:
                        f.write(pass_accumulated_code)
                else:
                    print(
                        f"[WARN] Pass {pass_idx} chunk {chunk_idx} returned empty result; retaining previous code."
                    )
        # After processing all chunks for this pass, update the overall accumulated code.
        accumulated_code = pass_accumulated_code
        # Also, save the overall accumulated code for this pass for later inspection.
        overall_filename = os.path.join(
            intermediate_dir, f"overall_after_pass_{pass_idx}.txt"
        )
        with open(overall_filename, "w", encoding="utf-8") as f:
            f.write(accumulated_code)
    return accumulated_code


# ----------------------------------------------------------------------
# Splitting LLM Output into Files
# ----------------------------------------------------------------------
def split_and_write_files(llm_output: str, output_dir: str):
    """
    Splits the LLM output (expected to have file markers) and writes each file to disk.
    """
    file_pattern = re.compile(r"^# filename:\s+(.+)$")
    current_file = None
    content_lines = []
    for line in llm_output.splitlines():
        file_match = file_pattern.match(line)
        if file_match:
            if current_file and content_lines:
                write_file(os.path.join(output_dir, current_file), content_lines)
                content_lines = []
            current_file = file_match.group(1).strip()
        else:
            if current_file is not None:
                content_lines.append(line + "\n")
    if current_file and content_lines:
        write_file(os.path.join(output_dir, current_file), content_lines)


def write_file(path: str, lines_content: List[str]):
    """
    Writes content to the given file path, ensuring directories exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines_content))
    print(f"[INFO] Written file: {path}")


# ----------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python final_convert_dotnet_to_python.py <input_directory> <output_directory>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"[ERROR] '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    intermediate_dir = os.path.join(output_dir, '_intermediate_files')
    os.makedirs(intermediate_dir, exist_ok=True)

    print("[INFO] Building advanced IR from .cs and .xml files...")
    ir_data = build_advanced_ir(input_dir)

    print("[INFO] Reading raw project source files (.cs and .xml)...")
    raw_project = read_raw_project_sources(input_dir)

    print("[INFO] Starting multi-pass conversion process. This may take some time...")
    final_code = multi_pass_conversion(
        ir_data, raw_project, intermediate_dir, temperature=0.5, max_tokens=MAX_TOKENS
    )

    if not final_code.strip():
        print("[ERROR] The LLM returned an empty result. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Splitting the final code into files and writing to the output directory..."
    )
    split_and_write_files(final_code, output_dir)

    print(
        "\n[INFO] Conversion complete! Please check the output directory for your modern Python codebase."
    )


if __name__ == "__main__":
    main()
