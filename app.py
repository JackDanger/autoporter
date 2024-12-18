import argparse
import os
import uuid
import re
import time
from tqdm import tqdm
import google.generativeai as genai

genai.configure(api_key=os.environ['API_KEY'])
MODEL_NAME = os.environ.get('MODEL', 'gemini-1.5-flash-8b')
model = genai.GenerativeModel(model_name=MODEL_NAME)


def print_step(step):
    print(f"[Step] {step}")


def analyze_project(project_path):
    print_step("Analyzing the input project structure.")
    input_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            file_path = os.path.join(root, file)
            input_files.append(file_path)
    print_step(f"Found {len(input_files)} source files.")
    return input_files


def strategy_file_by_file_translation(input_files, project_path, output_dir):
    print_step("Starting Strategy 1: File-by-file translation.")
    for file_path in tqdm(input_files, desc="Translating files", unit="file"):
        relative_path = os.path.relpath(file_path, project_path)
        relative_path = os.path.normpath(relative_path)
        output_path = os.path.join(output_dir, relative_path)
        output_path = os.path.splitext(output_path)[0] + '.py'
        if os.path.exists(output_path):
            tqdm.write(f"Skipping existing file: {output_path}")
            continue
        tqdm.write(f"Translating {file_path} -> {output_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except: UnicodeDecodeError
            pass
        translated_code = translate_code(code)
        if not translated_code:
            tqdm.write(f"Translation failed for {file_path}. Skipping.")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_code)
    print_step("Strategy 1 completed.")


def strategy_simplify_python_app(strategy1_output_dir, output_dir):
    print_step("Starting Strategy 1.1: Simplifying the Python application.")
    python_files = []
    for root, _, files in os.walk(strategy1_output_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    for file_path in tqdm(python_files, desc="Simplifying files", unit="file"):
        relative_path = os.path.relpath(file_path, strategy1_output_dir)
        output_path = os.path.join(output_dir, relative_path)
        if os.path.exists(output_path):
            tqdm.write(f"Skipping existing file: {output_path}")
            continue
        tqdm.write(f"Simplifying {file_path} -> {output_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        simplified_code = simplify_python_code(code)
        if not simplified_code:
            tqdm.write(f"Simplification failed for {file_path}. Skipping.")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simplified_code)
    print_step("Strategy 1.1 completed.")


def strategy_reimplement_from_python_summaries(python_files, project_dir, output_dir):
    print_step("Starting Strategy 2: Reimplementing from simplified Python code summaries.")
    project_description = extract_project_description_from_python(python_files, project_dir, output_dir)
    if not project_description:
        print_step("Failed to extract project description from Python code. Skipping Strategy 2.")
        return
    design = generate_high_level_design(project_description)
    if not design:
        print_step("Failed to generate high-level design. Skipping Strategy 2.")
        return
    implement_python_project(design, output_dir)
    print_step("Strategy 2 completed.")


def strategy_reimplement_from_design(input_files, project_dir, output_dir):
    print_step("Starting Strategy 3: Reimplementing from high-level design based on C# summaries.")
    project_description = extract_project_description(input_files, project_dir, output_dir)
    if not project_description:
        print_step("Failed to extract project description. Skipping Strategy 3.")
        return
    design = generate_high_level_design(project_description)
    if not design:
        print_step("Failed to generate high-level design. Skipping Strategy 3.")
        return
    implement_python_project(design, output_dir)
    print_step("Strategy 3 completed.")


def strategy_modular_implementation(input_files, project_dir, output_dir):
    print_step("Starting Strategy 4: Modular Implementation Based on High-Level Design.")
    project_description = extract_project_description(input_files, project_dir, output_dir)
    if not project_description:
        print_step("Failed to extract project description. Skipping Strategy 4.")
        return
    design = generate_high_level_design(project_description)
    if not design:
        print_step("Failed to generate high-level design. Skipping Strategy 4.")
        return
    implement_python_project_modular(design, output_dir)
    print_step("Strategy 4 completed.")


def strategy_full_summary(input_files, project_dir, output_dir):
    print_step("Starting Strategy 5: Full Comprehensive Summary for Complete Replacement.")
    all_code_segments = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            all_code_segments.append(f"\n[FILE]: {file_path}\n{code}\n")
    combined_code = "\n".join(all_code_segments)

    prompt = (
        "You are a systems analyst tasked with producing a fully comprehensive specification of an entire software system, given all of its source code. "
        "Your specification should enable someone to re-implement the system with absolutely identical external behavior, without any guidance from the original code. "
        "The specification should include:\n\n"
        "- A complete description of all external inputs and outputs of the system.\n"
        "- A detailed description of all external interfaces (for example, endpoints if present), including every parameter and data structure.\n"
        "- A precise and exhaustive description of any persistent storage schema, including all entities, their fields, constraints, and relationships.\n"
        "- A step-by-step description of the business logic, including how data flows between interfaces and storage.\n"
        "- An exhaustive outline of all internal components, their responsibilities, their interactions, and the logic that each implements.\n\n"
        "Do not mention any specific programming languages, frameworks, or technologies. Focus solely on the domain logic, the data that flows through the system, "
        "the transformations that occur, and the interfaces and storage mechanisms from a conceptual perspective. The goal is for the resulting specification to be so "
        "detailed and precise that the entire software could be re-created with identical input/output behavior.\n\n"
        "Provide only the specification, without any additional commentary or introduction.\n\n"
        f"{combined_code}\n"
    )
    specification = call_gemini(prompt)
    specification_text = specification.strip()

    strategy5_output_dir = os.path.join(output_dir, 'strategy5')
    os.makedirs(strategy5_output_dir, exist_ok=True)
    spec_file_path = os.path.join(strategy5_output_dir, "full_system_specification.txt")
    with open(spec_file_path, 'w', encoding='utf-8') as f:
        f.write(specification_text)

    print_step("Strategy 5 completed.")
    return spec_file_path


def strategy_reimplement_from_full_spec(spec_file_path, output_dir):
    print_step("Starting Strategy 6: Reimplementing from the Full Comprehensive Specification.")
    with open(spec_file_path, 'r', encoding='utf-8') as f:
        full_spec = f.read()

    prompt = (
        "As an expert Python developer, implement a new project described fully by the specification below. "
        "Use FastAPI for the HTTP interface, SQLAlchemy for database interactions, Alembic for database migrations, "
        "and follow a blueprint (modular) pattern for structuring the code. "
        "Organize the project using modern Python best practices (e.g., separate directories for routes, models, services, etc.). "
        "Ensure the code follows Python best practices, is well-commented, and can be run and tested. "
        "All functionality, inputs, and outputs must match the specification exactly.\n\n"
        "Provide the code files in the following format:\n\n"
        "### filename.py\n"
        "```python\n"
        "# code here\n"
        "```\n\n"
        "Provide only the code files as specified, without any additional text.\n\n"
        f"{full_spec}\n"
    )
    response_content = call_gemini(prompt)
    code_files = parse_code_files_from_response_multiple_files(response_content)
    strategy6_output_dir = os.path.join(output_dir, 'strategy6')
    os.makedirs(strategy6_output_dir, exist_ok=True)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(strategy6_output_dir, sanitized_file_name)
        if os.path.exists(output_path):
            print_step(f"Skipping existing file: {output_path}")
            continue
        print_step(f"Writing file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
    print_step("Strategy 6 completed.")


def strategy_full_rewrite_from_entire_source(input_files, project_path, output_dir):
    print_step("Starting Strategy 7: Full Rewrite from Entire Source Code in a Single Prompt.")
    # Gather all original code into a single prompt
    all_code_segments = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            all_code_segments.append(f"\n[FILE]: {file_path}\n{code}\n")
    combined_code = "\n".join(all_code_segments)

    # Craft the prompt for maximum quality, correctness, completeness, and factoring
    prompt = (
        "You are a world-class Python architect and developer. You have been given the entire legacy codebase below, written in various legacy languages. "
        "Your task is to produce a complete Python application that faithfully reproduces all functionality and behavior of the original system, while "
        "modernizing its architecture, improving its clarity, factoring the code into logical modules, and following Python best practices. "
        "Your Python application should:\n\n"
        "- Provide identical external behavior, including inputs, outputs, and business logic.\n"
        "- Utilize FastAPI for HTTP endpoints (if applicable), SQLAlchemy for database interactions, and pytest for testing.\n"
        "- Employ a clean, well-organized, and modular project structure (e.g., separate directories for routes, models, services, tests, etc.).\n"
        "- Include comments explaining the purpose of key sections of code.\n"
        "- Be lint-compliant, following PEP 8 guidelines.\n"
        "- Include sufficient code-level tests to ensure correctness.\n\n"
        "Output your solution as a set of files, using the following format:\n\n"
        "### filename.py\n"
        "```python\n"
        "# code here\n"
        "```\n\n"
        "Include only the code files as specified, with no additional commentary. "
        "Ensure the final solution can be directly run and tested with minimal configuration. "
        "Don't forget any of the business logic, models, or endpoints. Be perfectly thorough.\n\n"
        f"{combined_code}\n"
    )

    response_content = call_gemini(prompt)
    code_files = parse_code_files_from_response_multiple_files(response_content)
    strategy7_output_dir = os.path.join(output_dir, 'strategy7')
    os.makedirs(strategy7_output_dir, exist_ok=True)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(strategy7_output_dir, sanitized_file_name)
        if os.path.exists(output_path):
            print_step(f"Skipping existing file: {output_path}")
            continue
        print_step(f"Writing file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
    print_step("Strategy 7 completed.")


def translate_code(code):
    print_step("Using LLM to translate code.")
    prompt = (
        "As an expert software engineer proficient in both modern and historic programming languages, please convert the following legacy enterprise code to Python. "
        "Ensure functionality is preserved, simplify unnecessary complexity, and follow Python best practices. "
        "Provide only the converted Python code, without any explanations or additional text.\n\n"
        f"{code}\n"
    )
    response = call_gemini(prompt)
    translated_code = response.strip()
    return translated_code


def simplify_python_code(code):
    print_step("Using LLM to simplify Python code.")
    prompt = (
        "As an experienced Python developer, refactor the following code to enhance simplicity and efficiency. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for unit testing where appropriate. "
        "Ensure the refactored code preserves functionality, follows best practices, and is lint-compliant. "
        "Include explanations as comments within the code. Provide only the refactored code, without any explanations or additional text.\n\n"
        f"{code}\n"
    )
    response = call_gemini(prompt)
    simplified_code = response.strip()
    return simplified_code


def extract_project_description(input_files, project_dir, output_dir):
    print_step("Extracting project description from source files.")
    partial_descriptions = []
    for file_path in tqdm(input_files, desc="Summarizing files", unit="file"):
        relative_path = os.path.relpath(file_path, project_dir)
        output_path = os.path.join(output_dir, 'descriptions', f"{relative_path}.description")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            tqdm.write(f"Using existing summary for {file_path}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            prompt = (
                "As a software analyst, produce a concise yet exhaustive summary of the following source file. "
                "List key elements using the format '[Type] Name: Purpose', where Type is Class, Method, or Property. "
                "Include only essential information and use abbreviations to minimize tokens. "
                "Exclude boilerplate and unimportant details. Provide only the summary, without any additional text.\n\n"
                f"{code}\n"
            )
            summary = call_gemini(prompt)
            summary_text = summary.strip()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            tqdm.write(f"Summarized {file_path}")
        partial_descriptions.append(summary_text)
    combined_description = "\n".join(partial_descriptions)
    prompt = (
        "As a software analyst, based on the following file summaries, produce a concise and exhaustive high-level description "
        "of the project's overall functionality. Focus on main features, architecture, and key components and their interactions. "
        "Present the description in a structured format using bullet points or key-value pairs to minimize tokens. "
        "Provide only the project description, without any additional text.\n\n"
        f"{combined_description}\n"
    )
    project_description = call_gemini(prompt)
    project_description_text = project_description.strip()
    return project_description_text


def extract_project_description_from_python(python_files, project_dir, output_dir):
    print_step("Extracting project description from simplified Python source files.")
    partial_descriptions = []
    for file_path in tqdm(python_files, desc="Summarizing Python files", unit="file"):
        relative_path = os.path.relpath(file_path, project_dir)
        output_path = os.path.join(output_dir, 'descriptions', f"{relative_path}.description")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            tqdm.write(f"Using existing summary for {file_path}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            prompt = (
                "As a software engineer, summarize the following Python file in extremely terse language as if you're "
                "writing notes to yourself. Jot down the bare minimum, in compact language, that you'll need in order "
                "to create something similar later. Avoid mentioning any boilerplate; identify only the most "
                "important parts. Ignore setup and config that could be guessed if it were missing. Provide only your notes for the file, without any additional text.\n\n"
                f"# Filename: {relative_path}\n{code}\n"
            )
            summary = call_gemini(prompt)
            summary_text = summary.strip()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            tqdm.write(f"Summarized {file_path}")
        partial_descriptions.append(summary_text)
    combined_description = "\n".join(partial_descriptions)
    prompt = (
        "As a software analyst, based on the following file summaries, produce a concise and exhaustive high-level description "
        "of the project's overall functionality. Focus on main features, architecture, and key components and their interactions. "
        "Present the description in a structured format using bullet points or key-value pairs to minimize tokens. "
        "Provide only the project description, without any additional text.\n\n"
        f"{combined_description}\n"
    )
    project_description = call_gemini(prompt)
    project_description_text = project_description.strip()
    return project_description_text


def generate_high_level_design(project_description):
    print_step("Generating high-level design.")
    prompt = (
        "As a software architect, create a detailed high-level design for a Python implementation of the project described below. "
        "The design should focus on simplicity, efficiency, and adherence to Python best practices. "
        "Include suggestions for using SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for testing. "
        "Present the design in a structured format, outlining modules, classes, key functions, and their relationships. "
        "Divide the design into modules using the format 'Module: ModuleName'. Provide only the high-level design, without any additional text.\n\n"
        f"{project_description}\n"
    )
    response = call_gemini(prompt)
    design = response.strip()
    return design


def implement_python_project(design, output_dir):
    print_step("Implementing Python project based on the design.")
    prompt = (
        "As a world-class Python developer, implement the Python project based on the high-level design provided below. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for tests. "
        "Ensure the code follows Python conventions, is well-documented with comments, and passes linting. "
        "Fully implement each part of the system, with careful attention to detail. "
        "Provide the code files in the following format:\n\n"
        "### filename.py\n"
        "```python\n"
        "# code for filename.py\n"
        "```\n\n"
        "Provide only the code files as specified, without any additional text.\n\n"
        f"{design}\n"
    )
    response_content = call_gemini(prompt)
    code_files = parse_code_files_from_response_multiple_files(response_content)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(output_dir, sanitized_file_name)
        if os.path.exists(output_path):
            print_step(f"Skipping existing file: {output_path}")
            continue
        print_step(f"Writing file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)


def implement_python_project_modular(design, output_dir):
    print_step("Implementing Python project in a modular fashion based on the design.")
    modules = parse_modules_from_design(design)
    for module_name, module_design in modules.items():
        implement_module(module_name, module_design, output_dir)


def parse_modules_from_design(design):
    print_step("Parsing modules from high-level design.")
    modules = {}
    module_sections = re.split(r"Module:\s*(.*?)\n", design)
    for i in range(1, len(module_sections), 2):
        module_name = module_sections[i].strip()
        module_design = module_sections[i + 1].strip()
        modules[module_name] = module_design
    return modules


def implement_module(module_name, module_design, output_dir):
    print_step(f"Implementing module: {module_name}")
    prompt = (
        f"As an expert Python developer, implement the '{module_name}' module as described below. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for tests where appropriate. "
        "Ensure the code follows Python conventions, is well-documented with comments, and passes linting. "
        "Provide the code in the following format:\n\n"
        f"### {module_name}.py\n"
        "```python\n"
        "# code for {module_name}.py\n"
        "```\n\n"
        "Provide only the code as specified, without any additional text.\n\n"
        f"{module_design}\n"
    )
    response_content = call_gemini(prompt)
    code_files = parse_code_files_from_response_multiple_files(response_content)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(output_dir, sanitized_file_name)
        print_step(f"Writing file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)


def parse_code_files_from_response_multiple_files(response_content):
    print_step("Parsing code files from LLM response.")
    code_files = {}

    # Enhanced regex pattern to handle variations in LLM output
    pattern = (
        r"(?:(?:#{0,3})?\s*([\w./]+))"  # Matches file paths like core/services/requester.py
        r"\n?```(?:python)?\n"           # Matches optional 'python' code fence indicator
        r"(.*?)"                         # Captures the code content
        r"\n```"                         # Matches the closing backticks
    )

    matches = re.finditer(pattern, response_content, re.DOTALL)
    for match in matches:
        filename = match.group(1).strip()
        code = match.group(2).strip()
        code_files[filename] = code

    if not code_files:
        print_step("No code files were found in the LLM response.")
        filename = f"{uuid.uuid4()}.py"
        code_files[filename] = response_content
    return code_files


def generate_unit_tests(output_dir):
    print_step("Generating unit tests for the Python project.")
    python_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    for file_path in tqdm(python_files, desc="Generating unit tests", unit="file"):
        test_file_name = f'test_{os.path.basename(file_path)}'
        test_file_path = os.path.join(os.path.dirname(file_path), test_file_name)
        if os.path.exists(test_file_path):
            tqdm.write(f"Skipping existing test file: {test_file_path}")
            continue
        tqdm.write(f"Generating unit tests for {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        unit_test_code = generate_unit_test(code)
        if not unit_test_code:
            tqdm.write(f"Unit test generation failed for {file_path}. Skipping.")
            continue
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(unit_test_code)
    print_step("Unit test generation completed.")


def generate_unit_test(code):
    prompt = (
        "As an expert Python developer specializing in writing unit tests using pytest, write comprehensive unit tests for the following code. "
        "Ensure the tests cover significant functionality and edge cases. "
        "Include explanations as comments within the test code. Provide only the test code, without any additional text.\n\n"
        f"{code}\n"
    )
    unit_test_code = call_gemini(prompt)
    unit_test_code = unit_test_code.strip()
    return unit_test_code


def sanitize_filename(filename):
    filename = filename.lstrip('/\\')
    filename = os.path.normpath(filename)
    if '..' in filename or filename.startswith(('/', '\\')):
        filename = os.path.basename(filename)
    return filename


def evaluate_project(project_dir):
    print_step(f"Evaluating the project in {project_dir}.")
    num_files = 0
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                num_files += 1
    print_step(f"Found {num_files} Python files in {project_dir}.")
    return num_files


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


def main():
    parser = argparse.ArgumentParser(
        description='Port a project and produce various implementation strategies.'
    )
    parser.add_argument(
        'project_path', help='Path to the project git repository.'
    )
    parser.add_argument(
        '--output_dir',
        default='python_project',
        help='Directory to output the Python project.',
    )
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{project_path}-{MODEL_NAME}"

    print_step("Starting the porting process.")
    input_files = analyze_project(project_path)

    strategy_scores = {}

    # Strategy 1
    strategy1_output_dir = os.path.join(output_dir, 'strategy1')
    os.makedirs(strategy1_output_dir, exist_ok=True)
    strategy_file_by_file_translation(
        input_files, project_path, strategy1_output_dir
    )
    strategy_scores['strategy1'] = evaluate_project(strategy1_output_dir)

    # Strategy 1.1
    strategy1_1_output_dir = os.path.join(output_dir, 'strategy1_1')
    os.makedirs(strategy1_1_output_dir, exist_ok=True)
    strategy_simplify_python_app(strategy1_output_dir, strategy1_1_output_dir)
    strategy_scores['strategy1_1'] = evaluate_project(strategy1_1_output_dir)

    # Strategy 2
    strategy2_output_dir = os.path.join(output_dir, 'strategy2')
    os.makedirs(strategy2_output_dir, exist_ok=True)
    python_files = []
    for root, _, files in os.walk(strategy1_1_output_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    strategy_reimplement_from_python_summaries(
        python_files, strategy1_1_output_dir, strategy2_output_dir
    )
    strategy_scores['strategy2'] = evaluate_project(strategy2_output_dir)

    # Strategy 3
    strategy3_output_dir = os.path.join(output_dir, 'strategy3')
    os.makedirs(strategy3_output_dir, exist_ok=True)
    strategy_reimplement_from_design(
        input_files, project_path, strategy3_output_dir
    )
    strategy_scores['strategy3'] = evaluate_project(strategy3_output_dir)

    # Strategy 4
    strategy4_output_dir = os.path.join(output_dir, 'strategy4')
    os.makedirs(strategy4_output_dir, exist_ok=True)
    strategy_modular_implementation(
        input_files, project_path, strategy4_output_dir
    )
    strategy_scores['strategy4'] = evaluate_project(strategy4_output_dir)

    # Strategy 5 (Full Comprehensive Summary)
    strategy5_output_dir = os.path.join(output_dir, 'strategy5')
    os.makedirs(strategy5_output_dir, exist_ok=True)
    full_spec_file = strategy_full_summary(input_files, project_path, output_dir)
    # Strategy 5 does not produce executable code, so we do not evaluate it.

    # Strategy 6 (Reimplementation from Full Spec)
    strategy6_output_dir = os.path.join(output_dir, 'strategy6')
    os.makedirs(strategy6_output_dir, exist_ok=True)
    strategy_reimplement_from_full_spec(full_spec_file, output_dir)
    strategy_scores['strategy6'] = evaluate_project(os.path.join(output_dir, 'strategy6'))

    # Strategy 7 (Full Rewrite from Entire Source in One Prompt)
    strategy7_output_dir = os.path.join(output_dir, 'strategy7')
    os.makedirs(strategy7_output_dir, exist_ok=True)
    strategy_full_rewrite_from_entire_source(input_files, project_path, output_dir)
    strategy_scores['strategy7'] = evaluate_project(os.path.join(output_dir, 'strategy7'))

    # Select the best strategy (among the ones producing code)
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    print_step(f"Selected {best_strategy} as the best strategy.")

    # Generate unit tests for the best strategy
    best_output_dir = os.path.join(output_dir, best_strategy)
    generate_unit_tests(best_output_dir)

    print_step("Porting process completed.")


if __name__ == '__main__':
    main()
