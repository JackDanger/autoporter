import argparse
import os
import re
import threading
import time

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def print_step(step):
    print(f"[Step] {step}")


def analyze_dotnet_project(project_path):
    print_step("Analyzing the .NET project structure.")
    dotnet_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(('.cs', '.vb', '.fs')):
                file_path = os.path.join(root, file)
                dotnet_files.append(file_path)
    print_step(f"Found {len(dotnet_files)} .NET source files.")
    return dotnet_files


def strategy_file_by_file_translation(dotnet_files, project_path, output_dir):
    print_step("Starting Strategy 1: File-by-file translation.")
    for file_path in tqdm(dotnet_files, desc="Translating files"):
        relative_path = os.path.relpath(file_path, project_path)
        relative_path = os.path.normpath(relative_path)
        output_path = os.path.join(output_dir, relative_path)
        output_path = os.path.splitext(output_path)[0] + '.py'
        if os.path.exists(output_path):
            print_step(f"File {output_path} already exists. Skipping.")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        translated_code = translate_code(code)
        if not translated_code:
            print_step(f"Translation failed for {file_path}. Skipping.")
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
    for file_path in tqdm(python_files, desc="Simplifying files"):
        relative_path = os.path.relpath(file_path, strategy1_output_dir)
        output_path = os.path.join(output_dir, relative_path)
        if os.path.exists(output_path):
            print_step(f"File {output_path} already exists. Skipping.")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        simplified_code = simplify_python_code(code)
        if not simplified_code:
            print_step(f"Simplification failed for {file_path}. Skipping.")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simplified_code)
    print_step("Strategy 1.1 completed.")


def strategy_reimplement_from_python_summaries(python_files, project_dir, output_dir):
    print_step("Starting Strategy 2: Reimplementing from simplified Python code summaries.")
    project_description = extract_project_description_from_python(
        python_files, project_dir, output_dir
    )
    if not project_description:
        print_step("Failed to extract project description from Python code. Skipping Strategy 2.")
        return
    design = generate_high_level_design(project_description)
    if not design:
        print_step("Failed to generate high-level design. Skipping Strategy 2.")
        return
    implement_python_project(design, output_dir)
    print_step("Strategy 2 completed.")


def strategy_reimplement_from_design(dotnet_files, project_dir, output_dir):
    print_step("Starting Strategy 3: Reimplementing from high-level design based on C# summaries.")
    project_description = extract_project_description(
        dotnet_files, project_dir, output_dir
    )
    if not project_description:
        print_step("Failed to extract project description. Skipping Strategy 3.")
        return
    design = generate_high_level_design(project_description)
    if not design:
        print_step("Failed to generate high-level design. Skipping Strategy 3.")
        return
    implement_python_project(design, output_dir)
    print_step("Strategy 3 completed.")


def translate_code(code):
    print_step("Using LLM to translate code.")
    prompt = (
        "As an expert software engineer proficient in both C# and Python, "
        "convert the following C# code to Python, ensuring functionality is preserved. "
        "Simplify unnecessary complexity, follow Python best practices, and include explanatory comments. "
        "Provide only the Python code between <BEGIN_PYTHON_CODE> and <END_PYTHON_CODE> markers.\n\n"
        f"<BEGIN_CSHARP_CODE>\n{code}\n<END_CSHARP_CODE>\n\n<BEGIN_PYTHON_CODE>\n"
    )
    response = call_local_llm(prompt)
    translated_code = extract_code_from_response(
        response, "<BEGIN_PYTHON_CODE>", "<END_PYTHON_CODE>"
    )
    return translated_code


def simplify_python_code(code):
    print_step("Using LLM to simplify Python code.")
    prompt = (
        "As an experienced Python developer, refactor the following Python code to enhance simplicity and efficiency. "
        "Introduce SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for unit testing where appropriate. "
        "Ensure the refactored code preserves the original functionality, follows Python best practices, and is lint-compliant. "
        "Include explanations or notes as Python code comments. "
        "Provide only the refactored Python code between <BEGIN_REFACTORED_CODE> and <END_REFACTORED_CODE> markers.\n\n"
        f"<BEGIN_ORIGINAL_PYTHON_CODE>\n{code}\n<END_ORIGINAL_PYTHON_CODE>\n\n<BEGIN_REFACTORED_CODE>\n"
    )
    response = call_local_llm(prompt)
    simplified_code = extract_code_from_response(
        response, "<BEGIN_REFACTORED_CODE>", "<END_REFACTORED_CODE>"
    )
    return simplified_code


def extract_project_description(dotnet_files, project_dir, output_dir):
    print_step("Extracting project description from C# source files.")
    partial_descriptions = []
    for file_path in dotnet_files:
        relative_path = os.path.relpath(file_path, project_dir)
        output_path = f"{os.path.join(output_dir, 'descriptions', relative_path)}.description"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                summary_text = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            prompt = (
                "As a senior software analyst, provide a concise summary of the following C# code file. "
                "Focus on the file's purpose, its inputs and outputs (such as public classes, methods, properties), "
                "and important business logic. Ignore boilerplate and unimportant details.\n\n"
                f"<BEGIN_CSHARP_CODE>\n{code}\n<END_CSHARP_CODE>\n\n<BEGIN_FILE_SUMMARY>\n"
            )
            summary = call_local_llm(prompt)
            summary_text = extract_code_from_response(
                summary, "<BEGIN_FILE_SUMMARY>", "<END_FILE_SUMMARY>"
            )
            with open(output_path, 'w') as f:
                f.write(summary_text)
        print(f"Described {file_path}")
        partial_descriptions.append(summary_text)
    combined_description = "\n".join(partial_descriptions)
    prompt = (
        "As a senior software analyst, based on the following summaries of C# code files, provide a high-level description "
        "of the project's overall functionality, including key components and their interactions. "
        "Focus on the main features, architecture, and business logic of the application.\n\n"
        f"<BEGIN_FILE_SUMMARIES>\n{combined_description}\n<END_FILE_SUMMARIES>\n\n<BEGIN_PROJECT_DESCRIPTION>\n"
    )
    project_description = call_local_llm(prompt)
    project_description_text = extract_code_from_response(
        project_description, "<BEGIN_PROJECT_DESCRIPTION>", "<END_PROJECT_DESCRIPTION>"
    )
    return project_description_text


def extract_project_description_from_python(python_files, project_dir, output_dir):
    print_step("Extracting project description from simplified Python source files.")
    partial_descriptions = []
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, project_dir)
        output_path = f"{os.path.join(output_dir, 'descriptions', relative_path)}.description"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                summary_text = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            prompt = (
                "As a senior software analyst, provide a concise summary of the following Python code file. "
                "Focus on the file's purpose, its inputs and outputs (such as classes, functions, variables), "
                "and important business logic. Ignore boilerplate and unimportant details.\n\n"
                f"<BEGIN_PYTHON_CODE>\n{code}\n<END_PYTHON_CODE>\n\n<BEGIN_FILE_SUMMARY>\n"
            )
            summary = call_local_llm(prompt)
            summary_text = extract_code_from_response(
                summary, "<BEGIN_FILE_SUMMARY>", "<END_FILE_SUMMARY>"
            )
            with open(output_path, 'w') as f:
                f.write(summary_text)
        print(f"Described {file_path}")
        partial_descriptions.append(summary_text)
    combined_description = "\n".join(partial_descriptions)
    prompt = (
        "As a senior software analyst, based on the following summaries of Python code files, provide a high-level description "
        "of the project's overall functionality, including key components and their interactions. "
        "Focus on the main features, architecture, and business logic of the application.\n\n"
        f"<BEGIN_FILE_SUMMARIES>\n{combined_description}\n<END_FILE_SUMMARIES>\n\n<BEGIN_PROJECT_DESCRIPTION>\n"
    )
    project_description = call_local_llm(prompt)
    project_description_text = extract_code_from_response(
        project_description, "<BEGIN_PROJECT_DESCRIPTION>", "<END_PROJECT_DESCRIPTION>"
    )
    return project_description_text


def generate_high_level_design(project_description):
    print_step("Generating high-level design.")
    prompt = (
        "As a software architect, create a detailed high-level design for a Python implementation of the project described below. "
        "The design should focus on simplicity, efficiency, and adherence to Python best practices. "
        "Include suggestions for using SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for testing. "
        "Present the design in a structured format, outlining modules, classes, key functions, and their relationships.\n\n"
        f"<BEGIN_PROJECT_DESCRIPTION>\n{project_description}\n<END_PROJECT_DESCRIPTION>\n\n<BEGIN_HIGH_LEVEL_DESIGN>\n"
    )
    response = call_local_llm(prompt)
    design = extract_code_from_response(
        response, "<BEGIN_HIGH_LEVEL_DESIGN>", "<END_HIGH_LEVEL_DESIGN>"
    )
    return design


def implement_python_project(design, output_dir):
    print_step("Implementing Python project based on the design.")
    prompt = (
        "As an expert Python developer, implement the Python project based on the high-level design provided below. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for tests. "
        "Ensure the code follows Python conventions, is well-documented with comments, and passes linting. "
        "Provide the code files in the following format:\n\n"
        "<BEGIN_FILE: filename.py>\n<code>\n<END_FILE>\n\n"
        f"<BEGIN_HIGH_LEVEL_DESIGN>\n{design}\n<END_HIGH_LEVEL_DESIGN>\n\n<BEGIN_PYTHON_CODE>\n"
    )
    response_content = call_local_llm(prompt)
    code_files = parse_code_files_from_response_multiple_files(response_content)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(output_dir, sanitized_file_name)
        if os.path.exists(output_path):
            print_step(f"File {output_path} already exists. Skipping.")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)


def parse_code_files_from_response_multiple_files(response_content):
    print_step("Parsing code files from LLM response.")
    code_files = {}
    pattern = r"<BEGIN_FILE:\s*(.*?)\s*>\n(.*?)\n<END_FILE>"
    matches = re.finditer(pattern, response_content, re.DOTALL)
    for match in matches:
        filename = match.group(1).strip()
        code = match.group(2).strip()
        code_files[filename] = code
    return code_files


def generate_unit_tests(output_dir):
    print_step("Generating unit tests for the Python project.")
    python_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    for file_path in tqdm(python_files, desc="Generating unit tests"):
        test_file_name = f'test_{os.path.basename(file_path)}'
        test_file_path = os.path.join(os.path.dirname(file_path), test_file_name)
        if os.path.exists(test_file_path):
            print_step(f"File {test_file_path} already exists. Skipping.")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        unit_test_code = generate_unit_test(code)
        if not unit_test_code:
            print_step(f"Unit test generation failed for {file_path}. Skipping.")
            continue
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(unit_test_code)
    print_step("Unit test generation completed.")


def generate_unit_test(code):
    prompt = (
        "As an expert Python developer specializing in writing unit tests using pytest, write comprehensive unit tests for the following Python code. "
        "Ensure the tests cover all significant functionality and edge cases. "
        "Include test-related explanations as Python code comments. "
        "Provide only the test code between <BEGIN_UNIT_TEST> and <END_UNIT_TEST> markers.\n\n"
        f"<BEGIN_PYTHON_CODE>\n{code}\n<END_PYTHON_CODE>\n\n<BEGIN_UNIT_TEST>\n"
    )
    unit_test_code = call_local_llm(prompt)
    unit_test_code = extract_code_from_response(
        unit_test_code, "<BEGIN_UNIT_TEST>", "<END_UNIT_TEST>"
    )
    return unit_test_code


def extract_code_from_response(response, start_marker, end_marker):
    pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        code = match.group(1).strip()
        return code
    else:
        return response.strip()


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


def call_local_llm(prompt):
    max_retries = 3
    retry_delay = 1  # Start with 1-second delay
    max_new_tokens = 512

    for attempt in range(max_retries):
        try:
            print_step("Tokenizing input...")
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            total_input_tokens = input_ids.shape[-1]
            print_step(f"Input has {total_input_tokens} tokens.")

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                streamer=streamer,
            )

            print_step("Generating response...")
            generation_thread = threading.Thread(
                target=model.generate, kwargs=generation_kwargs
            )
            generation_thread.start()

            response = ''
            with tqdm(total=max_new_tokens, desc="Generating output", unit="token") as pbar:
                for new_text in streamer:
                    response += new_text
                    tokens_generated = len(tokenizer.encode(new_text, add_special_tokens=False))
                    pbar.update(tokens_generated)
            return response.strip()
        except Exception as e:
            print_step(f"Error generating response: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print_step("Failed to get a valid response from the local LLM.")
    return ''


def load_model():
    model_name = os.environ.get('MODEL', 'EleutherAI/gpt-neo-2.7B')
    print_step(f"Loading the local LLM model '{model_name}'. This may take some time...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(
        description='Port a .NET project to Python using a local LLM.'
    )
    parser.add_argument(
        'project_path', help='Path to the .NET project git repository.'
    )
    parser.add_argument(
        '--output_dir',
        default='python_project',
        help='Directory to output the Python project.',
    )
    args = parser.parse_args()

    global model, tokenizer, device
    model, tokenizer, device = load_model()

    project_path = os.path.abspath(args.project_path)
    output_dir = os.path.abspath(args.output_dir)

    print_step("Starting the porting process.")
    dotnet_files = analyze_dotnet_project(project_path)

    strategy_scores = {}

    # Strategy 1
    strategy1_output_dir = os.path.join(output_dir, 'strategy1')
    os.makedirs(strategy1_output_dir, exist_ok=True)
    strategy_file_by_file_translation(
        dotnet_files, project_path, strategy1_output_dir
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
    # Get the simplified Python files from strategy 1.1
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
        dotnet_files, project_path, strategy3_output_dir
    )
    strategy_scores['strategy3'] = evaluate_project(strategy3_output_dir)

    # Select the best strategy
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    print_step(f"Selected {best_strategy} as the best strategy.")

    # Generate unit tests for the best strategy
    best_output_dir = os.path.join(output_dir, best_strategy)
    generate_unit_tests(best_output_dir)

    print_step("Porting process completed.")


if __name__ == '__main__':
    main()
