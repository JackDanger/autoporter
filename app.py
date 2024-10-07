import os
import time
import argparse
from openai import OpenAI
from openai import RateLimitError, APIError

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Set your OpenAI API key


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
    for file_path in dotnet_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print_step(f"Translating {file_path}.")
        translated_code = translate_code(code)
        relative_path = os.path.relpath(file_path, project_path)
        relative_path = os.path.normpath(relative_path)
        output_path = os.path.join(output_dir, relative_path)
        output_path = os.path.splitext(output_path)[0] + '.py'
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
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print_step(f"Simplifying {file_path}.")
        simplified_code = simplify_python_code(code)
        relative_path = os.path.relpath(file_path, strategy1_output_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simplified_code)
    print_step("Strategy 1.1 completed.")


def strategy_reimplement_from_design(dotnet_files, output_dir):
    print_step("Starting Strategy 2: Reimplementing from high-level design.")
    project_description = extract_project_description(dotnet_files)
    print_step("Generating high-level design.")
    design = generate_high_level_design(project_description)
    print_step("Implementing Python project based on the design.")
    implement_python_project(design, output_dir)
    print_step("Strategy 2 completed.")


def translate_code(code):
    print_step("Using LLM to translate code.")
    prompt = (
        "You are an expert software engineer proficient in both C# and Python. "
        "Convert the following C# code to Python, ensuring that functionality is preserved. "
        "Simplify any unnecessary complexity and follow Python best practices and conventions. "
        "Provide well-formatted, lint-compliant Python code.\n\n"
        f"C# Code:\n{code}\n\nPython Code:"
    )
    response = call_openai_api(prompt)
    return response


def simplify_python_code(code):
    print_step("Using LLM to simplify Python code.")
    prompt = (
        "You are an experienced Python developer. Refactor the following Python code to enhance simplicity and efficiency. "
        "Introduce SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for unit testing where appropriate. "
        "Ensure the refactored code preserves the original functionality, follows Python best practices, and is lint-compliant.\n\n"
        f"Original Python Code:\n{code}\n\nRefactored Python Code:"
    )
    simplified_code = call_openai_api(prompt)
    return simplified_code


def extract_project_description(dotnet_files):
    print_step("Extracting project description from source files.")
    code_snippets = []
    for file_path in dotnet_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        code_snippets.append(code)
    combined_code = "\n".join(code_snippets)
    prompt = (
        "You are a senior software analyst. Based on the following C# codebase, provide a detailed, high-level description "
        "of the project's functionality, including key components and their interactions.\n\n"
        f"C# Codebase:\n{combined_code}\n\nProject Description:"
    )
    project_description = call_openai_api(prompt)
    return project_description


def generate_high_level_design(project_description):
    print_step("Generating high-level design.")
    prompt = (
        "You are a software architect. Using the following project description, create a detailed high-level design for a Python "
        "implementation of the project. The design should focus on simplicity, efficiency, and adherence to Python best practices. "
        "Include suggestions for using SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for testing.\n\n"
        f"Project Description:\n{project_description}\n\nHigh-Level Design:"
    )
    design = call_openai_api(prompt)
    return design


def implement_python_project(design, output_dir):
    print_step("Implementing Python project based on the design.")
    prompt = (
        "You are an expert Python developer. Implement the Python project based on the following high-level design. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for tests. "
        "Ensure the code follows Python conventions, is well-documented, and passes linting. "
        "Provide the code files in the format:\n\n[filename.py]\n<code>\n\n"
        f"High-Level Design:\n{design}\n\nPython Code:"
    )
    response_content = call_openai_api(prompt)
    code_files = parse_code_files_from_response(response_content)
    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(output_dir, sanitized_file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)


def parse_code_files_from_response(response_content):
    print_step("Parsing code files from LLM response.")
    code_files = {}
    lines = response_content.splitlines()
    current_filename = None
    code_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            if current_filename and code_lines:
                code_files[current_filename] = '\n'.join(code_lines).strip()
                code_lines = []
            current_filename = line[1:-1]
        else:
            code_lines.append(line)
    if current_filename and code_lines:
        code_files[current_filename] = '\n'.join(code_lines).strip()
    return code_files


def sanitize_filename(filename):
    filename = filename.lstrip('/\\')
    filename = os.path.normpath(filename)
    if '..' in filename or filename.startswith(('/', '\\')):
        filename = os.path.basename(filename)
    return filename


def generate_unit_tests(output_dir):
    print_step("Generating unit tests for the Python project.")
    python_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print_step(f"Generating unit test for {file_path}.")
        unit_test_code = generate_unit_test(code)
        test_file_name = f'test_{os.path.basename(file_path)}'
        test_file_path = os.path.join(os.path.dirname(file_path), test_file_name)
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(unit_test_code)


def generate_unit_test(code):
    prompt = (
        "You are an expert Python developer specializing in writing unit tests using pytest. "
        "Write comprehensive unit tests for the following Python code. Ensure the tests cover all significant functionality and edge cases. "
        "Provide the test code in a lint-compliant format.\n\n"
        f"Python Code:\n{code}\n\nPytest Unit Tests:"
    )
    unit_test_code = call_openai_api(prompt)
    return unit_test_code


def evaluate_project(project_dir):
    print_step(f"Evaluating the project in {project_dir}.")
    num_files = 0
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                num_files += 1
    print_step(f"Found {num_files} Python files in {project_dir}.")
    return num_files


def call_openai_api(prompt, model='gpt-4'):
    max_retries = 5
    retry_delay = 1  # Start with 1 second delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0)
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIError) as e:
            print_step(f"API error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print_step("Failed to get a valid response from OpenAI API.")
    return ''


def main():
    parser = argparse.ArgumentParser(description='Port a .NET project to Python using LLMs.')
    parser.add_argument('project_path', help='Path to the .NET project git repository.')
    parser.add_argument('--output_dir', default='python_project', help='Directory to output the Python project.')
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path)
    output_dir = os.path.abspath(args.output_dir)

    print_step("Starting the porting process.")
    dotnet_files = analyze_dotnet_project(project_path)

    strategy_scores = {}

    # Strategy 1
    strategy1_output_dir = os.path.join(output_dir, 'strategy1')
    os.makedirs(strategy1_output_dir, exist_ok=True)
    strategy_file_by_file_translation(dotnet_files, project_path, strategy1_output_dir)
    strategy_scores['strategy1'] = evaluate_project(strategy1_output_dir)

    # Strategy 1.1
    strategy1_1_output_dir = os.path.join(output_dir, 'strategy1_1')
    os.makedirs(strategy1_1_output_dir, exist_ok=True)
    strategy_simplify_python_app(strategy1_output_dir, strategy1_1_output_dir)
    strategy_scores['strategy1_1'] = evaluate_project(strategy1_1_output_dir)

    # Strategy 2
    strategy2_output_dir = os.path.join(output_dir, 'strategy2')
    os.makedirs(strategy2_output_dir, exist_ok=True)
    strategy_reimplement_from_design(dotnet_files, strategy2_output_dir)
    strategy_scores['strategy2'] = evaluate_project(strategy2_output_dir)

    # Select the best strategy
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    print_step(f"Selected {best_strategy} as the best strategy.")

    # Generate unit tests for the best strategy
    best_output_dir = os.path.join(output_dir, best_strategy)
    generate_unit_tests(best_output_dir)

    print_step("Porting process completed.")


if __name__ == '__main__':
    main()
