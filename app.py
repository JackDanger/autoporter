import os
import argparse
import requests
import json


def print_step(step):
    print(f"[Step] {step}")


def analyze_dotnet_project(project_path):
    print_step("Analyzing the .NET project structure.")
    dotnet_files = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(('.cs', '.vb', '.fs')):
                file_path = os.path.join(root, file)
                dotnet_files.append(file_path)
    print_step(f"Found {len(dotnet_files)} .NET source files.")
    return dotnet_files


def strategy_file_by_file_translation(dotnet_files, output_dir):
    print_step("Starting Strategy 1: File-by-file translation.")
    for file_path in dotnet_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print_step(f"Translating {file_path}.")
        translated_code = translate_code(code)
        relative_path = os.path.relpath(file_path)
        output_path = os.path.join(output_dir, relative_path)
        output_path = os.path.splitext(output_path)[0] + '.py'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_code)
    print_step("Strategy 1 completed.")


def strategy_simplify_python_app(strategy1_output_dir, output_dir):
    print_step("Starting Strategy 1.1: Simplifying the Python app.")
    python_files = []
    for root, dirs, files in os.walk(strategy1_output_dir):
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
    prompt = f"""You are an expert in converting .NET code to Python code, ensuring functionality is preserved and unnecessary complexity is simplified.

Convert the following .NET code to Python, simplifying unnecessary complexity and including necessary comments:

{code}
"""
    response = query_ollama(prompt)
    return response


def simplify_python_code(code):
    print_step("Using LLM to simplify Python code.")
    prompt = f"""You are an expert Python developer. Refactor the following Python code to simplify unnecessary complexity. Introduce SQLAlchemy for all database connections, FastAPI for all HTTP endpoints, and pytest for all tests.

Ensure that the functionality is preserved, and provide necessary comments:

{code}
"""
    simplified_code = query_ollama_iterative(prompt)
    return simplified_code


def extract_project_description(dotnet_files):
    print_step("Extracting project description from source files.")
    code_snippets = []
    for file_path in dotnet_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        code_snippets.append(code)
    combined_code = "\n".join(code_snippets)
    prompt = f"""You are an expert software analyst.

Provide a high-level description of the project's functionality based on the following code:

{combined_code}
"""
    project_description = query_ollama(prompt)
    return project_description


def generate_high_level_design(project_description):
    print_step("Generating high-level design.")
    prompt = f"""You are an expert software architect.

Based on the following project description, create a high-level design for a Python implementation, focusing on simplicity and efficiency:

{project_description}
"""
    design = query_ollama(prompt)
    return design


def implement_python_project(design, output_dir):
    print_step("Implementing Python project based on the design.")
    prompt = f"""You are an expert Python developer.

Implement the Python project based on the following design, include code files and necessary comments. Use SQLAlchemy for database connections, FastAPI for HTTP endpoints, and pytest for tests. Provide the code files in the format:

[filename.py]
<code>

{design}
"""
    response_content = query_ollama_iterative(prompt)
    code_files = parse_code_files_from_response(response_content)
    for file_name, code in code_files.items():
        output_path = os.path.join(output_dir, file_name)
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
            # Save previous file
            if current_filename and code_lines:
                code_files[current_filename] = '\n'.join(code_lines).strip()
                code_lines = []
            # Start new file
            current_filename = line[1:-1]  # Remove square brackets
        else:
            code_lines.append(line)
    # Save the last file
    if current_filename and code_lines:
        code_files[current_filename] = '\n'.join(code_lines).strip()
    return code_files


def generate_unit_tests(output_dir):
    print_step("Generating unit tests for the Python project.")
    python_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        print_step(f"Generating unit test for {file_path}.")
        unit_test_code = generate_unit_test(code, file_path)
        test_file_name = f'test_{os.path.basename(file_path)}'
        test_file_path = os.path.join(os.path.dirname(file_path), test_file_name)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(unit_test_code)


def generate_unit_test(code, file_path):
    prompt = f"""You are an expert in writing Python unit tests using pytest framework.

Write unit tests for the following Python code:

{code}
"""
    unit_test_code = query_ollama_iterative(prompt)
    return unit_test_code


def evaluate_project(project_dir):
    print_step(f"Evaluating the project in {project_dir}.")
    # Simple evaluation: count the number of Python files generated
    num_files = 0
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                num_files += 1
    print_step(f"Found {num_files} Python files in {project_dir}.")
    return num_files  # Higher is better in this simple metric


def query_ollama(prompt, model='llama3.1:8b'):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'prompt': prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
    if response.status_code == 200:
        generated_text = ''
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode('utf-8'))
                    generated_text += json_data.get('response', '')
                except json.JSONDecodeError:
                    continue
        return generated_text.strip()
    else:
        print_step(f"Error querying Ollama API: {response.status_code} {response.text}")
        return ''


def query_ollama_iterative(prompt, model='llama3.1:8b'):
    # Iteratively refine the response to ensure correctness
    response = ''
    max_iterations = 5
    for i in range(max_iterations):
        print_step(f"Iteration {i+1} for prompt.")
        partial_response = query_ollama(prompt, model=model)
        # Check if the response meets the criteria (e.g., correct format)
        if validate_response(partial_response):
            response = partial_response
            break
        else:
            # Refine the prompt with feedback
            prompt += "\n\nPlease ensure the response is correctly formatted and complete."
    return response.strip()


def validate_response(response):
    # Simple validation to check if response is non-empty
    return bool(response.strip())


def main():
    parser = argparse.ArgumentParser(description='Port a .NET project to Python using LLMs.')
    parser.add_argument('project_path', help='Path to the .NET project git repository.')
    parser.add_argument('--output_dir', default='python_project', help='Directory to output the Python project.')
    args = parser.parse_args()

    project_path = args.project_path
    output_dir = args.output_dir

    print_step("Starting the porting process.")
    dotnet_files = analyze_dotnet_project(project_path)

    # Consider multiple strategies
    strategy_scores = {}

    # Strategy 1
    strategy1_output_dir = os.path.join(output_dir, 'strategy1')
    os.makedirs(strategy1_output_dir, exist_ok=True)
    strategy_file_by_file_translation(dotnet_files, strategy1_output_dir)
    # Evaluate Strategy 1
    strategy_scores['strategy1'] = evaluate_project(strategy1_output_dir)

    # Strategy 1.1
    strategy1_1_output_dir = os.path.join(output_dir, 'strategy1_1')
    os.makedirs(strategy1_1_output_dir, exist_ok=True)
    strategy_simplify_python_app(strategy1_output_dir, strategy1_1_output_dir)
    # Evaluate Strategy 1.1
    strategy_scores['strategy1_1'] = evaluate_project(strategy1_1_output_dir)

    # Strategy 2
    strategy2_output_dir = os.path.join(output_dir, 'strategy2')
    os.makedirs(strategy2_output_dir, exist_ok=True)
    strategy_reimplement_from_design(dotnet_files, strategy2_output_dir)
    # Evaluate Strategy 2
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
