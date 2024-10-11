import argparse
import os
import re
import time
from tqdm import tqdm
import google.generativeai as genai


genai.configure(api_key=os.environ['API_KEY'])
model = genai.GenerativeModel(model_name='gemini-1.5-flash-8b')


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


def strategy1_full_project_reimplementation(dotnet_files, project_dir, output_dir):
    print_step("Starting Strategy 1: Full project reimplementation using the entire codebase.")
    code_snippets = []
    for file_path in tqdm(dotnet_files, desc="Reading .NET files", unit="file"):
        relative_path = os.path.relpath(file_path, project_dir)
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        snippet = f"// Filename: {relative_path}\n{code}\n"
        code_snippets.append(snippet)
    full_code = "\n".join(code_snippets)

    prompt = (
        "As an expert Python developer, please reimplement the following .NET project in Python. "
        "Preserve the project structure and functionality, and organize the code into appropriate Python modules and packages. "
        "Use SQLAlchemy for database interactions, FastAPI for HTTP endpoints, and pytest for unit testing where appropriate. "
        "Be comprehensive in keeping all the logic and refactor the implementation toward something simple and good. "
        "Make this like a million-dollar engineering migration, producing tests and thoroughly perfect code. "
        "Provide the code files in the following format:\n\n"
        "'Filename: relative/path/to/filename.py'\n<code>\n\n"
        "Provide only the code files as specified, without any additional text.\n\n"
        "Here is the entire .NET project code:\n\n"
        f"{full_code}\n"
    )

    response_content = call_gemini(prompt)

    code_files = parse_code_files_from_response_multiple_files(response_content)

    code_files = perform_cleanup_and_sanity_checks(code_files)

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
    print_step("Strategy 1 completed.")


def strategy2_improve_project(strategy1_output_dir, output_dir):
    print_step("Starting Strategy 2: Enhancing the project with monitoring, security, and dependencies.")
    python_files = []
    for root, _, files in os.walk(strategy1_output_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)

    code_snippets = []
    for file_path in tqdm(python_files, desc="Reading Python files", unit="file"):
        relative_path = os.path.relpath(file_path, strategy1_output_dir)
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        snippet = f"# Filename: {relative_path}\n{code}\n"
        code_snippets.append(snippet)
    full_code = "\n".join(code_snippets)

    prompt = (
        "As an expert Python developer with a focus on monitoring, security, and best practices, please review and enhance the following Python project code. "
        "Ensure the application is secure, efficient, and includes necessary dependencies using Poetry for dependency management. "
        "Add logging, monitoring capabilities, and address any security concerns. "
        "Consider code optimization, error handling, and adherence to PEP8 standards. "
        "Provide the updated code files in the following format:\n\n"
        "'Filename: relative/path/to/filename.py'\n<updated code>\n\n"
        "Provide only the code files as specified, without any additional text.\n\n"
        "Here is the existing Python project code:\n\n"
        f"{full_code}\n"
    )

    response_content = call_gemini(prompt)

    code_files = parse_code_files_from_response_multiple_files(response_content)

    code_files = perform_cleanup_and_sanity_checks(code_files)

    for file_name, code in code_files.items():
        sanitized_file_name = sanitize_filename(file_name)
        output_path = os.path.join(output_dir, sanitized_file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print_step(f"Updated file: {output_path}")
    print_step("Strategy 2 completed.")


def parse_code_files_from_response_multiple_files(response_content, recursion_depth=0, max_recursion_depth=3):
    print_step("Parsing code files from LLM response.")
    code_files = {}
    pattern = r"'Filename:\s*(.*?)'\n(.*?)(?=(?:'Filename:|$))"
    matches = re.finditer(pattern, response_content, re.DOTALL)
    last_match_end = 0
    for match in matches:
        filename = match.group(1).strip()
        code = match.group(2).strip()
        code_files[filename] = code
        last_match_end = match.end()
    unparsed_parts = response_content[last_match_end:].strip()
    if unparsed_parts and recursion_depth < max_recursion_depth:
        print_step("Found unparsed parts in the response.")
        additional_code_files = clarify_unparsed_parts(unparsed_parts, recursion_depth + 1, max_recursion_depth)
        code_files.update(additional_code_files)
    return code_files


def clarify_unparsed_parts(unparsed_parts, recursion_depth, max_recursion_depth):
    print_step("Using LLM to clarify unparsed parts.")
    prompt = (
        "The following text was generated as part of a code generation task but was not properly formatted. "
        "Please extract any code files from it, and provide them in the following format:\n\n"
        "'Filename: relative/path/to/filename.py'\n<code>\n\n"
        "Provide only the code files as specified, without any additional text.\n\n"
        f"{unparsed_parts}\n"
    )
    response = call_gemini(prompt)
    additional_code_files = parse_code_files_from_response_multiple_files(response, recursion_depth, max_recursion_depth)
    return additional_code_files


def perform_cleanup_and_sanity_checks(code_files):
    print_step("Performing cleanup and sanity checks on the code files.")
    cleaned_code_files = {}
    for filename, code in code_files.items():
        if not code.strip():
            print_step(f"Code for file {filename} is empty. Skipping.")
            continue
        cleaned_code_files[filename] = code
    return cleaned_code_files


def sanitize_filename(filename):
    filename = filename.lstrip('/\\')
    filename = os.path.normpath(filename)
    if '..' in filename or filename.startswith(('/', '\\')):
        filename = os.path.basename(filename)
    return filename


def call_gemini(prompt):
    max_retries = 15
    retry_delay = 1  # Start with 1-second delay

    for attempt in range(max_retries):
        try:
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=1,
                    temperature=0.8,
                    max_output_tokens=8192,
                )
            ).text
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
        description='Port a .NET project to Python using the Google PaLM API.'
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

    project_path = os.path.abspath(args.project_path)
    output_dir = os.path.abspath(args.output_dir)

    print_step("Starting the porting process.")
    dotnet_files = analyze_dotnet_project(project_path)

    # Strategy 1
    strategy1_output_dir = os.path.join(output_dir, 'strategy1')
    os.makedirs(strategy1_output_dir, exist_ok=True)
    strategy1_full_project_reimplementation(
        dotnet_files, project_path, strategy1_output_dir
    )

    # Strategy 2
    strategy2_output_dir = os.path.join(output_dir, 'strategy2')
    os.makedirs(strategy2_output_dir, exist_ok=True)
    strategy2_improve_project(strategy1_output_dir, strategy2_output_dir)

    print_step("Porting process completed.")


if __name__ == '__main__':
    main()
