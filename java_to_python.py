import os
import sys
import re
from openai import OpenAI

client = OpenAI()

# Set your model and optionally add temperature, max_tokens, etc.
MODEL = "o1-preview"


def gather_java_files_content(start_path):
    java_files_content = []
    # Walk the directory structure to find .java files
    for root, dirs, files in os.walk(start_path):
        for filename in files:
            if filename.endswith(".java"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                rel_path = os.path.relpath(full_path, start_path)
                java_files_content.append(f"# filename: {rel_path}\n{content}\n\n")
    return "".join(java_files_content)


def create_prompt(java_files_str):
    # Initial instructions and context
    initial_instructions = (
        "What follows is the contents of a Java application. You will be asked to port this to Python.\n"
        "Analyze it as a veteran application migration engineer and principal-level open source Python developer.\n"
    )

    # Final instructions after all files
    final_instructions = (
        "Port the above Java application to Python. Make it use SQLAlchemy, Alembic, FastAPI, and the blueprint pattern.\n"
        "It should pass PEP8 standards. Ensure the overall application is structured as a best-in-class modern Python "
        "web app with files, classes, services, functions, and variables all named for the role they play.\n"
        "Remove any unnecessary Java ceremonial code that Python does not require.\n"
    )

    # Combine everything into one prompt
    prompt = initial_instructions + java_files_str + final_instructions
    return prompt


def call_openai_api(prompt):
    messages = [
        {"role": "system", "content": "You are a senior engineering assistant."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0)

    return response.choices[0].message.content


def split_and_write_files(llm_output, output_dir):
    lines = llm_output.splitlines()
    current_file = None
    content = []

    for line in lines:
        # Detect comments with valid file paths
        if line.startswith("# filename: ") and re.match(r"# filename:\s[\w./\-]+\.py", line):
            # Write the content of the previous file if any
            if current_file and content:
                print(current_file)
                write_to_file(current_file, content, output_dir)
                content = []

            # Get the new file path
            current_file = line[len("# filename: "):].strip()
        else:
            content.append(line + "\n")

    # Write the last file's content if any
    if current_file and content:
        write_to_file(current_file, content, output_dir)


def write_to_file(file_path, content, output_dir):
    out_path = os.path.join(output_dir, file_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file:
        file.writelines(content)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_app.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    java_files_str = gather_java_files_content(input_dir)
    prompt = create_prompt(java_files_str)
    print(prompt)
    llm_output = call_openai_api(prompt)
    print(llm_output)
    split_and_write_files(llm_output, output_dir)
