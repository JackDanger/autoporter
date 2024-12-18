#!/usr/bin/env python3
"""
Analyze a git repository using the Gemini 2.0 API.

Steps:
1. Enumerate files in the repository.
2. Use the Gemini API to generate a plan.
3. Execute each step in the plan with the Gemini API, accumulating analysis.
4. Produce a README.gen file with a comprehensive analysis.
"""

import os
import sys
import traceback
from tqdm import tqdm
from google import genai


def main():
    """Main entry point for the repository analysis script."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_repo.py <path_to_git_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        print("The provided path does not exist or is not a directory.")
        sys.exit(1)

    # Check for the API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Initialize Gemini client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("Failed to initialize Gemini client:", str(e))
        sys.exit(1)

    model_name = 'gemini-2.0-flash-exp'

    # Enumerate all files in the repository
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        # Exclude .git directories
        if '.git' in root:
            continue
        for filename in files:
            if filename.startswith('.git'):
                continue
            full_path = os.path.join(root, filename)
            file_list.append(full_path)

    if not file_list:
        print("No files found in the provided repository path.")
        sys.exit(0)

    file_listing = "\n".join(file_list)

    plan_prompt = f"""
You are an AI assistant specialized in analyzing source code repositories.
The repository files are listed below. You will create a step-by-step plan 
for analyzing the repository.

Each step should be a line in the plan, and each step should be a clearly 
defined action. The final goal is to produce a file called 'README.gen' that 
is a thoroughly reasoned analysis of the repo.

Analysis requirements:
- Identify the kind(s) of application(s) in the repo (e.g., web service, 
  command-line tool, library, etc.)
- Identify data models used, including any classes, ORM models, or schemas.
- Identify any HTTP server endpoints, including their paths and methods 
  if possible.
- Identify any external services or APIs the code calls out to, and how.
- Summarize the architecture and main functionalities.
- Note any build or deployment processes if discoverable 
  (e.g., Dockerfiles, CI/CD configs).
- Note any testing frameworks or tests discovered.
- Conclude with an insightful evaluation of the code quality, maintainability, 
  and possible areas of improvement.

The steps in the plan should start from reading files, organizing them by type 
or role, then analyzing their contents with respect to architecture, endpoints, 
data models, external calls, etc. The final step should clearly say something 
like "Write the final README.gen file with the comprehensive analysis."

Do not include the code listing in the final README; the README should be an 
analysis and summary, not just a code dump.

Files in the repo:
{file_listing}

Please produce a list of actions (one per line) describing your intended 
approach. The final step should clearly say something like: 
"Write the final README.gen file with the comprehensive analysis."
"""

    # Generate the plan using Gemini
    try:
        plan_response = client.models.generate_content(
            model=model_name,
            contents=plan_prompt.strip()
        )
        plan_text = plan_response.text.strip()
    except Exception as e:
        print("Error generating analysis plan:", str(e))
        sys.exit(1)

    plan_file = os.path.join(repo_path, "plan.txt")
    try:
        with open(plan_file, "w", encoding="utf-8") as pf:
            pf.write(plan_text + "\n")
    except IOError as e:
        print("Error writing to plan file:", str(e))
        sys.exit(1)

    # Now we execute the plan step-by-step.
    # For each step, prompt Gemini with the step instructions and the current
    # accumulated analysis. Finally, produce README.gen.
    try:
        with open(plan_file, "r", encoding="utf-8") as pf:
            steps = [line.strip() for line in pf if line.strip()]
    except IOError as e:
        print("Error reading plan file:", str(e))
        sys.exit(1)

    readme_file = os.path.join(repo_path, "README.gen")
    accumulated_analysis = ""

    # Execute each plan step
    for i in tqdm(range(len(steps)), desc="Executing plan steps"):
        step = steps[i]
        step_prompt = f"""
You are an AI assistant. You have a repository of code files.
You have the following accumulated analysis so far:
{accumulated_analysis}

Now, you must perform the following step from the plan:

Step: "{step}"

You have the following files in the repository:
{file_listing}

Be extremely clear and verbose in how you derive your reasoning.
If this step involves analyzing code, provide insights.
If this step involves summarizing architecture, do so carefully.
If this step involves identifying endpoints, data models, or external calls, 
list them out.
If this step involves finalizing the analysis into README.gen, then produce 
the final, polished analysis ready for the README.

Make sure the response is well-structured and easy to read.
"""

        try:
            step_response = client.models.generate_content(
                model=model_name,
                contents=step_prompt.strip()
            )
        except Exception as e:
            # Handle any exceptions in requesting model content
            print("Error generating step response:", str(e))
            traceback.print_exc()
            sys.exit(1)

        step_output = step_response.text.strip()

        # Check if this step indicates writing the README.gen
        final_step_keywords = ["README.gen", "final README"]
        if any(keyword.lower() in step.lower() for keyword in final_step_keywords):
            # This is the final step: write the output to README.gen
            try:
                with open(readme_file, "w", encoding="utf-8") as rf:
                    rf.write(step_output + "\n")
            except IOError as e:
                print("Error writing README.gen:", str(e))
                sys.exit(1)
        else:
            # Accumulate analysis for the next steps
            accumulated_analysis += "\n" + step_output

    print(f"Analysis complete. The final README is at: {readme_file}")


if __name__ == "__main__":
    main()
