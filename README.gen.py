#!/usr/bin/env python3
"""
Analyze a git repository using either Gemini 2.0 API or OpenAI's o1 model.

Idempotency Features:
- If plan.txt exists, do not re-generate the plan.
- Each step result is stored in plan_step_X.txt where X is the step index.
- If steps are already computed, they are not re-computed, enabling resuming from partial runs.

Steps:
1. Enumerate files in the repository.
2. Determine backend (Gemini if GEMINI_API_KEY, else OpenAI if OPENAI_API_TOKEN).
3. If plan.txt doesn't exist, generate the plan and save it.
4. For each step in the plan, if its result file exists, reuse it. Otherwise, run and save it.
5. Accumulate analysis from previous steps and pass it along.
6. Produce README.gen file at the final step.
"""

import os
import sys
import traceback
from tqdm import tqdm
from google import genai
import openai

MAX_RETRIES = 3


def print_error_and_exit(message):
    """Prints an error message and exits the script."""
    print(f"Error: {message}")
    sys.exit(1)


def infer(prompt, gemini_client=None, gemini_model_name=None):
    """
    Infer a response using either Gemini or OpenAI's o1 model, depending on which keys are set.

    Priority:
    - If GEMINI_API_KEY is set, use Gemini.
    - Else if OPENAI_API_TOKEN is set, use OpenAI.
    - If both are set, Gemini is preferred.
    - If none are set, error out.

    Uses basic retry logic. Returns the response text or exits on failure.
    """

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_token = os.environ.get("OPENAI_API_TOKEN")

    if not gemini_key and not openai_token:
        print_error_and_exit("Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set. Cannot proceed.")

    # Decide which backend to use
    use_gemini = bool(gemini_key)  # If gemini_key is present, prefer Gemini
    if not use_gemini and openai_token:
        # Use OpenAI
        openai.api_key = openai_token
        backend_name = "OpenAI o1"
    else:
        # Use Gemini
        backend_name = "Gemini"

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[DEBUG] Attempt {attempt}/{MAX_RETRIES} to call {backend_name} API.")
        try:
            if use_gemini:
                # Gemini inference
                response = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt.strip()
                )
                if not response.text:
                    print("[DEBUG] Received empty text response from Gemini API. Retrying...")
                    if attempt == MAX_RETRIES:
                        print_error_and_exit("No text returned by Gemini API after multiple attempts.")
                    continue
                return response.text.strip()
            else:
                # OpenAI inference
                completion = openai.ChatCompletion.create(
                    model="o1",
                    messages=[{"role": "user", "content": prompt.strip()}]
                )
                # Extract the content from the response
                if not completion or 'choices' not in completion or len(completion.choices) == 0:
                    print("[DEBUG] Received empty or malformed response from OpenAI. Retrying...")
                    if attempt == MAX_RETRIES:
                        print_error_and_exit("No text returned by OpenAI after multiple attempts.")
                    continue
                return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"[DEBUG] {backend_name} API call attempt {attempt} failed: {e}")
            traceback.print_exc()
            if attempt == MAX_RETRIES:
                print_error_and_exit(f"Failed to get a valid response from the {backend_name} API after multiple attempts.")

    # Should not reach here due to exit in loop
    return ""


def rebuild_accumulated_analysis(repo_path, num_steps):
    """
    Rebuild the accumulated analysis by reading all existing step files in order.
    Returns a string containing the concatenated analysis from completed steps.
    """
    analysis = ""
    for i in range(num_steps):
        step_file = os.path.join(repo_path, f"plan_step_{i}.txt")
        if os.path.exists(step_file):
            with open(step_file, "r", encoding="utf-8") as sf:
                step_output = sf.read().strip()
                analysis += "\n" + step_output
        else:
            break
    return analysis.strip()


def main():
    """Main entry point for the repository analysis script."""
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: python analyze_repo.py <path_to_git_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        print_error_and_exit("The provided path does not exist or is not a directory.")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_token = os.environ.get("OPENAI_API_TOKEN")

    if not gemini_key and not openai_token:
        print_error_and_exit(
            "Neither GEMINI_API_KEY nor OPENAI_API_TOKEN is set.\n"
            "Please set one before running this script."
        )

    # Initialize Gemini client if GEMINI_API_KEY is present
    gemini_client = None
    gemini_model_name = 'gemini-2.0-flash-exp'
    # gemini_model_name = 'gemini-1.5-pro'
    if gemini_key:
        print("[DEBUG] Initializing Gemini client...")
        try:
            gemini_client = genai.Client(api_key=gemini_key)
        except Exception as e:
            print_error_and_exit(f"Failed to initialize Gemini client: {str(e)}")

    # Enumerate all files in the repository
    print("[DEBUG] Enumerating files in the repository. This can be slow for large directories...")
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
        # No files found means nothing to analyze
        print("No files found in the provided repository path. Exiting.")
        sys.exit(0)

    file_listing = "\n".join(file_list)

    plan_file = os.path.join(repo_path, "plan.txt")

    # If plan.txt already exists, use it, else generate it
    if os.path.exists(plan_file):
        print("[DEBUG] plan.txt already exists. Using existing plan...")
        # Validate plan is not empty
        with open(plan_file, "r", encoding="utf-8") as pf:
            steps = [line.strip() for line in pf if line.strip()]
        if not steps:
            print_error_and_exit("plan.txt is empty or invalid.")
        # Check final step
        if not any("README.gen" in step for step in steps):
            print_error_and_exit("The plan does not contain a final step to write README.gen.")
    else:
        # Generate the plan
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

        print("[DEBUG] Generating the analysis plan. This may take a while...")
        plan_text = infer(plan_prompt, gemini_client=gemini_client, gemini_model_name=gemini_model_name)
        if not plan_text:
            print_error_and_exit("Received empty plan text from the inference method.")

        steps = [line.strip() for line in plan_text.split("\n") if line.strip()]
        if not steps:
            print_error_and_exit("The plan returned by the inference method is empty or invalid.")

        if not any("README.gen" in step for step in steps):
            print_error_and_exit("The plan does not contain a final step to write the README.gen file.")

        print(f"[DEBUG] Writing the plan to {plan_file}...")
        try:
            with open(plan_file, "w", encoding="utf-8") as pf:
                pf.write(plan_text + "\n")
        except IOError as e:
            print_error_and_exit(f"Error writing to plan file: {str(e)}")

    # (Re)read steps from the existing plan file to ensure consistency
    with open(plan_file, "r", encoding="utf-8") as pf:
        steps = [line.strip() for line in pf if line.strip()]

    readme_file = os.path.join(repo_path, "README.gen")

    # Determine which steps are already done
    # Steps stored as plan_step_0.txt, plan_step_1.txt, etc.
    completed_steps = 0
    for i in range(len(steps)):
        step_file = os.path.join(repo_path, f"plan_step_{i}.txt")
        if os.path.exists(step_file):
            completed_steps += 1
        else:
            break

    print(f"[DEBUG] Found {completed_steps} completed steps. Resuming from step {completed_steps}.")

    # Rebuild accumulated analysis from completed steps
    accumulated_analysis = rebuild_accumulated_analysis(repo_path, completed_steps)

    # Execute remaining steps
    for i in tqdm(range(completed_steps, len(steps)), desc="Executing plan steps"):
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

        if len(step_prompt) > 15000:
            print("[DEBUG] Warning: The prompt is very large. This could cause delays or truncation issues.")

        print(f"[DEBUG] Sending step prompt to inference method:\nStep {i}: {step}")
        step_output = infer(step_prompt, gemini_client=gemini_client, gemini_model_name=gemini_model_name)

        # Save step output to plan_step_i.txt for idempotency
        step_file_path = os.path.join(repo_path, f"plan_step_{i}.txt")
        try:
            with open(step_file_path, "w", encoding="utf-8") as sf:
                sf.write(step_output + "\n")
        except IOError as e:
            print_error_and_exit(
                f"Error writing {step_file_path}: {str(e)}.\n"
                "Make sure you have write permissions to the repository directory."
            )

        # Update accumulated analysis
        accumulated_analysis += "\n" + step_output

        # Check if this step indicates writing the README.gen
        final_step_keywords = ["README.gen", "final readme"]
        if any(keyword.lower() in step.lower() for keyword in final_step_keywords):
            print(f"[DEBUG] Final step detected. Writing final analysis to {readme_file}...")
            try:
                with open(readme_file, "w", encoding="utf-8") as rf:
                    rf.write(step_output + "\n")
            except IOError as e:
                print_error_and_exit(
                    f"Error writing README.gen: {str(e)}.\n"
                    "Make sure you have write permissions to the repository directory."
                )

    print(f"[DEBUG] Analysis complete. The final README is at: {readme_file}")


if __name__ == "__main__":
    main()
