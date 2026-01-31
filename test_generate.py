#!/usr/bin/env python3
"""
TLA+ Specification Generator with Self-Correction Loop
Retries with SANY error feedback if parsing fails
"""
import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Load environment
project_root = Path(__file__).resolve().parent

# ANSI colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def load_env():
    """Load environment variables from .env file"""
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")

def get_llm():
    """Initialize LLM based on configuration"""
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    
    backend = os.getenv("LLM_BACKEND", "ollama")
    model = os.getenv("LLM_MODEL", "deepseek-r1:7b")
    
    if backend == "ollama":
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1  # Slight randomness for retries
        )
    elif backend == "openai":
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    elif backend == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

def load_few_shot_examples(num_examples=3):
    """Load few-shot examples from train.json"""
    train_path = project_root / "outputs" / "train.json"
    data_dir = project_root / "data"
    
    with open(train_path) as f:
        data = json.load(f)["data"]
    
    examples = []
    for entry in data[:num_examples * 2]:  # Try more to get enough valid ones
        if len(examples) >= num_examples:
            break
            
        model_name = entry["model"]
        
        # Load comments
        comments_file = entry.get("comments_clean")
        if comments_file:
            comments_path = data_dir / model_name / "txt" / comments_file
            if comments_path.exists():
                comments = comments_path.read_text().strip()
            else:
                continue
        else:
            continue
        
        # Load TLA+ spec
        tla_file = entry.get("tla_clean")
        if tla_file:
            tla_path = data_dir / model_name / "tla" / tla_file
            if tla_path.exists():
                tla_spec = tla_path.read_text().strip()
            else:
                continue
        else:
            continue
        
        examples.append({
            "model": model_name,
            "comments": comments,
            "tla": tla_spec
        })
    
    return examples

def build_initial_prompt(comments: str, module_name: str, few_shot_examples: list) -> str:
    """Build the initial generation prompt"""
    system = f"""You are an expert TLA+ specification writer. Generate a complete, syntactically correct TLA+ specification.

CRITICAL RULES:
1. Output ONLY valid TLA+ code - NO markdown, NO explanations, NO thinking
2. Start with exactly: ---- MODULE {module_name} ----
3. End with exactly: ====
4. Define ALL operators before using them (e.g., define Min(a,b) before using it)
5. Use EXTENDS Integers, Sequences, etc. for standard operators
6. Every action must specify ALL variables with primed (') or UNCHANGED
7. Do NOT use UNCHANGED << >> (empty tuple is invalid)

COMMON TLA+ PATTERNS:
- Min(a,b) == IF a < b THEN a ELSE b
- Max(a,b) == IF a > b THEN a ELSE b
- For unchanged vars: UNCHANGED <<var1, var2>> or var1' = var1 /\\ var2' = var2
"""
    
    few_shot_text = ""
    for ex in few_shot_examples[:2]:  # Use 2 examples to keep prompt shorter
        few_shot_text += f"""
=== Example: {ex['model']} ===
Comments:
{ex['comments'][:1000]}

TLA+ Specification:
{ex['tla'][:2000]}

"""
    
    prompt = f"""{system}

{few_shot_text}

=== Your Task ===
Generate a TLA+ specification for module "{module_name}" based on these comments:

{comments}

Output ONLY the TLA+ code starting with ---- MODULE {module_name} ----"""
    
    return prompt

def build_repair_prompt(module_name: str, original_spec: str, error_output: str, attempt: int) -> str:
    """Build a prompt to fix SANY errors"""
    
    # Extract the most useful error info
    error_lines = []
    for line in error_output.split('\n'):
        line = line.strip()
        if line and any(x in line.lower() for x in ['error', 'unknown', 'expecting', 'unexpected', 'cannot find', 'undefined', 'line ', 'col ']):
            error_lines.append(line)
    
    error_summary = '\n'.join(error_lines[:15])  # Limit error lines
    
    prompt = f"""Fix this TLA+ specification. The SANY parser found errors.

ERRORS FROM SANY:
{error_summary}

BROKEN CODE:
{original_spec}

RULES:
1. Output ONLY the fixed TLA+ code
2. Module name MUST be: {module_name}
3. Do NOT include both old and new versions - only output the FIXED version
4. Do NOT use UNCHANGED if you already specify all primed variables
5. If you define var1' = X and var2' = Y, do NOT also add UNCHANGED <<var1, var2>>
6. UNCHANGED is ONLY needed when a variable does NOT change

Output the complete fixed specification now:"""
    
    return prompt

def clean_response(response: str, module_name: str) -> str:
    """Clean LLM response to extract pure TLA+ code"""
    # Remove thinking tags if present (for reasoning models)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Extract from code blocks if present
    if "```" in response:
        code_blocks = re.findall(r'```(?:tla\+?|tlaplus)?\n?(.*?)\n?```', response, re.DOTALL)
        if code_blocks:
            response = code_blocks[0]
    
    # Find module start
    if "---- MODULE" in response:
        start = response.find("---- MODULE")
        response = response[start:]
    
    # Fix module name to match expected filename
    module_match = re.search(r'---- MODULE (\w+)', response)
    if module_match:
        llm_module_name = module_match.group(1)
        if llm_module_name != module_name:
            response = re.sub(
                r'---- MODULE \w+ ----',
                f'---- MODULE {module_name} ----',
                response,
                count=1
            )
    
    # Ensure proper ending
    if "====" not in response:
        response += "\n===="
    
    # Fix common issues
    response = re.sub(r'UNCHANGED\s*<<\s*>>', 'TRUE', response)  # Empty UNCHANGED
    response = re.sub(r'\]_\(([^)]+)\)', r']_<<\1>>', response)  # Fix ]_(vars) to ]_<<vars>>
    response = re.sub(r'\]_([a-zA-Z_][a-zA-Z0-9_]*)\s*$', r']_<<\1>>', response, flags=re.MULTILINE)  # Fix ]_var to ]_<<var>>
    
    # Fix malformed WF/SF fairness conditions
    # WF_next(Action)_<<vars>> -> WF_<<vars>>(Action)
    response = re.sub(r'WF_\w+\((\w+)\)_<<([^>]+)>>', r'WF_<<\2>>(\1)', response)
    response = re.sub(r'SF_\w+\((\w+)\)_<<([^>]+)>>', r'SF_<<\2>>(\1)', response)
    # Remove lines with malformed fairness that couldn't be fixed
    response = re.sub(r'/\\.*WF_\w+\([^)]+\)_<<[^>]+>>.*\n', '', response)
    response = re.sub(r'/\\.*SF_\w+\([^)]+\)_<<[^>]+>>.*\n', '', response)
    
    return response.strip()

def run_sany(spec_content: str, module_name: str) -> tuple[bool, str]:
    """Run SANY parser on specification"""
    tla_tools_dir = os.getenv("TLA_TOOLS_DIR", "/opt/tla")
    jar_path = f"{tla_tools_dir}/tla2tools.jar"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / f"{module_name}.tla"
        spec_path.write_text(spec_content)
        
        try:
            result = subprocess.run(
                ["java", "-cp", jar_path, "tla2sany.SANY", str(spec_path)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout + "\n" + result.stderr
            # Success if exit code 0 OR "Parsing completed" in output OR "Semantic processing" completed without errors
            success = (result.returncode == 0 and "Fatal" not in output and "Error" not in output) or "Parsing completed" in output
            if result.returncode == 0 and "Semantic processing of module" in output and "error" not in output.lower():
                success = True
            return success, output
        except subprocess.TimeoutExpired:
            return False, "SANY timeout after 30 seconds"
        except Exception as e:
            return False, f"SANY error: {str(e)}"

def generate_with_retries(model_name: str, comments: str, max_retries: int = 3, save_output: bool = True):
    """Generate TLA+ spec with self-correction loop"""
    from langchain_core.messages import HumanMessage
    
    llm = get_llm()
    backend = os.getenv("LLM_BACKEND", "ollama")
    model = os.getenv("LLM_MODEL", "local")
    
    # Load few-shot examples
    num_shots = int(os.getenv("NUM_FEW_SHOTS", "3"))
    examples = load_few_shot_examples(num_shots)
    
    print(f"{BLUE}Generating TLA+ spec for: {model_name}{RESET}")
    print(f"Using {len(examples)} few-shot examples")
    print(f"Max retries: {max_retries}")
    print("-" * 60)
    
    # Initial generation
    print(f"\n{YELLOW}[Attempt 1/{max_retries + 1}] Initial generation...{RESET}")
    initial_prompt = build_initial_prompt(comments, model_name, examples)
    
    response = llm.invoke([HumanMessage(content=initial_prompt)])
    spec = clean_response(response.content, model_name)
    
    # Validate
    success, sany_output = run_sany(spec, model_name)
    
    attempt = 1
    history = [{
        "attempt": attempt,
        "spec": spec,
        "sany_pass": success,
        "sany_output": sany_output
    }]
    
    if success:
        print(f"{GREEN}✓ SANY passed on first try!{RESET}")
    else:
        print(f"{RED}✗ SANY failed{RESET}")
        # Show brief error
        for line in sany_output.split('\n'):
            if any(x in line.lower() for x in ['error', 'unknown', 'expecting']):
                print(f"  {line[:100]}")
    
    # Retry loop
    while not success and attempt <= max_retries:
        attempt += 1
        print(f"\n{YELLOW}[Attempt {attempt}/{max_retries + 1}] Repairing based on SANY errors...{RESET}")
        
        repair_prompt = build_repair_prompt(model_name, spec, sany_output, attempt)
        response = llm.invoke([HumanMessage(content=repair_prompt)])
        spec = clean_response(response.content, model_name)
        
        # Validate again
        success, sany_output = run_sany(spec, model_name)
        
        history.append({
            "attempt": attempt,
            "spec": spec,
            "sany_pass": success,
            "sany_output": sany_output
        })
        
        if success:
            print(f"{GREEN}✓ SANY passed after {attempt} attempts!{RESET}")
        else:
            print(f"{RED}✗ SANY still failing{RESET}")
            for line in sany_output.split('\n'):
                if any(x in line.lower() for x in ['error', 'unknown', 'expecting']):
                    print(f"  {line[:100]}")
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print(f"{GREEN}SUCCESS after {attempt} attempt(s){RESET}")
    else:
        print(f"{RED}FAILED after {attempt} attempt(s){RESET}")
    print("=" * 60)
    
    # Show final spec
    print(f"\n{BLUE}--- Final Specification ---{RESET}")
    print(spec[:1500])
    if len(spec) > 1500:
        print(f"... ({len(spec)} total characters)")
    
    # Save output
    if save_output:
        output_dir = project_root / "outputs" / f"{backend}_{model}" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{model_name}.tla"
        output_path.write_text(spec)
        print(f"\n{BLUE}Saved to: {output_path}{RESET}")
        
        # Save attempt history
        history_path = output_dir / f"{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"History saved to: {history_path}")
    
    return {
        "model": model_name,
        "spec": spec,
        "sany_pass": success,
        "attempts": attempt,
        "history": history
    }

def find_model_comments(model_name: str) -> str | None:
    """Find and load comments for a model"""
    data_dir = project_root / "data"
    val_path = project_root / "outputs" / "val.json"
    train_path = project_root / "outputs" / "train.json"
    
    model_entry = None
    for json_path in [val_path, train_path]:
        with open(json_path) as f:
            data = json.load(f)["data"]
            for entry in data:
                if entry["model"] == model_name:
                    model_entry = entry
                    break
        if model_entry:
            break
    
    if not model_entry:
        print(f"Model '{model_name}' not found in train/val data")
        return None
    
    # Load comments - try multiple paths
    comments_file = model_entry.get("comments_clean")
    if not comments_file:
        print(f"No comments_clean file for {model_name}")
        return None
    
    # Try direct path
    comments_path = data_dir / model_name / "txt" / comments_file
    if comments_path.exists():
        return comments_path.read_text().strip()
    
    # Try fallback search
    for path in data_dir.rglob(comments_file):
        return path.read_text().strip()
    
    print(f"Comments file not found: {comments_file}")
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TLA+ spec with self-correction")
    parser.add_argument("--model", type=str, default="DieHard",
                       help="Model name to generate (default: DieHard)")
    parser.add_argument("--retries", type=int, default=3,
                       help="Max retry attempts (default: 3)")
    parser.add_argument("--list", action="store_true",
                       help="List available models")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output")
    
    args = parser.parse_args()
    load_env()
    
    if args.list:
        val_path = project_root / "outputs" / "val.json"
        with open(val_path) as f:
            data = json.load(f)["data"]
        print("Available models in validation set:")
        for entry in data[:25]:
            print(f"  - {entry['model']}")
        if len(data) > 25:
            print(f"  ... and {len(data) - 25} more")
    else:
        comments = find_model_comments(args.model)
        if comments:
            result = generate_with_retries(
                args.model, 
                comments, 
                max_retries=args.retries,
                save_output=not args.no_save
            )
            
            # Exit code based on success
            exit(0 if result["sany_pass"] else 1)
        else:
            exit(1)
