#!/usr/bin/env python3
"""
Compare Few-Shot vs CoVe (Chain of Verification) prompting for TLA+ generation
"""
import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import argparse

# Load environment
project_root = Path(__file__).resolve().parent

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
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
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
    
    if backend == "ollama":
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1
        )
    elif backend == "openai":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
    elif backend == "anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0.1)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def load_few_shot_examples(num_examples=3):
    """Load few-shot examples from train.json"""
    train_path = project_root / "outputs" / "train.json"
    data_dir = project_root / "data"
    
    with open(train_path) as f:
        data = json.load(f)["data"]
    
    examples = []
    for entry in data[:num_examples * 2]:
        if len(examples) >= num_examples:
            break
            
        model_name = entry["model"]
        comments_file = entry.get("comments_clean")
        tla_file = entry.get("tla_clean")
        
        if not comments_file or not tla_file:
            continue
            
        comments_path = data_dir / model_name / "txt" / comments_file
        tla_path = data_dir / model_name / "tla" / tla_file
        
        if comments_path.exists() and tla_path.exists():
            examples.append({
                "model": model_name,
                "comments": comments_path.read_text().strip(),
                "tla": tla_path.read_text().strip()
            })
    
    return examples

def load_model_data(model_name: str):
    """Load comments and ground truth for a model"""
    data_dir = project_root / "data"
    all_models_path = data_dir / "all_models.json"
    
    with open(all_models_path) as f:
        all_models_data = json.load(f)
        all_models = all_models_data.get("data", all_models_data)
    
    model_data = next((m for m in all_models if m["model"] == model_name), None)
    if not model_data:
        raise ValueError(f"Model {model_name} not found")
    
    # Load comments
    comments_file = model_data.get("comments_clean")
    if comments_file:
        comments_path = data_dir / model_name / "txt" / comments_file
        if comments_path.exists():
            comments = comments_path.read_text().strip()
        else:
            raise FileNotFoundError(f"Comments not found: {comments_path}")
    else:
        raise ValueError(f"No comments_clean for {model_name}")
    
    return comments

# ==================== FEW-SHOT APPROACH ====================

def build_few_shot_prompt(comments: str, module_name: str, few_shot_examples: list) -> str:
    """Build few-shot learning prompt"""
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
    for ex in few_shot_examples[:2]:
        few_shot_text += f"""
=== Example: {ex['model']} ===
Comments:
{ex['comments'][:800]}...

TLA+ Specification:
{ex['tla'][:1500]}...

"""
    
    prompt = f"""{system}

{few_shot_text}

=== Your Task ===
Generate a TLA+ specification for module "{module_name}" based on these comments:

{comments}

Output ONLY the TLA+ code starting with ---- MODULE {module_name} ----"""
    
    return prompt

# ==================== CoVe APPROACH ====================

def build_cove_initial_prompt(comments: str, module_name: str) -> str:
    """Build CoVe initial generation prompt (no examples)"""
    return f"""You are an expert TLA+ specification writer. Generate a complete, syntactically correct TLA+ specification.

CRITICAL RULES:
1. Output ONLY valid TLA+ code - NO markdown, NO explanations
2. Start with exactly: ---- MODULE {module_name} ----
3. End with exactly: ====
4. Define ALL operators before using them
5. Use EXTENDS Integers, Sequences, etc. for standard operators
6. Every action must specify ALL variables with primed (') or UNCHANGED
7. Do NOT use UNCHANGED << >> (empty tuple is invalid)

Comments:
{comments}

Generate TLA+ specification for module "{module_name}":"""

def build_cove_verification_prompt(spec: str, module_name: str) -> str:
    """Build CoVe verification prompt"""
    return f"""You are a TLA+ expert reviewer. Analyze the following specification and identify ALL potential issues.

Specification:
{spec}

Generate a numbered list of verification questions to check for errors:
1. Are all operators defined before use?
2. Are all variables properly primed or marked UNCHANGED in actions?
3. Is the syntax valid (correct fairness notation, tuple syntax)?
4. Are there any undefined symbols?
5. Does the Spec formula correctly reference Init and Next?

For EACH question above, answer it with specific details from the code. If you find errors, describe them precisely."""

def build_cove_revision_prompt(spec: str, verification_result: str, module_name: str) -> str:
    """Build CoVe revision prompt based on verification"""
    return f"""Based on the verification analysis, revise the TLA+ specification to fix ALL identified issues.

Original Specification:
{spec}

Verification Analysis:
{verification_result}

Generate the CORRECTED specification. Output ONLY valid TLA+ code starting with ---- MODULE {module_name} ---- and NO explanations."""

# ==================== COMMON UTILITIES ====================

def clean_response(response: str, module_name: str) -> str:
    """Clean and fix common LLM output issues"""
    # Remove markdown code blocks
    if "```" in response:
        parts = response.split("```")
        for part in parts:
            if "MODULE" in part:
                response = part.strip()
                break
    
    # Remove thinking tags for reasoning models
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    
    # Fix ]_(vars) -> ]_<<vars>>
    response = re.sub(r'\]_\(([^)]+)\)', r']_<<\1>>', response)
    
    # Fix WF_next(Action)_<<vars>> -> WF_<<vars>>(Action)
    response = re.sub(r'WF_(\w+)\((\w+)\)_<<([^>]+)>>', r'WF_<<\3>>(\2)', response)
    response = re.sub(r'SF_(\w+)\((\w+)\)_<<([^>]+)>>', r'SF_<<\3>>(\2)', response)
    
    # Fix module name mismatches
    if f"MODULE {module_name}" not in response and "MODULE" in response:
        response = re.sub(r'MODULE \w+', f'MODULE {module_name}', response)
    
    # Remove UNCHANGED << >> (empty tuple)
    response = re.sub(r'UNCHANGED\s*<<\s*>>', '', response)
    
    return response.strip()

def run_sany(spec_content: str, module_name: str) -> tuple[bool, str]:
    """Run SANY parser on spec content"""
    tla_tools = Path(os.getenv("TLA_TOOLS_DIR", "/opt/tla")) / "tla2tools.jar"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        spec_file = tmpdir / f"{module_name}.tla"
        spec_file.write_text(spec_content)
        
        try:
            result = subprocess.run(
                ["java", "-cp", str(tla_tools), "tla2sany.SANY", str(spec_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Success: exit code 0 AND no fatal/errors, OR contains "Parsing completed"
            success = (result.returncode == 0 and "Fatal" not in output and "Error" not in output) \
                      or "Parsing completed" in output
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "SANY timeout (30s)"
        except Exception as e:
            return False, f"SANY error: {str(e)}"

# ==================== GENERATION FUNCTIONS ====================

def generate_few_shot(model_name: str, comments: str, few_shot_examples: list, llm) -> tuple[str, dict]:
    """Generate spec using few-shot prompting"""
    print(f"{BLUE}[Few-Shot]{RESET} Generating with {len(few_shot_examples)} examples...")
    
    prompt = build_few_shot_prompt(comments, model_name, few_shot_examples)
    
    start_time = datetime.now()
    response = llm.invoke(prompt)
    duration = (datetime.now() - start_time).total_seconds()
    
    spec = clean_response(response.content, model_name)
    
    success, output = run_sany(spec, model_name)
    
    metrics = {
        "approach": "few-shot",
        "num_examples": len(few_shot_examples),
        "generation_time": duration,
        "success": success,
        "spec_length": len(spec),
        "validation_output": output[:500]
    }
    
    return spec, metrics

def generate_cove(model_name: str, comments: str, llm) -> tuple[str, dict]:
    """Generate spec using Chain of Verification"""
    print(f"{CYAN}[CoVe]{RESET} Step 1: Initial generation...")
    
    # Step 1: Initial generation
    prompt1 = build_cove_initial_prompt(comments, model_name)
    start_time = datetime.now()
    response1 = llm.invoke(prompt1)
    time1 = (datetime.now() - start_time).total_seconds()
    
    initial_spec = clean_response(response1.content, model_name)
    
    # Step 2: Verification
    print(f"{CYAN}[CoVe]{RESET} Step 2: Verification analysis...")
    prompt2 = build_cove_verification_prompt(initial_spec, model_name)
    start_time = datetime.now()
    response2 = llm.invoke(prompt2)
    time2 = (datetime.now() - start_time).total_seconds()
    
    verification = response2.content
    
    # Step 3: Revision
    print(f"{CYAN}[CoVe]{RESET} Step 3: Revising based on verification...")
    prompt3 = build_cove_revision_prompt(initial_spec, verification, model_name)
    start_time = datetime.now()
    response3 = llm.invoke(prompt3)
    time3 = (datetime.now() - start_time).total_seconds()
    
    final_spec = clean_response(response3.content, model_name)
    
    # Validate final spec
    success, output = run_sany(final_spec, model_name)
    
    metrics = {
        "approach": "cove",
        "generation_time": time1,
        "verification_time": time2,
        "revision_time": time3,
        "total_time": time1 + time2 + time3,
        "success": success,
        "spec_length": len(final_spec),
        "verification_length": len(verification),
        "validation_output": output[:500]
    }
    
    return final_spec, metrics

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Compare Few-Shot vs CoVe prompting")
    parser.add_argument("--models", nargs="+", default=["DieHard", "Majority", "TwoPhase"],
                       help="Models to test")
    parser.add_argument("--num-examples", type=int, default=3,
                       help="Number of few-shot examples")
    parser.add_argument("--output", default="comparison_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    load_env()
    llm = get_llm()
    
    backend = os.getenv("LLM_BACKEND", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
    
    print(f"\n{'='*70}")
    print(f"{YELLOW}Few-Shot vs CoVe Comparison{RESET}")
    print(f"Backend: {backend} | Model: {model}")
    print(f"{'='*70}\n")
    
    # Load few-shot examples once
    print(f"Loading {args.num_examples} few-shot examples...")
    few_shot_examples = load_few_shot_examples(args.num_examples)
    print(f"Loaded examples: {[ex['model'] for ex in few_shot_examples]}\n")
    
    results = []
    
    for model_name in args.models:
        print(f"\n{YELLOW}{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}{RESET}\n")
        
        try:
            comments = load_model_data(model_name)
            
            # Test Few-Shot
            print(f"\n--- {BLUE}FEW-SHOT APPROACH{RESET} ---")
            fs_spec, fs_metrics = generate_few_shot(model_name, comments, few_shot_examples, llm)
            
            if fs_metrics["success"]:
                print(f"{GREEN}✓ Few-Shot PASSED{RESET} ({fs_metrics['generation_time']:.1f}s)")
            else:
                print(f"{RED}✗ Few-Shot FAILED{RESET} ({fs_metrics['generation_time']:.1f}s)")
                print(f"  Error: {fs_metrics['validation_output'][:200]}")
            
            # Test CoVe
            print(f"\n--- {CYAN}CoVe APPROACH{RESET} ---")
            cove_spec, cove_metrics = generate_cove(model_name, comments, llm)
            
            if cove_metrics["success"]:
                print(f"{GREEN}✓ CoVe PASSED{RESET} ({cove_metrics['total_time']:.1f}s total)")
            else:
                print(f"{RED}✗ CoVe FAILED{RESET} ({cove_metrics['total_time']:.1f}s total)")
                print(f"  Error: {cove_metrics['validation_output'][:200]}")
            
            # Save specs
            output_dir = project_root / "outputs" / "comparison" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            (output_dir / "few_shot.tla").write_text(fs_spec)
            (output_dir / "cove.tla").write_text(cove_spec)
            
            result = {
                "model": model_name,
                "few_shot": fs_metrics,
                "cove": cove_metrics,
                "winner": "few-shot" if fs_metrics["success"] and not cove_metrics["success"] 
                         else "cove" if cove_metrics["success"] and not fs_metrics["success"]
                         else "both" if fs_metrics["success"] and cove_metrics["success"]
                         else "neither"
            }
            results.append(result)
            
        except Exception as e:
            print(f"{RED}Error processing {model_name}: {e}{RESET}")
            results.append({
                "model": model_name,
                "error": str(e)
            })
    
    # Save results
    output_path = project_root / args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n\n{YELLOW}{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}{RESET}\n")
    
    fs_wins = sum(1 for r in results if r.get("winner") == "few-shot")
    cove_wins = sum(1 for r in results if r.get("winner") == "cove")
    both_wins = sum(1 for r in results if r.get("winner") == "both")
    neither = sum(1 for r in results if r.get("winner") == "neither")
    
    print(f"Few-Shot wins: {GREEN}{fs_wins}{RESET}")
    print(f"CoVe wins: {CYAN}{cove_wins}{RESET}")
    print(f"Both passed: {GREEN}{both_wins}{RESET}")
    print(f"Both failed: {RED}{neither}{RESET}")
    
    # Time comparison
    fs_times = [r["few_shot"]["generation_time"] for r in results if "few_shot" in r]
    cove_times = [r["cove"]["total_time"] for r in results if "cove" in r]
    
    if fs_times and cove_times:
        print(f"\nAverage time:")
        print(f"  Few-Shot: {sum(fs_times)/len(fs_times):.1f}s")
        print(f"  CoVe: {sum(cove_times)/len(cove_times):.1f}s")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
