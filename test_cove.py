#!/usr/bin/env python3
"""
TLA+ Specification Generator with Chain of Verification (CoVe) Prompting
Uses self-verification before SANY validation, then repairs if needed
"""
import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

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
    model = os.getenv("LLM_MODEL", "llama3.3:latest")
    
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

def load_few_shot_examples(num_examples=2):
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
        
        if comments_file and tla_file:
            comments_path = data_dir / model_name / "txt" / comments_file
            tla_path = data_dir / model_name / "tla" / tla_file
            if comments_path.exists() and tla_path.exists():
                examples.append({
                    "model": model_name,
                    "comments": comments_path.read_text().strip()[:1200],
                    "tla": tla_path.read_text().strip()[:2000]
                })
    return examples

# =============================================================================
# CHAIN OF VERIFICATION PROMPTS
# =============================================================================

def build_generation_prompt(comments: str, module_name: str, examples: list) -> str:
    """Step 1: Initial generation prompt"""
    few_shot = ""
    for ex in examples[:2]:
        few_shot += f"\n--- Example: {ex['model']} ---\nComments:\n{ex['comments']}\n\nTLA+:\n{ex['tla']}\n"
    
    return f"""You are an expert TLA+ specification writer.

TASK: Generate a complete TLA+ specification for module "{module_name}".

STRICT SYNTAX RULES:
1. Output ONLY valid TLA+ code (not PlusCal)
2. Start with: ---- MODULE {module_name} ----
3. End with: ====
4. EXTENDS only: Integers, Naturals, Sequences, FiniteSets, TLC
5. Define operators with "==" like: Min(a,b) == IF a < b THEN a ELSE b
6. Do NOT use "Define" blocks (that's PlusCal, not TLA+)
7. Use <<var1, var2>> for tuples
8. Use # or /= for not-equal (not \\neq)

CORRECT Min/Max DEFINITIONS:
Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

{few_shot}

--- Your Task ---
Comments:
{comments}

Generate the TLA+ specification (remember: no "Define" blocks, use "==" for definitions):"""


def build_verification_questions_prompt(spec: str, module_name: str) -> str:
    """Step 2: Generate verification questions about the spec"""
    return f"""Analyze this TLA+ specification for syntax and semantic errors.

SPECIFICATION:
{spec}

Generate 6 verification questions checking for these SPECIFIC issues:

1. Does the module start with "---- MODULE {module_name} ----" and end with "===="?
2. Are all operator definitions using "==" syntax (e.g., "Op(x) == ...")? (NOT using "Define" blocks which are PlusCal, not TLA+)
3. Does EXTENDS only include valid standard modules (Integers, Naturals, Sequences, FiniteSets, TLC)?
4. Are Min/Max defined correctly as "Min(a,b) == IF a < b THEN a ELSE b"? (NOT using /\\ or \\/ in the definition)
5. Are tuples written with angle brackets <<x, y>> not parentheses (x, y)?
6. Is the Spec formula correct with "[][Next]_<<vars>>" using angle brackets?

Output these 6 questions numbered 1-6:"""


def build_verification_answers_prompt(spec: str, questions: str) -> str:
    """Step 3: Answer verification questions independently"""
    return f"""Carefully verify this TLA+ specification by answering each question.

SPECIFICATION:
{spec}

VERIFICATION QUESTIONS:
{questions}

For each question, answer with:
- "YES" if the specification satisfies the requirement
- "NO: [specific issue]" if there's a problem

Answer each question carefully:"""


def build_refinement_prompt(spec: str, verification_results: str, module_name: str) -> str:
    """Step 4: Refine based on verification results"""
    return f"""Refine this TLA+ specification based on the verification results.

ORIGINAL SPECIFICATION:
{spec}

VERIFICATION RESULTS:
{verification_results}

Fix ALL issues identified as "NO" in the verification.
The module name MUST be: {module_name}

RULES:
- Output ONLY the corrected TLA+ code
- Start with: ---- MODULE {module_name} ----
- End with: ====
- Use <<var1, var2>> for tuples
- Define all operators before use

Output the refined specification:"""


def build_sany_repair_prompt(spec: str, sany_errors: str, module_name: str) -> str:
    """Repair prompt based on SANY errors"""
    error_lines = [l.strip() for l in sany_errors.split('\n') 
                   if any(x in l.lower() for x in ['error', 'unknown', 'expecting', 'line ', 'col '])]
    
    return f"""Fix the TLA+ specification based on these SANY parser errors.

SANY ERRORS:
{chr(10).join(error_lines[:12])}

BROKEN SPECIFICATION:
{spec}

RULES:
- Module name MUST be: {module_name}
- Output ONLY the fixed TLA+ code
- Use <<var1, var2>> for tuples (NOT parentheses)
- Define all operators before using them

Fixed specification:"""


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def clean_response(response: str, module_name: str) -> str:
    """Clean LLM response to extract pure TLA+ code"""
    # Remove thinking tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Extract from code blocks
    if "```" in response:
        blocks = re.findall(r'```(?:tla\+?|tlaplus)?\n?(.*?)\n?```', response, re.DOTALL)
        if blocks:
            response = blocks[0]
    
    # Find module start
    if "---- MODULE" in response:
        response = response[response.find("---- MODULE"):]
    
    # Fix module name
    match = re.search(r'---- MODULE (\w+)', response)
    if match and match.group(1) != module_name:
        response = re.sub(r'---- MODULE \w+ ----', f'---- MODULE {module_name} ----', response, count=1)
    
    # Fix endings - must be exactly ====
    response = re.sub(r'={4,}.*$', '====', response, flags=re.MULTILINE)
    response = re.sub(r'\nEND\s*$', '', response)
    
    # Ensure ending
    if not response.rstrip().endswith("===="):
        response = response.rstrip() + "\n===="
    
    # Remove PlusCal-style Define blocks (invalid in pure TLA+)
    response = re.sub(r'\nDefine\s*\n', '\n', response)
    
    # Fix trailing periods after definitions
    response = re.sub(r'\s+\.\s*$', '', response, flags=re.MULTILINE)
    response = re.sub(r'\s+\.\s*\n', '\n', response)
    
    # Fix let/in to LET/IN
    response = re.sub(r'\blet\b', 'LET', response)
    response = re.sub(r'\bin\b', 'IN', response)
    
    # Common fixes
    response = re.sub(r'UNCHANGED\s*<<\s*>>', 'TRUE', response)
    response = re.sub(r'\]_\(([^)]+)\)', r']_<<\1>>', response)  # ]_(x,y) -> ]_<<x,y>>
    response = re.sub(r'\]_([a-zA-Z_]\w*)\s*$', r']_<<\1>>', response, flags=re.MULTILINE)
    response = re.sub(r'\\neq', '#', response)  # \neq -> # (TLA+ syntax)
    
    return response.strip()


def run_sany(spec: str, module_name: str) -> tuple[bool, str]:
    """Run SANY parser"""
    jar = f"{os.getenv('TLA_TOOLS_DIR', '/opt/tla')}/tla2tools.jar"
    
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{module_name}.tla"
        path.write_text(spec)
        try:
            result = subprocess.run(
                ["java", "-cp", jar, "tla2sany.SANY", str(path)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout + "\n" + result.stderr
            return "Parsing completed" in output, output
        except Exception as e:
            return False, str(e)


def extract_questions(response: str) -> list[str]:
    """Extract numbered questions from LLM response"""
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if re.match(r'^\d+[\.\)]\s*', line):
            questions.append(re.sub(r'^\d+[\.\)]\s*', '', line))
    return questions


def cove_generate(model_name: str, comments: str, llm, examples: list, verbose: bool = True) -> tuple[str, dict]:
    """
    Chain of Verification generation:
    1. Generate initial spec
    2. Generate verification questions
    3. Answer questions
    4. Refine based on answers
    """
    from langchain_core.messages import HumanMessage
    
    trace = {"steps": []}
    
    # Step 1: Initial Generation
    if verbose:
        print(f"\n{CYAN}[CoVe Step 1/4] Generating initial specification...{RESET}")
    
    gen_prompt = build_generation_prompt(comments, model_name, examples)
    response = llm.invoke([HumanMessage(content=gen_prompt)])
    initial_spec = clean_response(response.content, model_name)
    trace["steps"].append({"step": "initial_generation", "spec": initial_spec})
    
    if verbose:
        print(f"  Generated {len(initial_spec)} chars")
    
    # Step 2: Generate Verification Questions
    if verbose:
        print(f"{CYAN}[CoVe Step 2/4] Generating verification questions...{RESET}")
    
    q_prompt = build_verification_questions_prompt(initial_spec, model_name)
    response = llm.invoke([HumanMessage(content=q_prompt)])
    questions = extract_questions(response.content)
    trace["steps"].append({"step": "questions", "questions": questions})
    
    if verbose:
        print(f"  Generated {len(questions)} questions")
        for q in questions[:3]:
            print(f"    - {q[:60]}...")
    
    # Step 3: Answer Verification Questions
    if verbose:
        print(f"{CYAN}[CoVe Step 3/4] Verifying specification...{RESET}")
    
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    v_prompt = build_verification_answers_prompt(initial_spec, questions_text)
    response = llm.invoke([HumanMessage(content=v_prompt)])
    verification = response.content
    trace["steps"].append({"step": "verification", "results": verification})
    
    # Check if any issues found
    has_issues = "NO:" in verification.upper() or "NO -" in verification.upper()
    
    if verbose:
        issues_found = verification.upper().count("NO")
        print(f"  Found {issues_found} potential issues")
    
    # Step 4: Refine if needed
    if has_issues:
        if verbose:
            print(f"{CYAN}[CoVe Step 4/4] Refining based on verification...{RESET}")
        
        r_prompt = build_refinement_prompt(initial_spec, verification, model_name)
        response = llm.invoke([HumanMessage(content=r_prompt)])
        refined_spec = clean_response(response.content, model_name)
        trace["steps"].append({"step": "refinement", "spec": refined_spec})
        
        if verbose:
            print(f"  Refined spec: {len(refined_spec)} chars")
        
        return refined_spec, trace
    else:
        if verbose:
            print(f"{CYAN}[CoVe Step 4/4] No issues found, keeping original{RESET}")
        return initial_spec, trace


def generate_with_cove(model_name: str, comments: str, max_retries: int = 2, save_output: bool = True):
    """Full pipeline: CoVe generation + SANY validation + repair loop"""
    from langchain_core.messages import HumanMessage
    
    llm = get_llm()
    backend = os.getenv("LLM_BACKEND", "ollama")
    model = os.getenv("LLM_MODEL", "local")
    examples = load_few_shot_examples(2)
    
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TLA+ Generation with Chain of Verification (CoVe){RESET}")
    print(f"{BLUE}Model: {model_name} | LLM: {backend}:{model}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # CoVe Generation
    spec, cove_trace = cove_generate(model_name, comments, llm, examples)
    
    # SANY Validation
    print(f"\n{YELLOW}[SANY] Validating specification...{RESET}")
    success, sany_output = run_sany(spec, model_name)
    
    attempt = 1
    history = [{
        "attempt": attempt,
        "method": "cove",
        "spec": spec,
        "sany_pass": success,
        "cove_trace": cove_trace
    }]
    
    if success:
        print(f"{GREEN}✓ SANY passed after CoVe!{RESET}")
    else:
        print(f"{RED}✗ SANY failed{RESET}")
        for line in sany_output.split('\n'):
            if any(x in line.lower() for x in ['error', 'unknown', 'expecting']):
                print(f"  {line[:80]}")
    
    # Repair loop if needed
    while not success and attempt <= max_retries:
        attempt += 1
        print(f"\n{YELLOW}[Repair {attempt}/{max_retries+1}] Fixing SANY errors...{RESET}")
        
        repair_prompt = build_sany_repair_prompt(spec, sany_output, model_name)
        response = llm.invoke([HumanMessage(content=repair_prompt)])
        spec = clean_response(response.content, model_name)
        
        success, sany_output = run_sany(spec, model_name)
        history.append({
            "attempt": attempt,
            "method": "sany_repair",
            "spec": spec,
            "sany_pass": success
        })
        
        if success:
            print(f"{GREEN}✓ SANY passed after repair!{RESET}")
        else:
            print(f"{RED}✗ SANY still failing{RESET}")
            for line in sany_output.split('\n'):
                if any(x in line.lower() for x in ['error', 'unknown', 'expecting']):
                    print(f"  {line[:80]}")
    
    # Results
    print(f"\n{BLUE}{'='*60}{RESET}")
    if success:
        print(f"{GREEN}SUCCESS after {attempt} attempt(s){RESET}")
    else:
        print(f"{RED}FAILED after {attempt} attempt(s){RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Show spec
    print(f"\n{BLUE}--- Final Specification ---{RESET}")
    print(spec[:1800])
    if len(spec) > 1800:
        print(f"... ({len(spec)} total chars)")
    
    # Save
    if save_output:
        output_dir = project_root / "outputs" / f"{backend}_{model}" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        (output_dir / f"{model_name}.tla").write_text(spec)
        print(f"\n{BLUE}Saved: {output_dir / f'{model_name}.tla'}{RESET}")
        
        with open(output_dir / f"{model_name}_cove.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    return {"model": model_name, "spec": spec, "sany_pass": success, "attempts": attempt, "history": history}


def find_model_comments(model_name: str) -> str | None:
    """Find comments for a model"""
    data_dir = project_root / "data"
    for json_path in [project_root / "outputs" / "val.json", project_root / "outputs" / "train.json"]:
        with open(json_path) as f:
            for entry in json.load(f)["data"]:
                if entry["model"] == model_name:
                    cf = entry.get("comments_clean")
                    if cf:
                        path = data_dir / model_name / "txt" / cf
                        if path.exists():
                            return path.read_text().strip()
                        for p in data_dir.rglob(cf):
                            return p.read_text().strip()
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TLA+ Generator with Chain of Verification")
    parser.add_argument("--model", type=str, default="DieHard", help="Model name")
    parser.add_argument("--retries", type=int, default=2, help="Max SANY repair attempts")
    parser.add_argument("--list", action="store_true", help="List models")
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    
    args = parser.parse_args()
    load_env()
    
    if args.list:
        with open(project_root / "outputs" / "val.json") as f:
            data = json.load(f)["data"]
        print("Available models:")
        for e in data[:25]:
            print(f"  - {e['model']}")
        if len(data) > 25:
            print(f"  ... and {len(data)-25} more")
    else:
        comments = find_model_comments(args.model)
        if comments:
            result = generate_with_cove(args.model, comments, args.retries, not args.no_save)
            exit(0 if result["sany_pass"] else 1)
        else:
            print(f"Comments not found for: {args.model}")
            exit(1)
