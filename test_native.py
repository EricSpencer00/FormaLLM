#!/usr/bin/env python3
"""
Minimal LLM connectivity test for FormaLLM
Tests Ollama backend without deprecated imports
"""
import os
import sys
from pathlib import Path

# Load environment
project_root = Path(__file__).resolve().parent
env_path = project_root / ".env"

def load_env():
    """Load environment variables from .env file"""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")

def test_ollama():
    """Test Ollama backend connectivity"""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("LLM_MODEL", "deepseek-r1:7b")
    
    print(f"Testing Ollama at {base_url} with model: {model}")
    print("-" * 60)
    
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0
    )
    
    test_prompt = "Say 'Hello, TLA+!' in exactly 3 words."
    print(f"Prompt: {test_prompt}")
    
    response = llm.invoke([HumanMessage(content=test_prompt)])
    print(f"Response: {response.content[:200]}...")
    print("\n✓ Ollama connection successful!")
    return True

def test_sany():
    """Test TLA+ SANY parser"""
    import subprocess
    import tempfile
    
    tla_tools_dir = os.getenv("TLA_TOOLS_DIR", "/opt/tla")
    jar_path = f"{tla_tools_dir}/tla2tools.jar"
    
    print(f"\nTesting SANY parser at {jar_path}")
    print("-" * 60)
    
    # Create minimal valid TLA+ spec
    test_spec = """---- MODULE Test ----
VARIABLES x
Init == x = 0
Next == x' = x + 1
====
"""
    
    with tempfile.NamedTemporaryFile(suffix=".tla", mode="w", delete=False) as f:
        f.write(test_spec)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["java", "-cp", jar_path, "tla2sany.SANY", temp_path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        print(f"SANY output: {output[:300]}")
        
        if "Parsing completed" in output:
            print("\n✓ SANY parser working!")
            return True
        else:
            print("\n✗ SANY parsing failed")
            return False
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    load_env()
    
    print("=" * 60)
    print("FormaLLM Native Environment Test")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  LLM_BACKEND: {os.getenv('LLM_BACKEND', 'NOT SET')}")
    print(f"  LLM_MODEL: {os.getenv('LLM_MODEL', 'NOT SET')}")
    print(f"  OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'NOT SET')}")
    print(f"  TLA_TOOLS_DIR: {os.getenv('TLA_TOOLS_DIR', 'NOT SET')}")
    print()
    
    ollama_ok = False
    sany_ok = False
    
    try:
        ollama_ok = test_ollama()
    except Exception as e:
        print(f"\n✗ Ollama test failed: {e}")
    
    try:
        sany_ok = test_sany()
    except Exception as e:
        print(f"\n✗ SANY test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Ollama: {'✓ OK' if ollama_ok else '✗ FAILED'}")
    print(f"  SANY:   {'✓ OK' if sany_ok else '✗ FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if (ollama_ok and sany_ok) else 1)
