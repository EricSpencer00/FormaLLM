# Few-Shot vs CoVe Prompting Comparison

## Experimental Setup
- **Model**: qwen2.5-coder:32b (Ollama local)
- **Backend**: Ollama (no API costs)
- **Test Models**: DieHard, Majority, TwoPhase
- **Validation**: SANY parser (TLA+ syntax/semantics)

## Approach Descriptions

### Few-Shot Learning
- **Method**: Show 2-3 complete examples (comments → TLA+ spec)
- **Examples Used**: Stones, KeyValueStore, Huang
- **Prompt Structure**: System rules + Examples + New task
- **Steps**: 1 (single generation)
- **Time**: ~20s average per spec

### Chain of Verification (CoVe)
- **Method**: Generate → Self-critique → Revise
- **Steps**: 
  1. Generate initial spec (no examples)
  2. LLM reviews own output with verification questions
  3. LLM revises based on self-critique
- **Time**: ~74s average per spec (3.6x slower)

## Results Summary

| Model | Few-Shot | CoVe | Winner |
|-------|----------|------|--------|
| **DieHard** | ❌ Parse Error | ✅ PASS | **CoVe** |
| **Majority** | ❌ Parse Error | ❌ Semantic Errors | Neither |
| **TwoPhase** | ❌ Semantic Errors | ❌ Semantic Errors | Neither |

### Success Rates
- **Few-Shot**: 0/3 (0%)
- **CoVe**: 1/3 (33%)

### Time Comparison
- **Few-Shot**: 20.3s average
- **CoVe**: 73.6s average (3.6x slower but higher quality)

## Detailed Analysis

### DieHard (CoVe Win)

**Few-Shot Error** (line 53):
```tla
Spec ==
/\ Init
/\ [][Next]_<<big, small>>
/\ WF_[Next]_<<big, small>>    <-- INVALID SYNTAX
```
Error: `Ill-structured fairness expression`

**CoVe Success**:
```tla
Spec ==
    /\ TypeOK
    /\ Init
    /\ [][Next]_<<big, small>>   <-- No fairness (valid)
```
CoVe correctly avoided the malformed fairness condition. The verification step likely caught this.

### Majority (Both Failed)

**Few-Shot Error**:
- Precedence conflict with `=>` operator at line 60
- Invalid theorem syntax

**CoVe Error**: 
- Passed parsing but semantic errors (3 errors)
- Unknown operators at lines 12, 35, 55
- CoVe got closer but still failed

### TwoPhase (Both Failed)

**Few-Shot Error**:
- Semantic errors in module processing
- Likely undefined operator references

**CoVe Error**:
- Semantic errors (2 errors)
- Unknown operators: `TYPEOK'` and `Next`
- Incomplete spec (only 377 bytes vs 947 for few-shot)

## Key Findings

### CoVe Advantages
1. **Self-Correction**: Verification step catches syntax errors before submission
2. **Higher Quality**: When successful, produces cleaner specs
3. **Better for Complex Specs**: Self-critique helps with intricate logic

### Few-Shot Advantages
1. **Speed**: 3.6x faster (20s vs 74s)
2. **Learning from Examples**: Sees working patterns directly
3. **Consistency**: More predictable structure

### Common Failure Modes
Both approaches struggled with:
- **Fairness Syntax**: `WF_<<vars>>(Action)` vs `WF_[Action]_<<vars>>`
- **Operator Precedence**: `=>` and `\land` conflicts
- **Undefined Operators**: Using before defining (Min, Max, etc.)
- **Incomplete Specs**: Missing definitions or malformed structures

## Recommendations

### For Production Use
1. **Hybrid Approach**: Use few-shot for speed, fall back to CoVe for failures
2. **Retry Strategy**: 
   - First attempt: Few-shot (fast)
   - If fails: CoVe (thorough self-check)
   - If still fails: Few-shot + SANY feedback repair
3. **Cost Consideration**: CoVe uses 3x more LLM tokens (3 prompts vs 1)

### For Local Models (Ollama)
- CoVe more effective on code-specialized models (qwen2.5-coder)
- Few-shot relies on quality examples (garbage in = garbage out)
- Consider increasing temperature slightly for CoVe to encourage self-critique

### For Cloud APIs (GPT-4/Claude)
- CoVe may be more cost-effective despite 3x tokens (higher first-attempt success)
- GPT-4 and Claude have better self-critique capabilities
- Few-shot still competitive for simple specs

## Next Steps

1. **Test More Models**: Run on full validation set (30 models)
2. **Hybrid Pipeline**: Implement few-shot → CoVe fallback
3. **CoVe Variations**: Try different verification question sets
4. **Few-Shot Tuning**: Experiment with number/selection of examples
5. **Cloud Model Comparison**: Test CoVe on GPT-4/Claude (likely much better)

## Conclusion

**CoVe wins on quality (33% vs 0% success)** but at **3.6x time cost**. 

For local models like qwen2.5-coder:32b, CoVe's self-verification catches syntax errors that few-shot misses. However, both struggle with semantic correctness, suggesting the underlying model needs stronger TLA+ understanding or more sophisticated prompting.

**Best Strategy**: Use few-shot for speed, CoVe for critical specs, and combine with SANY feedback loops for maximum success rate.
