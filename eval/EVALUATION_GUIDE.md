# RAG Evaluation Guide

## Overview
This guide explains how to evaluate your RAG chatbot using the provided evaluation tools. The evaluation measures:

- **Accuracy**: How correctly answers match ground truth
- **Recall**: How much important information is captured
- **Faithfulness**: How faithful answers are to source documents (no hallucinations)
- **Completeness**: How thorough and complete answers are
- **Latency**: Response time for each query
- **Breakdown by difficulty level and question type**

## Prerequisites

1. **Backend running**: `cd backend && uvicorn main:app --reload --port 8000`
2. **Environment configured**: `.env` file with required settings
3. **Test document**: PDF file in `eval/` directory
4. **Evaluation dataset**: JSON with questions and ground truth

## Files Needed

### 1. Test Document
- **Location**: `eval/ai_evaluation_test_document.pdf` or `eval/gemini_evaluation_prompt.pdf`
- Contains: The document your RAG system will search through

### 2. Evaluation Dataset
- **Location**: `eval/evaluation_dataset.json`
- **Structure**:
```json
[
  {
    "id": "Q01",
    "question": "What was the global investment in healthcare AI in 2023?",
    "ground_truth": "Global investment in healthcare AI reached 45 billion USD...",
    "context": "According to a 2023 report...",
    "page_number": 1,
    "difficulty": "easy",  // easy, medium, hard
    "question_type": "factual",  // factual, reasoning, synthesis
    "section": "1. Introduction"
  }
  // ... more questions
]
```

## Running Evaluations

### Option 1: Basic Evaluation (Original)
```bash
cd eval
python run_eval.py --file ai_evaluation_test_document.pdf --dataset evaluation_dataset.json
```

Measures:
- Single overall score (0-10)
- Latency

### Option 2: Comprehensive Evaluation ⭐ RECOMMENDED
```bash
cd eval
python comprehensive_eval.py --file ai_evaluation_test_document.pdf --dataset evaluation_dataset.json
```

Measures:
- **Accuracy** (0-10): Correctness vs ground truth
- **Recall** (0-10): Completeness of information
- **Faithfulness** (0-10): No hallucinations/false info
- **Completeness** (0-10): Thoroughness of answer
- **Latency**: Response time in seconds

**Output includes**:
- Overall metrics across all questions
- Metrics breakdown by difficulty (easy/medium/hard)
- Metrics breakdown by question type (factual/reasoning/synthesis)
- Detailed per-question results

### Option 3: Custom Evaluation
Edit `comprehensive_eval.py` to customize:
- Evaluation metrics
- LLM judge model
- Score thresholds
- Report format

## Understanding Results

### Overall Metrics Example
```
📈 OVERALL METRICS (out of 10):
   Accuracy:      8.45
   Recall:        7.92
   Faithfulness:  8.76
   Completeness:  7.89

⏱️  LATENCY METRICS:
   Average:       2.34s
   Min:           1.12s
   Max:           5.47s
```

### Breakdown by Difficulty
```
🎯 BREAKDOWN BY DIFFICULTY:
   EASY:
      Count: 15 | Accuracy: 8.8 | Recall: 8.5 | Faithfulness: 9.0 | Latency: 1.8s
   MEDIUM:
      Count: 10 | Accuracy: 8.2 | Recall: 7.8 | Faithfulness: 8.5 | Latency: 2.5s
   HARD:
      Count: 5 | Accuracy: 7.0 | Recall: 6.8 | Faithfulness: 8.0 | Latency: 3.2s
```

## Interpreting Scores

### Accuracy (0-10)
- 8-10: Excellent, answers match ground truth very closely
- 6-8: Good, mostly correct with minor differences
- 4-6: Fair, partially correct but missing details
- 0-4: Poor, significant inaccuracies

### Recall (0-10)
- 8-10: All important information is included
- 6-8: Most important information captured
- 4-6: Some key information missing
- 0-4: Limited information provided

### Faithfulness (0-10)
- 8-10: Completely faithful to source, no hallucinations
- 6-8: Mostly faithful, minimal extrapolation
- 4-6: Some unsupported claims
- 0-4: Many hallucinations or false claims

### Latency
- 0-1s: Excellent (immediate response)
- 1-3s: Good (acceptable)
- 3-5s: Fair (noticeable delay)
- >5s: Poor (should investigate bottleneck)

## Workflow

1. **Prepare test document**
   ```bash
   # Place your PDF in eval/
   cp your_document.pdf eval/
   ```

2. **Create evaluation dataset**
   ```bash
   # questions.json should have the format shown above
   # Tools: LLM-generated QA pairs from your document
   ```

3. **Run comprehensive evaluation**
   ```bash
   cd eval
   python comprehensive_eval.py --file your_document.pdf --dataset questions.json
   ```

4. **Review results**
   ```bash
   # Check eval/results/ for timestamped JSON reports
   ls -la eval/results/
   ```

5. **Analyze by category**
   - If easy questions score lower → improve retrieval
   - If latency is high → optimize embedding/LLM calls
   - If faithfulness is low → improve prompt engineering
   - If recall is low → improve chunking strategy

## Environment Variables

Set these in `.env`:
```
OLLAMA_BASE_URL=http://192.168.6.233:11434
JUDGE_MODEL=gemma4:31b
API_URL=http://localhost:8000
```

## Example: Evaluating AI Healthcare Document

```bash
cd eval

# Run evaluation on healthcare document
python comprehensive_eval.py \
  --file ai_evaluation_test_document.pdf \
  --dataset evaluation_dataset.json

# View results
cat results/comprehensive_eval_20240618_093200.json | python -m json.tool
```

## Tips for Better Evaluations

1. **Diverse questions**: Include easy, medium, and hard questions
2. **Different types**: Mix factual, reasoning, and synthesis questions
3. **Multiple iterations**: Re-evaluate after system changes
4. **Track over time**: Compare results across evaluation runs
5. **Ground truth quality**: Ensure ground truth is accurate and complete

## Troubleshooting

### "Cannot connect to API"
```bash
# Make sure backend is running
cd backend
uvicorn main:app --reload --port 8000
```

### "Judge model not found"
```bash
# Pull the model
ollama pull gemma4:31b

# Or use different model
export JUDGE_MODEL=llama2
python comprehensive_eval.py ...
```

### Slow evaluations
- Reduce number of questions
- Use faster judge model (gemma2 instead of gemma4)
- Check latency breakdown - identify slow queries

## Next Steps

1. Run comprehensive evaluation on your docs
2. Review metrics by difficulty and type
3. Identify weak areas:
   - Low accuracy? → Improve retrieval
   - Low recall? → Fix chunking
   - Low faithfulness? → Better prompting
   - High latency? → Optimize pipeline
4. Make improvements
5. Re-evaluate to measure impact
