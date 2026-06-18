import requests
import json
import os
import sys
import time

import argparse
import uuid

# Configuration
API_URL = "http://localhost:8000"
OLLAMA_URL = "http://192.168.6.233:11434/api/generate"
JUDGE_MODEL = "gemma4:31b"
DEFAULT_TEST_FILE = "eval/gemini_evaluation_prompt.pdf"
DEFAULT_DATASET = "eval/evaluation_dataset.json"

def evaluate_with_llm(question, ground_truth, answer):
    """Use the LLM to score the answer based on the ground truth."""
    prompt = f"""
    You are an impartial judge evaluating the quality of an AI-generated answer based on a ground truth reference.
    
    [Question]
    {question}
    
    [Ground Truth]
    {ground_truth}
    
    [Generated Answer]
    {answer}
    
    Rate the Generated Answer compared to the Ground Truth on a scale of 0 to 10:
    - 10: Perfectly captures all facts and details from the ground truth.
    - 7-9: Captures the main facts correctly but may miss minor details or use different wording.
    - 4-6: Partially correct but misses significant facts or includes some inaccuracies.
    - 0-3: Completely incorrect, irrelevant, or fails to answer the question based on the ground truth.
    
    Provide only the numeric score (0-10) as your response.
    """
    try:
        # Calling Ollama directly for the judge to use the 31B model
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            score_text = response.json().get('response', '0').strip()
            # More robust regex to find a score like "8/10" or "Score: 9"
            import re
            # Look for "Score: X" or just a number at the start or end
            match = re.search(r'([0-9]|10)(?:\s*/\s*10)?', score_text)
            if match:
                score = int(match.group(1))
                return score / 10.0
            else:
                print(f"  ⚠️ Could not parse judge score from: {score_text[:50]}...")
    except:
        pass
    return 0.0

def run_evaluation(test_file, dataset_file):
    session_id = str(uuid.uuid4())
    print("🚀 Starting AI Evaluation Suite (LLM-as-a-Judge)")
    print(f"📄 Test Document: {test_file}")
    print(f"📊 Dataset: {dataset_file}")
    print("-" * 60)

    # 1. Clear session
    print(f"🧹 Clearing sessions...")
    try:
        requests.delete(f"{API_URL}/session/{session_id}")
        requests.delete(f"{API_URL}/session/judge-session")
    except Exception as e:
        print(f"❌ Failed to connect to API at {API_URL}. Is the backend running?")
        sys.exit(1)

    # 2. Upload test document
    print(f"📄 Uploading and Indexing...")
    with open(test_file, 'rb') as f:
        files = {'files': (os.path.basename(test_file), f, 'application/pdf')}
        response = requests.post(f"{API_URL}/upload?session_id={session_id}", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        sys.exit(1)
    
    print(f"✅ Document indexed: {response.json().get('chunks_indexed', 0)} chunks.")

    # 3. Load questions
    with open(dataset_file, 'r') as f:
        questions = json.load(f)

    # 4. Run questions
    results = []
    total_score = 0
    
    for idx, q in enumerate(questions):
        question_id = q.get('id', f"Q{idx+1}")
        question_text = q['question']
        ground_truth = q['ground_truth']
        
        print(f"\n[{question_id}] {question_text}")
        
        start_time = time.time()
        try:
            chat_response = requests.post(
                f"{API_URL}/chat",
                json={"session_id": session_id, "query": question_text},
                timeout=120
            )
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout (120s) - Skipping question")
            continue
        
        latency = time.time() - start_time
        
        if chat_response.status_code != 200:
            print(f"  ❌ API Error: {chat_response.text}")
            continue
            
        data = chat_response.json()
        answer = data.get('answer', '')
        
        # LLM based evaluation
        score = evaluate_with_llm(question_text, ground_truth, answer)
        total_score += score
        
        status = "✅ PASS" if score >= 0.8 else "⚠️ PARTIAL" if score >= 0.4 else "❌ FAIL"
        
        print(f"  GT: {ground_truth[:80]}..." if len(ground_truth) > 80 else f"  GT: {ground_truth}")
        print(f"  A:  {answer[:80]}..." if len(answer) > 80 else f"  A:  {answer}")
        print(f"  Score: {score*10:.1f}/10 | Status: {status} | Latency: {latency:.2f}s")
        
        results.append({
            "id": question_id,
            "question": question_text,
            "answer": answer,
            "ground_truth": ground_truth,
            "score": score,
            "latency": latency
        })

    # 5. Summary
    avg_score = (total_score / len(questions)) * 100 if questions else 0
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("-" * 60)
    print(f"Total Questions: {len(questions)}")
    print(f"Average Match:   {avg_score:.1f}%")
    print(f"Total Latency:   {sum(r['latency'] for r in results):.2f}s")
    print("=" * 60)

    # Save results
    os.makedirs("eval/results", exist_ok=True)
    result_path = "eval/results/latest_run.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument("--file", default=DEFAULT_TEST_FILE, help="Path to test PDF")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to evaluation JSON")
    args = parser.parse_args()

    run_evaluation(args.file, args.dataset)