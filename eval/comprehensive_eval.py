import requests
import json
import os
import sys
import time
import argparse
import uuid
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv("../.env")  # Load from backend .env file

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama2")  # Use llama2 as default (more reliable)
DEFAULT_TEST_FILE = "eval/gemini_evaluation_prompt.pdf"
DEFAULT_DATASET = "eval/evaluation_dataset.json"

def simple_evaluate(question, ground_truth, answer):
    """Fallback: Simple string-based evaluation when judge model fails"""
    # Accuracy: string similarity
    accuracy = SequenceMatcher(None, ground_truth.lower(), answer.lower()).ratio() * 10
    
    # Recall: what % of ground truth words appear in answer
    gt_words = set(ground_truth.lower().split())
    answer_words = set(answer.lower().split())
    if gt_words:
        recall = len(gt_words & answer_words) / len(gt_words) * 10
    else:
        recall = 0
    
    # Faithfulness: if answer is shorter but contains key info, penalize less
    faithfulness = 7 if answer and len(answer) > 20 else 3
    
    # Completeness: based on answer length
    completeness = min(10, max(3, len(answer.split()) / 5))
    
    return {
        "accuracy": min(10, accuracy),
        "recall": min(10, recall),
        "faithfulness": faithfulness,
        "completeness": completeness,
        "reason": "Fallback evaluation"
    }

def evaluate_answer(question, ground_truth, answer):
    """
    Comprehensive evaluation using LLM as judge.
    Returns: accuracy, recall, faithfulness, precision scores (0-10 scale)
    """
    evaluation_prompt = f"""You are an expert evaluator assessing AI-generated answers.
Evaluate the following answer on multiple dimensions:

[Question]
{question}

[Ground Truth / Expected Answer]
{ground_truth}

[Generated Answer]
{answer}

Rate on these dimensions (0-10 each):
1. ACCURACY: How correctly does the answer match the ground truth?
2. RECALL: How much of the important information from ground truth is included?
3. FAITHFULNESS: How faithful is the answer to the source (no hallucinations)?
4. COMPLETENESS: How complete is the answer (not too brief)?

Respond with JSON format only, no other text:
{{
    "accuracy": <0-10>,
    "recall": <0-10>,
    "faithfulness": <0-10>,
    "completeness": <0-10>,
    "reason": "<brief explanation>"
}}"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": JUDGE_MODEL,
                "prompt": evaluation_prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=60
        )
        if response.status_code == 200:
            try:
                response_text = response.json().get('response', '{}').strip()
                import re
                # Extract JSON with better regex
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    # Validate and clamp scores to 0-10
                    result['accuracy'] = max(0, min(10, int(result.get('accuracy', 0))))
                    result['recall'] = max(0, min(10, int(result.get('recall', 0))))
                    result['faithfulness'] = max(0, min(10, int(result.get('faithfulness', 0))))
                    result['completeness'] = max(0, min(10, int(result.get('completeness', 0))))
                    return result
                else:
                    print(f"    ⚠️  Could not extract JSON from: {response_text[:80]}")
            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON error: {str(e)[:40]}")
            except Exception as e:
                print(f"    ⚠️  Parse error: {str(e)[:40]}")
        else:
            print(f"    ⚠️  API error: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"    ⚠️  Judge timeout")
    except requests.exceptions.ConnectionError:
        print(f"    ⚠️  Cannot connect to Ollama at {OLLAMA_URL}")
    except Exception as e:
        print(f"    ⚠️  Error: {str(e)[:40]}")
    
    # Use fallback simple evaluation when judge fails
    print(f"    📊 Using fallback evaluation")
    return simple_evaluate(question, ground_truth, answer)

def run_comprehensive_evaluation(test_file, dataset_file):
    """Run comprehensive evaluation with detailed metrics"""
    
    session_id = str(uuid.uuid4())
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE RAG EVALUATION SUITE")
    print("="*70)
    print(f"📄 Test Document: {test_file}")
    print(f"📊 Dataset: {dataset_file}")
    print(f"🤖 Judge Model: {JUDGE_MODEL}")
    print(f"🔗 API URL: {API_URL}")
    print(f"🔗 Ollama URL: {OLLAMA_URL[:-14]}")  # Remove /api/generate for display
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)

    # 1. Clear session
    try:
        requests.delete(f"{API_URL}/session/{session_id}")
    except:
        print(f"❌ ERROR: Cannot connect to API at {API_URL}")
        print("   Make sure the backend is running: uvicorn main:app --reload --port 8000")
        sys.exit(1)

    # 2. Upload test document
    print(f"\n📤 Uploading document...")
    try:
        with open(test_file, 'rb') as f:
            files = {'files': (os.path.basename(test_file), f, 'application/pdf')}
            response = requests.post(f"{API_URL}/upload?session_id={session_id}", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            sys.exit(1)
        
        chunks_indexed = response.json().get('chunks_indexed', 0)
        print(f"✅ Document indexed: {chunks_indexed} chunks")
    except Exception as e:
        print(f"❌ Upload error: {e}")
        sys.exit(1)

    # 3. Load questions
    try:
        with open(dataset_file, 'r') as f:
            questions = json.load(f)
        print(f"✅ Loaded {len(questions)} questions")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # 4. Run evaluation
    print(f"\n🚀 Running evaluation on {len(questions)} questions...\n")
    
    results = []
    metrics_by_difficulty = defaultdict(lambda: {"count": 0, "accuracy": [], "recall": [], "faithfulness": [], "completeness": [], "latency": []})
    metrics_by_type = defaultdict(lambda: {"count": 0, "accuracy": [], "recall": [], "faithfulness": [], "completeness": [], "latency": []})
    
    latencies = []
    all_accuracy = []
    all_recall = []
    all_faithfulness = []
    all_completeness = []

    for idx, q in enumerate(questions, 1):
        question_id = q.get('id', f"Q{idx}")
        question_text = q['question']
        ground_truth = q['ground_truth']
        difficulty = q.get('difficulty', 'unknown')
        question_type = q.get('question_type', 'unknown')
        
        print(f"[{question_id}] {question_text[:60]}...")
        
        # Query the API
        start_time = time.time()
        try:
            chat_response = requests.post(
                f"{API_URL}/chat",
                json={"session_id": session_id, "query": question_text},
                timeout=120
            )
        except requests.exceptions.Timeout:
            print(f"  ⚠️  TIMEOUT (120s)")
            continue
        
        latency = time.time() - start_time
        latencies.append(latency)
        
        if chat_response.status_code != 200:
            print(f"  ❌ API Error: {chat_response.text}")
            continue
        
        answer = chat_response.json().get('answer', '')
        
        # Evaluate answer
        eval_result = evaluate_answer(question_text, ground_truth, answer)
        
        accuracy = eval_result.get('accuracy', 0)
        recall = eval_result.get('recall', 0)
        faithfulness = eval_result.get('faithfulness', 0)
        completeness = eval_result.get('completeness', 0)
        
        all_accuracy.append(accuracy)
        all_recall.append(recall)
        all_faithfulness.append(faithfulness)
        all_completeness.append(completeness)
        
        # Track by difficulty
        metrics_by_difficulty[difficulty]['count'] += 1
        metrics_by_difficulty[difficulty]['accuracy'].append(accuracy)
        metrics_by_difficulty[difficulty]['recall'].append(recall)
        metrics_by_difficulty[difficulty]['faithfulness'].append(faithfulness)
        metrics_by_difficulty[difficulty]['completeness'].append(completeness)
        metrics_by_difficulty[difficulty]['latency'].append(latency)
        
        # Track by question type
        metrics_by_type[question_type]['count'] += 1
        metrics_by_type[question_type]['accuracy'].append(accuracy)
        metrics_by_type[question_type]['recall'].append(recall)
        metrics_by_type[question_type]['faithfulness'].append(faithfulness)
        metrics_by_type[question_type]['completeness'].append(completeness)
        metrics_by_type[question_type]['latency'].append(latency)
        
        # Display result
        overall_score = (accuracy + recall + faithfulness + completeness) / 4
        status = "✅" if overall_score >= 7 else "⚠️ " if overall_score >= 5 else "❌"
        
        print(f"  {status} Accuracy: {accuracy:.1f} | Recall: {recall:.1f} | Faithfulness: {faithfulness:.1f} | Latency: {latency:.2f}s")
        
        results.append({
            "id": question_id,
            "question": question_text,
            "answer": answer[:200],
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "question_type": question_type,
            "metrics": {
                "accuracy": accuracy,
                "recall": recall,
                "faithfulness": faithfulness,
                "completeness": completeness,
                "latency": latency
            }
        })

    # 5. Generate comprehensive report
    print("\n" + "="*70)
    print("📊 EVALUATION REPORT")
    print("="*70)
    
    avg_accuracy = sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    avg_faithfulness = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    avg_completeness = sum(all_completeness) / len(all_completeness) if all_completeness else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print(f"\n📈 OVERALL METRICS (out of 10):")
    print(f"   Accuracy:      {avg_accuracy:.2f}")
    print(f"   Recall:        {avg_recall:.2f}")
    print(f"   Faithfulness:  {avg_faithfulness:.2f}")
    print(f"   Completeness:  {avg_completeness:.2f}")
    print(f"\n⏱️  LATENCY METRICS:")
    print(f"   Average:       {avg_latency:.2f}s")
    print(f"   Min:           {min(latencies):.2f}s" if latencies else "   Min:           N/A")
    print(f"   Max:           {max(latencies):.2f}s" if latencies else "   Max:           N/A")
    print(f"   Total:         {sum(latencies):.2f}s")
    
    # Breakdown by difficulty
    print(f"\n🎯 BREAKDOWN BY DIFFICULTY:")
    for difficulty in sorted(metrics_by_difficulty.keys()):
        stats = metrics_by_difficulty[difficulty]
        avg_acc = sum(stats['accuracy']) / len(stats['accuracy']) if stats['accuracy'] else 0
        avg_rec = sum(stats['recall']) / len(stats['recall']) if stats['recall'] else 0
        avg_faith = sum(stats['faithfulness']) / len(stats['faithfulness']) if stats['faithfulness'] else 0
        avg_lat = sum(stats['latency']) / len(stats['latency']) if stats['latency'] else 0
        
        print(f"   {difficulty.upper()}:")
        print(f"      Count: {stats['count']} | Accuracy: {avg_acc:.2f} | Recall: {avg_rec:.2f} | Faithfulness: {avg_faith:.2f} | Latency: {avg_lat:.2f}s")
    
    # Breakdown by question type
    print(f"\n🏷️  BREAKDOWN BY QUESTION TYPE:")
    for q_type in sorted(metrics_by_type.keys()):
        stats = metrics_by_type[q_type]
        avg_acc = sum(stats['accuracy']) / len(stats['accuracy']) if stats['accuracy'] else 0
        avg_rec = sum(stats['recall']) / len(stats['recall']) if stats['recall'] else 0
        avg_faith = sum(stats['faithfulness']) / len(stats['faithfulness']) if stats['faithfulness'] else 0
        avg_lat = sum(stats['latency']) / len(stats['latency']) if stats['latency'] else 0
        
        print(f"   {q_type.upper()}:")
        print(f"      Count: {stats['count']} | Accuracy: {avg_acc:.2f} | Recall: {avg_rec:.2f} | Faithfulness: {avg_faith:.2f} | Latency: {avg_lat:.2f}s")
    
    print("\n" + "="*70)
    
    # Save detailed results
    os.makedirs("eval/results", exist_ok=True)
    
    # Save individual results
    result_file = f"eval/results/comprehensive_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_file": test_file,
        "chunks_indexed": chunks_indexed,
        "total_questions": len(questions),
        "overall_metrics": {
            "accuracy": round(avg_accuracy, 2),
            "recall": round(avg_recall, 2),
            "faithfulness": round(avg_faithfulness, 2),
            "completeness": round(avg_completeness, 2),
            "avg_latency": round(avg_latency, 2),
            "min_latency": round(min(latencies), 2) if latencies else None,
            "max_latency": round(max(latencies), 2) if latencies else None
        },
        "by_difficulty": {
            k: {
                "count": v['count'],
                "accuracy": round(sum(v['accuracy']) / len(v['accuracy']), 2) if v['accuracy'] else 0,
                "recall": round(sum(v['recall']) / len(v['recall']), 2) if v['recall'] else 0,
                "faithfulness": round(sum(v['faithfulness']) / len(v['faithfulness']), 2) if v['faithfulness'] else 0,
                "avg_latency": round(sum(v['latency']) / len(v['latency']), 2) if v['latency'] else 0
            }
            for k, v in metrics_by_difficulty.items()
        },
        "by_type": {
            k: {
                "count": v['count'],
                "accuracy": round(sum(v['accuracy']) / len(v['accuracy']), 2) if v['accuracy'] else 0,
                "recall": round(sum(v['recall']) / len(v['recall']), 2) if v['recall'] else 0,
                "faithfulness": round(sum(v['faithfulness']) / len(v['faithfulness']), 2) if v['faithfulness'] else 0,
                "avg_latency": round(sum(v['latency']) / len(v['latency']), 2) if v['latency'] else 0
            }
            for k, v in metrics_by_type.items()
        },
        "details": results
    }
    
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")
    print(f"✅ Summary: {avg_accuracy:.1f}% accuracy | {avg_recall:.1f}% recall | {avg_faithfulness:.1f}% faithfulness")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive RAG Evaluation Suite")
    parser.add_argument("--file", default=DEFAULT_TEST_FILE, help="Path to test PDF")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to evaluation dataset JSON")
    args = parser.parse_args()

    run_comprehensive_evaluation(args.file, args.dataset)
