import json, requests, sys, uuid

BACKEND_URL = "http://localhost:8000"
session_id = str(uuid.uuid4())
model_name = sys.argv[1] if len(sys.argv) > 1 else "default"

with open("eval/eval_questions.json") as f:
    questions = json.load(f)

results = []
for q in questions:
    resp = requests.post(f"{BACKEND_URL}/chat", json={"session_id": session_id, "query": q["question"]}).json()
    hit = any(kw.lower() in resp["answer"].lower() for kw in q["expected_keywords"])
    results.append({
        "question": q["question"],
        "answer": resp["answer"],
        "top_similarity": resp["sources"][0]["similarity"] if resp["sources"] else None,
        "keyword_hit": hit,
    })

with open(f"eval/results/{model_name}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. {sum(r['keyword_hit'] for r in results)}/{len(results)} keyword hits.")