from langsmith import Client
import json
import os

os.environ["LANGCHAIN_API_KEY"] = "YOUR API KEY"

def create_langsmith_dataset(json_file: str = "test_queries.json", 
                             dataset_name: str = "Test Dataset for RAG Movie Expert Single"):
    
    client = Client()

    with open(json_file, "r") as f:
        examples = json.load(f)
    
    dataset_examples = []
    for ex in examples:
        dataset_examples.append({
            "input": {"query": ex["query"]},
            "output": {
                "reference_answer": ex["gold_movies"][0]["title"],
                "gold_movies": ex["gold_movies"][0],
            }
        })
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Synthetic movie queries for RAG evaluation"
    )

    client.create_examples(
        inputs=[ex["input"] for ex in dataset_examples],
        outputs=[ex["output"] for ex in dataset_examples],
        dataset_id=dataset.id
    )

    print(f"Dataset created: {dataset_name} (ID: {dataset.id}) with {len(dataset_examples)} examples")
    return dataset

def hit_rate_at_k(run, example, k=5):
    gold_id = example.outputs["gold_movies"][0]["tmdb_id"]
    retrieved = run.outputs["retrieved_docs"][:k]
    retrieved_ids = [doc["id"] for doc in retrieved]
    return {"score": 1.0 if gold_id in retrieved_ids else 0.0, "key": f"hit@{k}"}

def mrr(run, example):
    gold_id = example.outputs["gold_movies"][0]["tmdb_id"]
    retrieved = run.outputs["retrieved_docs"]
    for i, doc in enumerate(retrieved):
        if doc["id"] == gold_id:
            return {"score": 1.0 / (i + 1), "key": "mrr"}
    return {"score": 0.0, "key": "mrr"}