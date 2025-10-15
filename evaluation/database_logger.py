"""
Database Logger for RAG System
Automatically logs all prompts, outputs, and evaluations to SQLite database
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os
import re
import google.generativeai as genai

DB_FILE = Path("rag_logs.db")


def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create interactions table (stores all RAG queries and responses)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            contexts TEXT,
            num_contexts INTEGER,
            session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create evaluations table (stores evaluation metrics)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            context_relevance REAL,
            context_relevance_reasoning TEXT,
            groundedness REAL,
            groundedness_reasoning TEXT,
            answer_relevance REAL,
            answer_relevance_reasoning TEXT,
            total_tokens INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions(id)
        )
    """)
    
    conn.commit()
    conn.close()


def log_interaction(question: str, answer: str, contexts: list, session_id: str = None, token_usage: dict = None) -> int:
    """
    Log a RAG interaction to the database
    Returns the interaction ID
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO interactions (timestamp, question, answer, contexts, num_contexts, session_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        question,
        answer,
        json.dumps(contexts),
        len(contexts),
        session_id or "default"
    ))
    
    interaction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # If token usage provided, create initial evaluation entry with token data
    if token_usage:
        _log_token_usage(interaction_id, token_usage)
    
    return interaction_id


def _log_token_usage(interaction_id: int, token_usage: dict):
    """Log token usage for an interaction"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO evaluations (
            interaction_id, total_tokens, input_tokens, output_tokens
        )
        VALUES (?, ?, ?, ?)
    """, (
        interaction_id,
        token_usage.get("total_tokens", 0),
        token_usage.get("input_tokens", 0),
        token_usage.get("output_tokens", 0)
    ))
    
    conn.commit()
    conn.close()


def evaluate_context_relevance(question: str, contexts: list) -> dict:
    """Evaluate how relevant the retrieved contexts are to the question"""
    context_text = "\n\n".join([f"Context {i+1}: {ctx.get('text', '')[:200]}..." for i, ctx in enumerate(contexts)])
    
    eval_prompt = f"""Rate how relevant these retrieved contexts are to answering the question on a scale of 0 to 1:
0 = completely irrelevant
1 = perfectly relevant

Question: {question}

Retrieved Contexts:
{context_text}

Format: SCORE: <number> REASONING: <text>"""
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"score": 0.5, "reasoning": "No API key"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(eval_prompt)
        text = resp.text.strip()
        
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))
        
        reasoning_match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
        
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Error: {str(e)}"}


def evaluate_groundedness(answer: str, contexts: list) -> dict:
    """Evaluate how well the answer is grounded in the provided contexts"""
    context_text = "\n\n".join([f"Context {i+1}: {ctx.get('text', '')[:200]}..." for i, ctx in enumerate(contexts)])
    
    eval_prompt = f"""Rate how well this answer is grounded in (supported by) the provided contexts on a scale of 0 to 1:
0 = answer contradicts or is not supported by contexts
1 = answer is fully grounded in the contexts

Contexts:
{context_text}

Answer: {answer}

Format: SCORE: <number> REASONING: <text>"""
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"score": 0.5, "reasoning": "No API key"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(eval_prompt)
        text = resp.text.strip()
        
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))
        
        reasoning_match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
        
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Error: {str(e)}"}


def evaluate_answer_relevance(question: str, answer: str) -> dict:
    """Evaluate how relevant the answer is to the question"""
    eval_prompt = f"""Rate how relevant this answer is to the question on a scale of 0 to 1:
0 = completely irrelevant
1 = perfectly relevant and addresses the question

Question: {question}
Answer: {answer}

Format: SCORE: <number> REASONING: <text>"""
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"score": 0.5, "reasoning": "No API key"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(eval_prompt)
        text = resp.text.strip()
        
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))
        
        reasoning_match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
        
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Error: {str(e)}"}


def log_evaluation(interaction_id: int, metrics: dict):
    """
    Log evaluation metrics for an interaction
    metrics should contain: context_relevance, groundedness, answer_relevance, token_usage
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO evaluations (
            interaction_id,
            context_relevance, context_relevance_reasoning,
            groundedness, groundedness_reasoning,
            answer_relevance, answer_relevance_reasoning,
            total_tokens, input_tokens, output_tokens
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        interaction_id,
        metrics.get("context_relevance", {}).get("score", 0),
        metrics.get("context_relevance", {}).get("reasoning", ""),
        metrics.get("groundedness", {}).get("score", 0),
        metrics.get("groundedness", {}).get("reasoning", ""),
        metrics.get("answer_relevance", {}).get("score", 0),
        metrics.get("answer_relevance", {}).get("reasoning", ""),
        metrics.get("token_usage", {}).get("total_tokens", 0),
        metrics.get("token_usage", {}).get("input_tokens", 0),
        metrics.get("token_usage", {}).get("output_tokens", 0)
    ))
    
    conn.commit()
    conn.close()


def auto_evaluate_interaction(interaction_id: int):
    """
    Automatically evaluate an interaction after it's logged
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get interaction data
    cursor.execute("""
        SELECT question, answer, contexts
        FROM interactions
        WHERE id = ?
    """, (interaction_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return
    
    question, answer, contexts_json = row
    contexts = json.loads(contexts_json) if contexts_json else []
    
    # Evaluate all metrics
    print(f"  🔄 Auto-evaluating interaction #{interaction_id}...")
    
    context_relevance = evaluate_context_relevance(question, contexts)
    groundedness = evaluate_groundedness(answer, contexts)
    answer_relevance = evaluate_answer_relevance(question, answer)
    
    # Update evaluation with scores (preserving existing token data if present)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if evaluation entry exists
    cursor.execute("SELECT id FROM evaluations WHERE interaction_id = ?", (interaction_id,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing entry (preserve tokens)
        cursor.execute("""
            UPDATE evaluations SET
                context_relevance = ?,
                context_relevance_reasoning = ?,
                groundedness = ?,
                groundedness_reasoning = ?,
                answer_relevance = ?,
                answer_relevance_reasoning = ?,
                evaluated_at = CURRENT_TIMESTAMP
            WHERE interaction_id = ?
        """, (
            context_relevance['score'],
            context_relevance['reasoning'],
            groundedness['score'],
            groundedness['reasoning'],
            answer_relevance['score'],
            answer_relevance['reasoning'],
            interaction_id
        ))
    else:
        # Create new entry (without tokens - they weren't logged)
        cursor.execute("""
            INSERT INTO evaluations (
                interaction_id,
                context_relevance, context_relevance_reasoning,
                groundedness, groundedness_reasoning,
                answer_relevance, answer_relevance_reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            context_relevance['score'],
            context_relevance['reasoning'],
            groundedness['score'],
            groundedness['reasoning'],
            answer_relevance['score'],
            answer_relevance['reasoning']
        ))
    
    conn.commit()
    conn.close()
    
    print(f"  ✅ Evaluation complete: CR={context_relevance['score']:.2f}, G={groundedness['score']:.2f}, AR={answer_relevance['score']:.2f}")
    
    # Return the evaluation scores
    return {
        "context_relevance": context_relevance['score'],
        "groundedness": groundedness['score'],
        "answer_relevance": answer_relevance['score']
    }


def get_interaction_evaluation(interaction_id: int) -> dict:
    """Get evaluation metrics for a specific interaction"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            context_relevance,
            groundedness,
            answer_relevance,
            total_tokens,
            input_tokens,
            output_tokens
        FROM evaluations
        WHERE interaction_id = ?
    """, (interaction_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "context_relevance": row[0],
        "groundedness": row[1],
        "answer_relevance": row[2],
        "total_tokens": row[3],
        "input_tokens": row[4],
        "output_tokens": row[5]
    }


def get_all_interactions():
    """Get all interactions with their evaluations"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            i.id, i.timestamp, i.question, i.answer, i.contexts, i.num_contexts,
            e.context_relevance, e.context_relevance_reasoning,
            e.groundedness, e.groundedness_reasoning,
            e.answer_relevance, e.answer_relevance_reasoning,
            e.total_tokens, e.input_tokens, e.output_tokens
        FROM interactions i
        LEFT JOIN evaluations e ON i.id = e.interaction_id
        ORDER BY i.id DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "timestamp": row[1],
            "question": row[2],
            "answer": row[3],
            "contexts": json.loads(row[4]) if row[4] else [],
            "num_contexts": row[5],
            "context_relevance": row[6],
            "context_relevance_reasoning": row[7],
            "groundedness": row[8],
            "groundedness_reasoning": row[9],
            "answer_relevance": row[10],
            "answer_relevance_reasoning": row[11],
            "total_tokens": row[12],
            "input_tokens": row[13],
            "output_tokens": row[14]
        })
    
    return results


def get_unevaluated_interactions():
    """Get all interactions that haven't been evaluated yet"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT i.id, i.question, i.answer, i.contexts
        FROM interactions i
        LEFT JOIN evaluations e ON i.id = e.interaction_id
        WHERE e.id IS NULL
        ORDER BY i.id
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "contexts": json.loads(row[3]) if row[3] else []
        })
    
    return results


def evaluate_all_unevaluated():
    """Evaluate all interactions that don't have evaluations yet"""
    unevaluated = get_unevaluated_interactions()
    
    if not unevaluated:
        print("✅ All interactions already evaluated!")
        return 0
    
    print(f"🔄 Found {len(unevaluated)} unevaluated interactions. Starting evaluation...")
    
    for item in unevaluated:
        auto_evaluate_interaction(item["id"])
    
    print(f"✅ Evaluation complete! Evaluated {len(unevaluated)} interactions.")
    return len(unevaluated)


# Initialize database on import
init_database()


if __name__ == "__main__":
    # Test the database
    print("Testing database logger...")
    
    # Test logging an interaction
    test_contexts = [
        {"title": "Test Article 1", "text": "This is test content about AI."},
        {"title": "Test Article 2", "text": "More information about machine learning."}
    ]
    
    interaction_id = log_interaction(
        question="What is AI?",
        answer="AI stands for Artificial Intelligence.",
        contexts=test_contexts
    )
    
    print(f"✅ Logged interaction #{interaction_id}")
    
    # Test auto-evaluation
    auto_evaluate_interaction(interaction_id)
    
    print("\n✅ Database test complete!")
