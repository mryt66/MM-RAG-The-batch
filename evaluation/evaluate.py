"""
Simple RAG Evaluation Script
Evaluates your RAG system and saves results to a simple JSON file.
No complex dependencies - just works!

Usage:
    python evaluate.py              # Run evaluation
    python evaluate.py --view       # View last results
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import initialize_system_chroma
from rag.retrieval import retrieve_relevant_articles, construct_prompt


RESULTS_FILE = Path(__file__).parent.parent / "evaluation_results.json"
QUESTIONS_FILE = Path(__file__).parent / "eval_questions.txt"


def rag_query(question: str, state) -> tuple:
    """Execute RAG query and return answer with contexts and token count"""
    # Retrieve relevant articles
    hits = retrieve_relevant_articles(state, question, top_k=5)
    
    # Extract contexts
    contexts = []
    for h in hits[:3]:
        article = h.get("article", {})
        text = article.get("text", "")[:500]
        title = article.get("title", "")
        if text:
            contexts.append({"title": title, "text": text})
    
    # Build prompt
    prompt, _ = construct_prompt(state, question, history_text="")
    
    # Generate answer and track tokens
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        answer = resp.text
        
        # Extract token usage
        usage = resp.usage_metadata if hasattr(resp, 'usage_metadata') else None
        if usage:
            input_tokens = usage.prompt_token_count if hasattr(usage, 'prompt_token_count') else 0
            output_tokens = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
            total_tokens = usage.total_token_count if hasattr(usage, 'total_token_count') else (input_tokens + output_tokens)
        else:
            input_tokens = output_tokens = total_tokens = 0
            
    except Exception as e:
        answer = f"Error: {e}"
        input_tokens = output_tokens = total_tokens = 0
    
    return answer, contexts, {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def evaluate_context_relevance(question: str, contexts: list) -> dict:
    """Evaluate how relevant the retrieved contexts are to the question"""
    context_text = "\n\n".join([f"Context {i+1}: {ctx['text'][:200]}..." for i, ctx in enumerate(contexts)])
    
    eval_prompt = f"""Rate how relevant these retrieved contexts are to answering the question on a scale of 0 to 1:
0 = completely irrelevant
1 = perfectly relevant

Question: {question}

Retrieved Contexts:
{context_text}

Format: SCORE: <number> REASONING: <text>"""
    
    try:
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
    context_text = "\n\n".join([f"Context {i+1}: {ctx['text'][:200]}..." for i, ctx in enumerate(contexts)])
    
    eval_prompt = f"""Rate how well this answer is grounded in (supported by) the provided contexts on a scale of 0 to 1:
0 = answer contradicts or is not supported by contexts
1 = answer is fully grounded in the contexts

Contexts:
{context_text}

Answer: {answer}

Format: SCORE: <number> REASONING: <text>"""
    
    try:
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


def run_evaluation():
    """Run complete evaluation"""
    print("="*60)
    print("RAG System Evaluation")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not set!")
        print("Set it with: $env:GEMINI_API_KEY='your-key'")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    # Load RAG system
    print("\n🔄 Loading RAG system...")
    state = initialize_system_chroma(show_progress=False)
    if not state:
        print("❌ Failed to load RAG system!")
        sys.exit(1)
    print("✅ RAG system ready")
    
    # Load questions
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = [
            "What are the latest developments in AI?",
            "How is ML used in healthcare?",
            "What are transformers in NLP?"
        ]
    
    print(f"\n📝 Evaluating {len(questions)} questions...")
    print("="*60)
    
    # Run evaluation
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        
        try:
            # Get answer
            print("  ⏳ Generating answer...")
            answer, contexts, token_usage = rag_query(question, state)
            
            # Evaluate all metrics
            print("  ⏳ Evaluating context relevance...")
            context_relevance = evaluate_context_relevance(question, contexts)
            
            print("  ⏳ Evaluating groundedness...")
            groundedness = evaluate_groundedness(answer, contexts)
            
            print("  ⏳ Evaluating answer relevance...")
            answer_relevance = evaluate_answer_relevance(question, answer)
            
            # Store result
            result = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "token_usage": token_usage,
                "metrics": {
                    "context_relevance": context_relevance,
                    "groundedness": groundedness,
                    "answer_relevance": answer_relevance
                },
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Show scores
            print(f"  📊 Metrics:")
            print(f"     • Context Relevance: {context_relevance['score']:.2f}")
            print(f"     • Groundedness: {groundedness['score']:.2f}")
            print(f"     • Answer Relevance: {answer_relevance['score']:.2f}")
            print(f"     • Total Tokens: {token_usage['total_tokens']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"Error: {e}",
                "contexts": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "metrics": {
                    "context_relevance": {"score": 0, "reasoning": str(e)},
                    "groundedness": {"score": 0, "reasoning": str(e)},
                    "answer_relevance": {"score": 0, "reasoning": str(e)}
                },
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results
    print("\n" + "="*60)
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "results": results
    }
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Calculate average metrics
    if results:
        avg_context_rel = sum(r["metrics"]["context_relevance"]["score"] for r in results) / len(results)
        avg_groundedness = sum(r["metrics"]["groundedness"]["score"] for r in results) / len(results)
        avg_answer_rel = sum(r["metrics"]["answer_relevance"]["score"] for r in results) / len(results)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in results)
    else:
        avg_context_rel = avg_groundedness = avg_answer_rel = total_tokens = 0
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Average Metrics:")
    print(f"   • Context Relevance: {avg_context_rel:.2f}")
    print(f"   • Groundedness: {avg_groundedness:.2f}")
    print(f"   • Answer Relevance: {avg_answer_rel:.2f}")
    print(f"   • Total Tokens Used: {total_tokens:,}")
    print(f"💾 Results saved to: {RESULTS_FILE}")
    print("="*60)
    
    print(f"\n📈 Summary:")
    for i, r in enumerate(results, 1):
        metrics = r["metrics"]
        avg_score = (metrics["context_relevance"]["score"] + metrics["groundedness"]["score"] + metrics["answer_relevance"]["score"]) / 3
        emoji = "🟢" if avg_score >= 0.7 else "🟡" if avg_score >= 0.4 else "🔴"
        print(f"  {i}. {emoji} {avg_score:.2f} - {r['question'][:50]}...")
    
    print(f"\n💡 To view detailed results: python evaluate.py --view")


def view_results():
    """View last evaluation results"""
    if not RESULTS_FILE.exists():
        print("❌ No evaluation results found. Run: python evaluate.py")
        return
    
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*60)
    print(f"Evaluation Results ({data['timestamp']})")
    print("="*60)
    
    results = data["results"]
    
    # Calculate averages
    if results:
        avg_context_rel = sum(r["metrics"]["context_relevance"]["score"] for r in results) / len(results)
        avg_groundedness = sum(r["metrics"]["groundedness"]["score"] for r in results) / len(results)
        avg_answer_rel = sum(r["metrics"]["answer_relevance"]["score"] for r in results) / len(results)
        total_tokens = sum(r["token_usage"]["total_tokens"] for r in results)
    else:
        avg_context_rel = avg_groundedness = avg_answer_rel = total_tokens = 0
    
    print(f"\n📊 Average Metrics:")
    print(f"   • Context Relevance: {avg_context_rel:.2f}")
    print(f"   • Groundedness: {avg_groundedness:.2f}")
    print(f"   • Answer Relevance: {avg_answer_rel:.2f}")
    print(f"   • Total Tokens: {total_tokens:,}")
    print(f"\n📝 Total Questions: {len(results)}")
    print("\n" + "="*60)
    
    for i, r in enumerate(results, 1):
        metrics = r["metrics"]
        tokens = r["token_usage"]
        
        context_rel = metrics["context_relevance"]["score"]
        groundedness = metrics["groundedness"]["score"]
        answer_rel = metrics["answer_relevance"]["score"]
        avg_score = (context_rel + groundedness + answer_rel) / 3
        
        emoji = "🟢" if avg_score >= 0.7 else "🟡" if avg_score >= 0.4 else "🔴"
        
        print(f"\n{i}. {emoji} Avg Score: {avg_score:.2f}/1.00")
        print(f"❓ Question: {r['question']}")
        print(f"💭 Answer: {r['answer'][:200]}...")
        print(f"\n� Detailed Metrics:")
        print(f"   • Context Relevance: {context_rel:.2f}")
        print(f"   • Groundedness: {groundedness:.2f}")
        print(f"   • Answer Relevance: {answer_rel:.2f}")
        print(f"   • Tokens Used: {tokens['total_tokens']}")
        print(f"📚 Contexts: {len(r['contexts'])} articles retrieved")
        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--view":
        view_results()
    else:
        run_evaluation()
