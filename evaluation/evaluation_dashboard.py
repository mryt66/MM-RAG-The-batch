"""
RAG Evaluation Dashboard
Interactive Streamlit dashboard to view evaluation results - similar to TruLens interface

Usage:
    streamlit run evaluation_dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from evaluation.database_logger import get_all_interactions, evaluate_all_unevaluated, get_unevaluated_interactions

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")

DB_FILE = Path(__file__).parent.parent / "rag_logs.db"


def create_leaderboard_df(data):
    """Create leaderboard dataframe from results"""
    results = data.get("results", [])
    
    rows = []
    for i, result in enumerate(results, 1):
        metrics = result.get("metrics", {})
        context_rel = metrics.get("context_relevance", {}).get("score", 0.0)
        groundedness = metrics.get("groundedness", {}).get("score", 0.0)
        answer_rel = metrics.get("answer_relevance", {}).get("score", 0.0)
        avg_score = (context_rel + groundedness + answer_rel) / 3
        
        tokens = result.get("token_usage", {}).get("total_tokens", 0)
        
        rows.append({
            "Q#": i,
            "Question": result.get("question", "")[:50] + "...",
            "Avg": avg_score,
            "Context": context_rel,
            "Ground": groundedness,
            "Answer": answer_rel,
            "Tokens": tokens,
            "Status": "🟢" if avg_score >= 0.7 else "🟡" if avg_score >= 0.4 else "🔴"
        })
    
    return pd.DataFrame(rows)


def main():
    st.title("📊 RAG System Evaluation Dashboard")
    st.markdown("---")
    
    # Always use database as the data source
    if not DB_FILE.exists():
        st.warning("⚠️ No database found! Start using the chat to log interactions.")
        st.info("Interactions are automatically logged when you use the RAG chat interface.")
        return
    
    # Get all interactions from database
    db_interactions = get_all_interactions()
    
    if not db_interactions:
        st.warning("⚠️ No interactions logged yet!")
        st.info("Use the RAG chat interface to log your first interaction.")
        return
    
    # Check for unevaluated interactions
    unevaluated = get_unevaluated_interactions()
    
    if unevaluated:
        st.info(f"ℹ️ {len(unevaluated)} interactions need evaluation. Click 'Evaluate All' below.")
        if st.button("🔄 Evaluate All Unevaluated Interactions", type="primary"):
            with st.spinner(f"Evaluating {len(unevaluated)} interactions..."):
                count = evaluate_all_unevaluated()
            st.success(f"✅ Evaluated {count} interactions!")
            st.rerun()
    
    # Convert database data to format expected by dashboard
    data = {"results": []}
    for item in db_interactions:
        if item["context_relevance"] is None:
            continue  # Skip unevaluated
        
        data["results"].append({
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "token_usage": {
                "total_tokens": item["total_tokens"] or 0,
                "input_tokens": item["input_tokens"] or 0,
                "output_tokens": item["output_tokens"] or 0
            },
            "metrics": {
                "context_relevance": {
                    "score": item["context_relevance"] or 0,
                    "reasoning": item["context_relevance_reasoning"] or ""
                },
                "groundedness": {
                    "score": item["groundedness"] or 0,
                    "reasoning": item["groundedness_reasoning"] or ""
                },
                "answer_relevance": {
                    "score": item["answer_relevance"] or 0,
                    "reasoning": item["answer_relevance_reasoning"] or ""
                }
            },
            "timestamp": item["timestamp"]
        })
    
    if not data["results"]:
        st.warning("⚠️ No evaluated interactions yet!")
        st.info("Interactions are logged but not yet evaluated. Click 'Evaluate All' above.")
        return
    
    # Sidebar - Evaluation Info
    with st.sidebar:
        st.header("📋 Evaluation Info")
        timestamp = data.get("timestamp", "Unknown")
        total_questions = data.get("total_questions", 0)
        results = data.get("results", [])
        
        st.metric("Run Time", timestamp)
        st.metric("Total Questions", total_questions)
        
        # Calculate average metrics
        if results:
            context_scores = [r.get("metrics", {}).get("context_relevance", {}).get("score", 0.0) for r in results]
            ground_scores = [r.get("metrics", {}).get("groundedness", {}).get("score", 0.0) for r in results]
            answer_scores = [r.get("metrics", {}).get("answer_relevance", {}).get("score", 0.0) for r in results]
            total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
            
            avg_context = sum(context_scores) / len(context_scores)
            avg_ground = sum(ground_scores) / len(ground_scores)
            avg_answer = sum(answer_scores) / len(answer_scores)
            overall_avg = (avg_context + avg_ground + avg_answer) / 3
        else:
            avg_context = avg_ground = avg_answer = overall_avg = total_tokens = 0
        
        st.metric("Overall Avg Score", f"{overall_avg:.2f}")
        st.metric("Total Tokens", f"{total_tokens:,}")
        
        st.markdown("---")
        st.markdown("### 📊 Average Metrics")
        st.metric("Context Relevance", f"{avg_context:.2f}")
        st.metric("Groundedness", f"{avg_ground:.2f}")
        st.metric("Answer Relevance", f"{avg_answer:.2f}")
        
        st.markdown("---")
        st.markdown("### 🎯 Score Ranges")
        st.markdown("🟢 **Good**: ≥ 0.70")
        st.markdown("🟡 **Fair**: 0.40 - 0.69")
        st.markdown("🔴 **Poor**: < 0.40")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "📈 Analytics", "🔍 Detailed Results"])
    
    with tab1:
        st.header("Leaderboard")
        
        # Create leaderboard table
        df = create_leaderboard_df(data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg": st.column_config.ProgressColumn(
                    "Avg Score",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Context": st.column_config.ProgressColumn(
                    "Context Rel",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Ground": st.column_config.ProgressColumn(
                    "Groundedness",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Answer": st.column_config.ProgressColumn(
                    "Answer Rel",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Tokens": st.column_config.NumberColumn(
                    "Tokens",
                    format="%d",
                ),
            }
        )
    
    with tab2:
        st.header("Analytics")
        
        # Calculate all metrics
        context_scores = [r.get("metrics", {}).get("context_relevance", {}).get("score", 0.0) for r in results]
        ground_scores = [r.get("metrics", {}).get("groundedness", {}).get("score", 0.0) for r in results]
        answer_scores = [r.get("metrics", {}).get("answer_relevance", {}).get("score", 0.0) for r in results]
        avg_scores = [(c + g + a) / 3 for c, g, a in zip(context_scores, ground_scores, answer_scores)]
        token_counts = [r.get("token_usage", {}).get("total_tokens", 0) for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # All metrics by question
            st.subheader("📊 All Metrics by Question")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Context Relevance', x=[f"Q{i+1}" for i in range(len(results))], 
                                y=context_scores, marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Groundedness', x=[f"Q{i+1}" for i in range(len(results))], 
                                y=ground_scores, marker_color='lightgreen'))
            fig.add_trace(go.Bar(name='Answer Relevance', x=[f"Q{i+1}" for i in range(len(results))], 
                                y=answer_scores, marker_color='lightyellow'))
            
            fig.update_layout(
                barmode='group',
                title="Metrics Comparison",
                xaxis_title="Question",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average score categories
            st.subheader("📈 Performance Distribution")
            good = sum(1 for s in avg_scores if s >= 0.7)
            fair = sum(1 for s in avg_scores if 0.4 <= s < 0.7)
            poor = sum(1 for s in avg_scores if s < 0.4)
            
            fig = go.Figure(data=[go.Pie(
                labels=['Good (≥0.70)', 'Fair (0.40-0.69)', 'Poor (<0.40)'],
                values=[good, fair, poor],
                marker_colors=['green', 'orange', 'red'],
                hole=0.3
            )])
            fig.update_layout(title="Overall Performance", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Token usage
        st.subheader("🪙 Token Usage")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tokens", f"{sum(token_counts):,}")
        col2.metric("Avg per Question", f"{sum(token_counts)/len(token_counts) if token_counts else 0:.0f}")
        col3.metric("Min Tokens", f"{min(token_counts) if token_counts else 0:,}")
        col4.metric("Max Tokens", f"{max(token_counts) if token_counts else 0:,}")
        
        # Token usage chart
        fig = go.Figure(data=[go.Bar(
            x=[f"Q{i+1}" for i in range(len(results))],
            y=token_counts,
            marker_color='mediumpurple',
            text=token_counts,
            textposition='auto',
        )])
        fig.update_layout(
            title="Token Usage by Question",
            xaxis_title="Question",
            yaxis_title="Tokens",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Detailed Results")
        
        # Display each question in detail
        for i, result in enumerate(results, 1):
            question = result.get("question", "")
            answer = result.get("answer", "")
            contexts = result.get("contexts", [])
            metrics = result.get("metrics", {})
            tokens = result.get("token_usage", {})
            
            context_rel = metrics.get("context_relevance", {}).get("score", 0.0)
            groundedness = metrics.get("groundedness", {}).get("score", 0.0)
            answer_rel = metrics.get("answer_relevance", {}).get("score", 0.0)
            avg_score = (context_rel + groundedness + answer_rel) / 3
            
            # Status indicator
            status = "🟢 Good" if avg_score >= 0.7 else "🟡 Fair" if avg_score >= 0.4 else "🔴 Poor"
            
            with st.expander(f"**Q{i}: {question[:80]}...** - {status} ({avg_score:.2f})"):
                st.markdown(f"**Question:** {question}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Answer:**")
                    st.info(answer)
                    
                    st.markdown("**Evaluation Details:**")
                    
                    st.markdown("*Context Relevance:*")
                    st.success(f"**Score: {context_rel:.2f}** - {metrics.get('context_relevance', {}).get('reasoning', '')}")
                    
                    st.markdown("*Groundedness:*")
                    st.success(f"**Score: {groundedness:.2f}** - {metrics.get('groundedness', {}).get('reasoning', '')}")
                    
                    st.markdown("*Answer Relevance:*")
                    st.success(f"**Score: {answer_rel:.2f}** - {metrics.get('answer_relevance', {}).get('reasoning', '')}")
                
                with col2:
                    st.metric("Average Score", f"{avg_score:.2f}/1.00")
                    st.metric("Total Tokens", f"{tokens.get('total_tokens', 0):,}")
                    st.metric("Input Tokens", f"{tokens.get('input_tokens', 0):,}")
                    st.metric("Output Tokens", f"{tokens.get('output_tokens', 0):,}")
                    st.metric("Contexts Used", len(contexts))
                    
                    if contexts:
                        st.markdown("**Context Sources:**")
                        for j, ctx in enumerate(contexts, 1):
                            title = ctx.get("title", "Unknown")
                            st.caption(f"{j}. {title}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    main()
