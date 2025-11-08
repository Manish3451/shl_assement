# streamlit_app.py
import sys
from pathlib import Path
import time
import os
import json
from src.services.chat import get_test_type_probs_via_llm

# ensure repo root is importable (run from repo root)
REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

# Define paths
VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store"
COMBINED_DATA_PATH = REPO_ROOT / "data" / "shl_all_assessments.json"

# Check if vector store exists, if not offer to build it
def check_and_build_vector_store():
    """Check if FAISS vector store exists, build if missing"""
    if not VECTOR_STORE_PATH.exists() or not any(VECTOR_STORE_PATH.iterdir()):
        st.warning("⚠️ Vector store not found. Building it for the first time...")
        
        with st.spinner("🔨 Building vector store (this may take 2-5 minutes)..."):
            try:
                # Check if combined data exists
                if not COMBINED_DATA_PATH.exists():
                    st.error("❌ Combined assessment data not found. Please run data preprocessing first:")
                    st.code("""
# Run these commands in your terminal:
cd backend/services
python scraper.py
python combine_res.py
                    """)
                    st.stop()

                
                # Import and run embedding generation
                from src.services.embedder import main as create_embeddings
                
                st.info("📊 Creating embeddings from assessment data...")
                create_embeddings()
                
                st.success("✅ Vector store created successfully!")
                st.balloons()
                time.sleep(2)
                st.rerun()
                
            except ImportError as e:
                st.error(f"❌ Import error: {str(e)}")
                st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
                st.stop()
            except Exception as e:
                st.error(f"❌ Failed to build vector store: {str(e)}")
                st.exception(e)
                st.stop()

# Run the check before proceeding
check_and_build_vector_store()

# Import your retriever and chat modules (after vector store check)
try:
    from src.services.retriever import retrieve_documents
    from src.services.chat import call_chat, format_recommendations
except Exception as e:
    st.error(f"❌ Failed to import modules: {str(e)}")
    st.info("Ensure vector store is built and all dependencies are installed.")
    st.stop()

# Basic UI
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🎯 SHL Assessment Recommendation System")
st.markdown("Find the perfect SHL assessment for your hiring needs using AI-powered search and recommendations.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show vector store status
    st.subheader("📦 System Status")
    if VECTOR_STORE_PATH.exists():
        st.success("✅ Vector store loaded")
        # Count files in vector store
        faiss_files = list(VECTOR_STORE_PATH.glob("*"))
        st.caption(f"{len(faiss_files)} files in vector store")
    else:
        st.error("❌ Vector store missing")
    
    st.markdown("---")
    
    st.subheader("Retrieval Settings")
    mode = st.selectbox(
        "Retrieval mode", 
        ["hybrid", "dense", "sparse"], 
        index=0,
        help="Hybrid combines keyword and semantic search"
    )
    
    alpha = st.slider(
        "Alpha (dense weight)", 
        0.0, 1.0, 0.7, 0.05,
        help="Weight for semantic search (1-alpha = keyword search weight)"
    )
    
    k = st.number_input(
        "Number of results (k)", 
        min_value=1, 
        max_value=50, 
        value=10, 
        step=1,
        help="How many assessments to retrieve and analyze"
    )
    
    st.subheader("LLM Settings")
    temperature = st.slider(
        "Temperature", 
        0.0, 2.0, 0.0, 0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.number_input(
        "Max tokens", 
        min_value=100, 
        max_value=4000, 
        value=1000, 
        step=100,
        help="Maximum length of LLM response"
    )
    
    st.markdown("---")
    
    st.subheader("🔍 Debug Options")
    show_raw_results = st.checkbox("Show raw retrieval results", value=False)
    show_context = st.checkbox("Show LLM context", value=False)
    show_timing = st.checkbox("Show timing details", value=True)
    
    st.markdown("---")
    
    st.subheader("Environment")
    # Check for API key in secrets or environment
    api_key_present = False
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key_present = True
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    elif 'OPENAI_API_KEY' in os.environ:
        api_key_present = True
    elif (REPO_ROOT / '.env').exists():
        api_key_present = True
    
    if api_key_present:
        st.success("✅ OpenAI API key detected")
    else:
        st.error("❌ OpenAI API key not found")
        st.info("Set OPENAI_API_KEY in Streamlit secrets or .env file")
    
    # Option to rebuild vector store
    st.markdown("---")
    if st.button("🔄 Rebuild Vector Store"):
        import shutil
        if VECTOR_STORE_PATH.exists():
            shutil.rmtree(VECTOR_STORE_PATH)
        st.rerun()

# Main content area
st.markdown("---")

# Query input
query = st.text_area(
    "🔎 Enter your assessment needs or job description",
    value="I need to assess problem-solving skills for software engineers",
    height=100,
    placeholder="E.g., 'Account manager assessment with sales focus' or 'Technical skills test for developers'"
)

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - "Account manager assessment with duration and format details"
    - "Technical skills test for software developers"
    - "Leadership assessment for senior managers"
    - "Customer service skills evaluation"
    - "Cognitive ability test for entry-level positions"
    """)

if not query.strip():
    st.warning("⚠️ Please enter a query to get started")
    st.stop()

# Main action buttons
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

with col_btn1:
    search_btn = st.button("🔍 Search Only", use_container_width=True, help="Retrieve assessments without LLM analysis")

with col_btn2:
    recommend_btn = st.button("✨ Get Recommendations", use_container_width=True, type="primary", help="Get AI-powered recommendations")

# Search Only functionality
if search_btn:
    with st.spinner("🔍 Searching for relevant assessments..."):
        start_time = time.time()
        
        try:
            # Retrieve documents
            results = retrieve_documents(
                query_text=query,
                alpha=alpha,
                k=k,
                mode=mode
            )
            
            elapsed = time.time() - start_time
            
            # Display results
            st.success(f"✅ Found {len(results)} assessments in {elapsed:.2f}s")
            
            if not results:
                st.warning("No assessments found. Try adjusting your query or settings.")
                st.stop()
            
            # Show results
            st.subheader(f"📋 Search Results ({len(results)} assessments)")
            
            for i, result in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        name = result.get("name", "Unknown Assessment")
                        st.markdown(f"### {i}. {name}")
                        
                        text = result.get("text", "")
                        if text:
                            preview = text[:300].replace("\n", " ")
                            st.write(preview + ("..." if len(text) > 300 else ""))
                    
                    with col2:
                        metadata = result.get("metadata", {})
                        url = metadata.get("url", "")
                        if url:
                            st.markdown(f"[🔗 View Details]({url})")
                    
                    st.divider()
            
            if show_timing:
                st.info(f"⏱️ Total time: {elapsed:.2f}s")
                
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            if st.checkbox("Show error details"):
                st.exception(e)

# Full Recommendation functionality (cleaned & non-duplicated)
if recommend_btn:
    start_all = time.time()
    classify_time = retrieval_time = llm_time = 0.0

    try:
        # Step 1: Classify query into test-type probabilities
        with st.spinner("🔎 Inferring query intent (test-type probabilities)..."):
            t0 = time.time()
            test_type_probs = get_test_type_probs_via_llm(query)
            t1 = time.time()
            classify_time = t1 - t0

        # Optionally display the probabilities
        st.subheader("Predicted test-type probabilities")
        for t in sorted(test_type_probs.keys()):
            st.write(f"{t}: {test_type_probs[t]:.2f}")

        # Step 2: Retrieve documents (pass test_type_probs so retriever can rerank)
        with st.spinner("🔍 Retrieving relevant assessments..."):
            r0 = time.time()
            results = retrieve_documents(
                query_text=query,
                alpha=alpha,
                k=k,
                mode=mode,
                test_type_probs=test_type_probs
            )
            r1 = time.time()
            retrieval_time = r1 - r0

        if not results:
            st.warning("⚠️ No assessments found. Try adjusting your query or settings.")
            st.stop()

        st.success(f"✅ Retrieved {len(results)} assessments in {retrieval_time:.2f}s")

        # Step 3: Call LLM for recommendations
        with st.spinner("🤖 Analyzing assessments and generating recommendations..."):
            t2 = time.time()
            chat_result = call_chat(
                query=query,
                candidates=results,
                temperature=temperature,
                max_tokens=max_tokens
            )
            t3 = time.time()
            llm_time = t3 - t2

        if not chat_result.get("success"):
            st.error(f"❌ LLM analysis failed: {chat_result.get('error', 'Unknown error')}")
            st.stop()

        st.success(f"✅ Analysis complete in {llm_time:.2f}s")

        # Step 4: Extract and display results
        parsed = chat_result.get("parsed_response", {})
        recommendations = parsed.get("recommendations", [])
        explanation = parsed.get("explanation", "")

        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🔍 Retrieved Assessments", "⚙️ Technical Details"])

        # --- Tab 1: Recommendations ---
        with tab1:
            st.header("✨ AI-Powered Recommendations")

            if explanation:
                st.markdown("### 📝 Analysis")
                st.info(explanation)
                st.markdown("---")

            if recommendations:
                st.markdown(f"### 🎯 Top {len(recommendations)} Recommended Assessments")
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        name = rec.get("name", "Unknown")
                        justification = rec.get("justification", "")
                        url = rec.get("url", "")

                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"#### {i}. {name}")
                            if justification:
                                st.write(justification)
                        with col2:
                            if url and url != "URL not available":
                                st.markdown(f"[🔗 Details]({url})")

                        st.divider()
            else:
                st.warning("No specific recommendations generated. See retrieved assessments in the next tab.")

            if show_context:
                with st.expander("📄 Full Formatted Response"):
                    st.text(format_recommendations(parsed))

        # --- Tab 2: Retrieved Assessments ---
        with tab2:
            st.header("🔍 All Retrieved Assessments")
            st.markdown(f"Showing all {len(results)} assessments retrieved from the database.")

            if show_raw_results:
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result.get('name', 'Unknown')}"):
                        st.json(result)
            else:
                for i, result in enumerate(results, 1):
                    name = result.get("name", "Unknown")
                    text = result.get("text", "")
                    metadata = result.get("metadata", {})
                    url = metadata.get("url", "")

                    st.markdown(f"### {i}. {name}")
                    if text:
                        preview = text[:400].replace("\n", " ")
                        st.write(preview + ("..." if len(text) > 400 else ""))
                    if url:
                        st.markdown(f"[🔗 View Full Assessment]({url})")
                    st.divider()

        # --- Tab 3: Technical Details ---
        with tab3:
            st.header("⚙️ Technical Details")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 Retrieval Info")
                st.json({
                    "mode": mode,
                    "alpha": alpha,
                    "k": k,
                    "num_retrieved": len(results),
                    "retrieval_time": f"{retrieval_time:.2f}s",
                    "classification_time": f"{classify_time:.2f}s"
                })

            with col2:
                st.subheader("🤖 LLM Info")
                usage = chat_result.get("usage", {})
                st.json({
                    "model": chat_result.get("model", "unknown"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompt_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "completion_tokens": usage.get("completion_tokens", 0) if usage else 0,
                    "total_tokens": usage.get("total_tokens", 0) if usage else 0,
                    "llm_time": f"{llm_time:.2f}s"
                })

            if show_context:
                st.subheader("📄 Raw LLM Response")
                st.code(chat_result.get("answer", ""), language="json")

            st.subheader("📊 Performance Summary")
            total_time = time.time() - start_all
            st.metric("Total Time", f"{total_time:.2f}s")

            perf_data = {
                "Classification": classify_time,
                "Retrieval": retrieval_time,
                "LLM Processing": llm_time,
                "Other": max(0, total_time - classify_time - retrieval_time - llm_time)
            }
            st.bar_chart(perf_data)

        if show_timing:
            st.markdown("---")
            st.success(
                f"✅ **Pipeline completed in {time.time() - start_all:.2f}s** "
                f"(Classification: {classify_time:.2f}s | Retrieval: {retrieval_time:.2f}s | LLM: {llm_time:.2f}s)"
            )

    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        if st.checkbox("Show full error traceback"):
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>SHL Assessment Recommendation System | Powered by OpenAI & LangChain</p>
</div>
""", unsafe_allow_html=True)
