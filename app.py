import ollama
import streamlit as st
from src.rag_pipeline import RagAssistant

# Page config
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🤖 RAG Q&A System")
st.markdown("""
Ask questions about **Retrieval-Augmented Generation** research papers.
The system retrieves relevant chunks from 32 ArXiv papers and generates answers using the selected model.
"""
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    options=["hybrid", "sparse", "dense"],
    index=0,  # Default to hybrid (best performance from evaluation)
    help="hybrid=BM25+Dense, sparse=BM25, dense=SentenceBERT",
)

top_k = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of chunks to retrieve",
)

# Try to list Ollama models, gracefully fallback if Ollama unavailable
try:
    ollama_models = [model.model for model in ollama.list().models]
except Exception:
    ollama_models = []

openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
model_options = openai_models.copy()
if ollama_models:
    model_options.append("─" * 10)  # Separator
    model_options.extend(ollama_models)

model = st.sidebar.selectbox(
    "LLM Model",
    options=model_options,
    index=0,
    help="OpenAI/Ollama model for answer generation",
)

# Prevent separator from being selected
if model and "─" in model:
    st.sidebar.warning("Please select a valid model")
    model = openai_models[0]  # Default to first OpenAI model

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.markdown(
    f"""
- **Documents**: 32 ArXiv papers
- **Chunks**: ~1,395 chunks
- **Chunk Size**: 500 tokens
- **Embedding Model**: all-MiniLM-L6-v2
"""
)

# Example questions
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Example Questions")
example_questions = [
    "What is Retrieval-Augmented Generation?",
    "How does hybrid retrieval work?",
    "What are the advantages of RAG over fine-tuning?",
    "What is the difference between dense and sparse retrieval?",
    "How does RRF fusion combine retrieval results?",
]
st.sidebar.markdown("\n".join([f"* {q}" for q in example_questions]))


# Load RAG assistant with caching
@st.cache_resource
def load_rag_assistant():
    """Load RAG assistant with specified configuration."""
    return RagAssistant()


# Initialize session state for question
if "question" not in st.session_state:
    st.session_state.question = ""

# Add example question buttons
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Example 1", use_container_width=True):
        st.session_state.question = example_questions[0]
with col2:
    if st.button("Example 2", use_container_width=True):
        st.session_state.question = example_questions[1]
with col3:
    if st.button("Example 3", use_container_width=True):
        st.session_state.question = example_questions[2]
with col4:
    if st.button("Example 4", use_container_width=True):
        st.session_state.question = example_questions[3]
with col5:
    if st.button("Example 5", use_container_width=True):
        st.session_state.question = example_questions[4]

# Main interface
question = st.text_area(
    "Your Question",
    value=st.session_state.question,
    height=100,
    placeholder="Type your question here...",
    help="Ask any question about RAG research",
)

# Update session state when user types
st.session_state.question = question

# Submit button
submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)

# Process question
if submit and question.strip():
    with st.spinner(f"Loading RAG index ({retrieval_mode} mode)..."):
        assistant = load_rag_assistant()

    with st.spinner("Retrieving relevant chunks and generating answer..."):
        try:
            # Get answer with context
            answer, context = assistant.query(
                question, model=model, retrieval_mode=retrieval_mode, top_k=top_k
            )

            # Display answer
            st.markdown("---")
            st.markdown("### ✅ Answer")
            st.markdown(answer)

            # Display retrieved chunks
            st.markdown("---")
            st.markdown(f"### 📄 Retrieved Chunks (Top {top_k})")

            if context:
                for i, chunk in enumerate(context, start=1):
                    with st.expander(
                        f"**Chunk {i}** - {chunk['metadata']['title']} - {','.join(chunk['metadata']['authors'])} (ArXiv ID: {chunk['metadata']['arxiv_id']})"
                    ):
                        # st.markdown(
                        #     f"**Similarity Score**: {chunk.get('score', 'N/A'):.4f}"
                        #     if isinstance(chunk.get("score"), (int, float))
                        #     else "**Similarity Score**: N/A"
                        # )
                        st.markdown(f"**Text**:")
                        st.text(chunk["chunk_text"])
            else:
                st.warning("No relevant chunks found.")

            # Display metadata
            st.markdown("---")
            st.markdown("### ℹ️ Query Info")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**Retrieval Mode**: {retrieval_mode}")
                st.markdown(f"**Top K**: {top_k}")
            with col_right:
                st.markdown(f"**Model**: {model}")
                st.markdown(f"**Chunks Retrieved**: {len(context)}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

elif submit and not question.strip():
    st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Built with Streamlit • Powered by OpenAI GPT-4o-mini •
    <a href='https://github.com/hongsly/ml-engineering-fundamentals' target='_blank'>GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)
