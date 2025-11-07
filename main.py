import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from fpdf import FPDF
import streamlit as st

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class BlogState(TypedDict):
    topic: str
    facts: str
    outline: str
    draft: str
    final_blog: str


def research_node(state: BlogState):
    topic = state["topic"]
    facts_prompt = f"List 10 factual and up-to-date insights about the topic '{topic}'."
    facts = llm.invoke(facts_prompt)
    return {"facts": facts.content}


def outline_node(state: BlogState):
    topic, facts = state["topic"], state["facts"]
    outline_prompt = f"""
    Create a clear, SEO-optimized blog outline for '{topic}' 
    based on these facts:
    {facts}
    """
    outline = llm.invoke(outline_prompt)
    return {"outline": outline.content}


def draft_node(state: BlogState):
    outline = state["outline"]
    draft_prompt = f"""
    Write a detailed and engaging blog post following this outline:
    {outline}

    Tone: Human, conversational, informative.
    Length: Around 800–1200 words.
    """
    draft = llm.invoke(draft_prompt)
    return {"draft": draft.content}


def rewrite_node(state: BlogState):
    draft = state["draft"]
    rewrite_prompt = f"""
    Rewrite this blog naturally to make it sound 100% human-written, 
    engaging, and plagiarism-free (no AI traces):
    {draft}
    """
    final = llm.invoke(rewrite_prompt)
    return {"final_blog": final.content}


graph = StateGraph(BlogState)
graph.add_node("research", research_node)
graph.add_node("outline", outline_node)
graph.add_node("draft", draft_node)
graph.add_node("rewrite", rewrite_node)
graph.add_edge("research", "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "rewrite")
graph.add_edge("rewrite", END)
graph.set_entry_point("research")
app = graph.compile()

st.set_page_config(
    page_title="Blog Lang | AI-Powered Blog Generator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
        color: #e2e8f0;
        min-height: 100vh;
    }

    /* Main Header */
    .main-header {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 3rem 2rem;
        color: #fff;
        text-align: center;
        position: relative;
        margin: 2rem auto 3rem;
        width: 90%;
        max-width: 900px;
        overflow: hidden;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .header-glow {
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 30% 30%, rgba(106, 17, 203, 0.3), transparent 50%),
            radial-gradient(circle at 70% 70%, rgba(37, 117, 252, 0.2), transparent 50%);
        animation: float 6s ease-in-out infinite;
        z-index: -1;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }

    .main-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        letter-spacing: -1px;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        opacity: 0.9;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.3rem 0;
        color: #c7d2fe;
    }

    /* Input Container */
    .input-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        width: 90%;
        max-width: 800px;
        margin: 0 auto 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Enhanced Input Fields */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div:hover {
        border-color: rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.08);
    }

    .stTextInput > div > div > input {
        background: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }

    /* Modern Button */
    .stButton button {
        border-radius: 14px;
        padding: 0.85rem 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 12px 25px rgba(106, 17, 203, 0.4),
            0 8px 15px rgba(37, 117, 252, 0.3);
    }

    .stButton button:hover::before {
        left: 100%;
    }

    /* Blog Output Area */
    .blog-output {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }

    /* Download Button */
    .download-btn {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%) !important;
    }

    .download-btn:hover {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 176, 155, 0.4) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #94a3b8;
        font-size: 0.95rem;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Progress Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .input-container {
            width: 95%;
            padding: 2rem;
        }

        .main-header {
            padding: 2rem 1rem;
            width: 95%;
        }

        .main-header h1 {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="main-header">
    <div class="header-glow"></div>
    <h1>✍️ Blog Lang</h1>
    <p>AI-Powered Blog Generation Platform</p>
    <p>Transform your ideas into engaging, human-like blog posts in seconds</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:


    topic = st.text_input(
        " **Enter your blog topic**",
        placeholder="e.g., The Future of AI in Education, Sustainable Living Tips, Digital Marketing Trends 2024...",
        key="topic_input"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("**Generate Blog Post**", use_container_width=True, key="generate_btn"):
        if not topic.strip():
            st.warning("⚠️ **Please enter a blog topic first!**")
        else:
            with st.spinner(""):
                progress_container = st.empty()
                progress_container.markdown("""
                <div style='text-align: center; padding: 2rem;'>
                    <div class='pulse-animation' style='font-size: 3rem; margin-bottom: 1rem;'>✨</div>
                    <h3 style='color: #a5b4fc; margin-bottom: 1rem;'>Blog Lang is crafting your masterpiece...</h3>
                    <p style='color: #94a3b8;'>Researching → Outlining → Writing → Refining</p>
                </div>
                """, unsafe_allow_html=True)

                try:
                    result = app.invoke({"topic": topic})
                    final_blog_text = result.get("final_blog", "No content generated.")

                    safe_text = (
                        final_blog_text
                        .replace("—", "-")
                        .replace("–", "-")
                        .replace("“", '"')
                        .replace("”", '"')
                        .replace("’", "'")
                    )

                    progress_container.empty()

                    st.success("""
                    🎉 **Blog generated successfully!** 
                    Your AI-crafted blog post is ready below.
                    """)


                    st.markdown("### 📖 **Your Generated Blog**")
                    st.markdown('<div class="blog-output">', unsafe_allow_html=True)
                    st.markdown(final_blog_text)
                    st.markdown('</div>', unsafe_allow_html=True)

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)

                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, topic, ln=True, align='C')
                    pdf.ln(10)
                    pdf.set_font("Arial", size=12)

                    pdf.multi_cell(0, 10, safe_text)
                    pdf.output("blog_post.pdf")

                    st.markdown("---")
                    st.markdown("### 📥 **Download Your Blog**")
                    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
                    with col_d2:
                        with open("blog_post.pdf", "rb") as f:
                            st.download_button(
                                label="📄 **Download as PDF**",
                                data=f,
                                file_name=f"{topic.replace(' ', '_')}_blog.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_btn"
                            )

                except Exception as e:
                    progress_container.empty()
                    st.error(f"""
                    ❌ **Something went wrong!**

                    Error: {str(e)}

                    Please try again with a different topic or check your internet connection.
                    """)

st.markdown("---")
st.markdown("""
<div class='footer'>
    <small>Made with ❤️ by Anam</small>
</div>
""", unsafe_allow_html=True)