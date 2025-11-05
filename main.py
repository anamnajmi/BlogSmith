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

st.set_page_config(page_title="Blog Lang", layout="wide")
st.title("Blog Lang")

topic = st.text_input("📝 Enter your blog topic", placeholder="e.g., The Future of AI in Education")

if st.button("Generate Blog"):
    if not topic.strip():
        st.warning("Please enter a blog topic first!")
    else:
        with st.spinner("Anam's Blog Lang is doing come cooking, Please Wait!"):
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


                st.success("✅ Blog generated successfully!")
                st.text_area("📰 Final Blog", safe_text, height=400)


                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, safe_text)
                pdf.output("result.pdf")

                # Provide download button
                with open("result.pdf", "rb") as f:
                    st.download_button(
                        label="Download Blog as PDF",
                        data=f,
                        file_name="blog_result.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
st.markdown("---")
st.markdown(
        '<p style="text-align:center;">Made with ❤️ by Anam</p>',
        unsafe_allow_html=True
    )