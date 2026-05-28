import streamlit as st
import os
import base64
import json
from utils import theme

st.set_page_config(page_title="Project Documentation Viewer | SwipeIQ V2", page_icon="📄", layout="wide")
theme.inject_css()
theme.render_sidebar()

st.title("📄 Project Documentation Viewer")
st.markdown("Browse the core assignment files and research notebooks without leaving the dashboard.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tab1, tab2, tab3 = st.tabs(["📑 Assignment Report (PDF)", "📓 Jupyter Notebook", "📝 Project Notes (MD)"])

with tab1:
    st.markdown("### 📑 Official Group Assignment Report")
    st.markdown("Tying the (Data) Knot: Predicting Meaningful Connections")
    pdf_path = os.path.join(ROOT_DIR, 'reports', 'WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.1(long).pdf')
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"PDF report not found at: {pdf_path}")
        
    st.markdown("---")
    st.markdown("### 🗺️ Project Architecture Mind Map")
    mind_map_path = os.path.join(ROOT_DIR, "reports", "NotebookLM Mind Map.png")
    if os.path.exists(mind_map_path):
        st.image(mind_map_path, caption="NotebookLM Project Taxonomy & Methodology Mind Map", use_container_width=True)

with tab2:
    st.markdown("### 📓 Master Pipeline Notebook")
    st.markdown("Read-only rendering of the `ML_dating_app_behaviour V5.ipynb` source file.")
    ipynb_path = os.path.join(ROOT_DIR, 'notebooks', 'ML_dating_app_behaviour V5.ipynb')
    
    if os.path.exists(ipynb_path):
        with st.container():
            st.markdown("<div style='background-color:rgba(0,0,0,0.2); padding: 20px; border-radius:8px;'>", unsafe_allow_html=True)
            try:
                with open(ipynb_path, "r", encoding="utf-8") as f:
                    notebook = json.load(f)
                    
                for cell in notebook.get("cells", []):
                    if cell["cell_type"] == "markdown":
                        source = "".join(cell.get("source", []))
                        st.markdown(source)
                    elif cell["cell_type"] == "code":
                        source = "".join(cell.get("source", []))
                        if source.strip():
                            st.code(source, language="python")
                        
                        # Parse outputs
                        for output in cell.get("outputs", []):
                            if output.get("output_type") == "stream":
                                text = "".join(output.get("text", []))
                                if text.strip():
                                    st.text(text)
                            elif output.get("output_type") in ["display_data", "execute_result"]:
                                if "data" in output:
                                    if "image/png" in output["data"]:
                                        img_data = base64.b64decode(output["data"]["image/png"])
                                        st.image(img_data, use_container_width=True)
                                    elif "text/html" in output["data"]:
                                        html_str = "".join(output["data"]["text/html"])
                                        st.components.v1.html(html_str, height=300, scrolling=True)
                                    elif "text/plain" in output["data"]:
                                        text = "".join(output["data"]["text/plain"])
                                        st.text(text)
            except Exception as e:
                st.error(f"Error reading notebook: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error(f"Notebook file not found at: {ipynb_path}")

with tab3:
    st.markdown("### 📝 Master Project Notes")
    md_path = os.path.join(ROOT_DIR, "PROJECT_NOTES.md")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 30px;">
        """, unsafe_allow_html=True)
        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error(f"Project Notes not found at: {md_path}")




# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
