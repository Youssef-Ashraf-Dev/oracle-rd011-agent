import streamlit as st
import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key
import traceback

# Streamlit Page Config
st.set_page_config(page_title="RD.011 Generator", page_icon="📄", layout="wide")

# Initialize logging if not already done
if "logger_initialized" not in st.session_state:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    st.session_state.logger_initialized = True

logger = logging.getLogger(__name__)

# --- Directories Setup ---
UPLOAD_DIR = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = Path(".env")
if not ENV_PATH.exists():
    ENV_PATH.touch()
load_dotenv(ENV_PATH)

# Ensure config imports the keys properly after we might have updated them
import config

# Lazy load graph to avoid loading models until needed
@st.cache_resource(show_spinner=False)
def get_graph():
    from graph import build_graph
    return build_graph()

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "run_status" not in st.session_state:
    st.session_state.run_status = "idle" # idle, running, waiting_approval, completed, error
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

def save_uploaded_files(uploaded_files, prefix=""):
    paths = []
    for f in uploaded_files:
        if f is not None:
            file_path = UPLOAD_DIR / f"{prefix}{f.name}"
            with open(file_path, "wb") as out:
                out.write(f.read())
            paths.append(str(file_path))
    return paths

def update_env_key(key, value):
    set_key(str(ENV_PATH), key, value)
    os.environ[key] = value

def format_document_plan(plan):
    """Format the DocumentPlan (dict or Pydantic model) into Markdown."""
    if not plan:
        return "No plan available."
    
    if hasattr(plan, "model_dump"):
        plan_dict = plan.model_dump()
    elif hasattr(plan, "dict"):
        plan_dict = plan.dict()
    else:
        plan_dict = plan
        
    doc_title = plan_dict.get('document_title', '')
    md = ""
    if doc_title and str(doc_title).lower() != "untitled":
        md += f"### Document Title: {doc_title}\n\n"
    
    sections = plan_dict.get('sections', [])
    if not sections:
        md += "No sections drafted.\n"
    
    for sec in sections:
        md += f"#### {sec.get('module_name', 'Module')} (ID: {sec.get('section_id', 'N/A')})\n"
        md += f"_{sec.get('module_intro', '')}_\n\n"
        
        processes = sec.get('processes', [])
        for proc in processes:
            md += f"- **[{proc.get('process_id')}] {proc.get('process_name')}**\n"
            md += f"  - *Description:* {proc.get('process_description')}\n"
            md += f"  - *Expected Output:* {proc.get('output', 'N/A')}\n"
        md += "\n---\n"
    return md

def format_issue_report(report):
    if not report:
        return "No issues detected."
        
    if hasattr(report, "model_dump"):
        rep_dict = report.model_dump()
    elif hasattr(report, "dict"):
        rep_dict = report.dict()
    else:
        rep_dict = report
        
    issues = rep_dict.get("issues", [])
    if not issues:
        return "No issues detected."
        
    md = ""
    for issue in issues:
        md += f"- **{issue.get('severity', 'WARNING')}**: {issue.get('description', '')} (Module: {issue.get('affected_module', 'N/A')})\n"
    return md

# --- UI Components ---

# 1. Sidebar: API Keys & Settings
with st.sidebar:
    if os.path.exists("assets/RIS-01.jpg"):
        st.image("assets/RIS-01.jpg", use_container_width=True)
    st.title("⚙️ Settings")
    
    st.subheader("API Keys")
    openrouter_key = st.text_input("OpenRouter API Key", value=os.environ.get("OPENROUTER_API_KEY", ""), type="password")
    if openrouter_key and openrouter_key != os.environ.get("OPENROUTER_API_KEY", ""):
        update_env_key("OPENROUTER_API_KEY", openrouter_key)
        
    groq_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    if groq_key and groq_key != os.environ.get("GROQ_API_KEY", ""):
        update_env_key("GROQ_API_KEY", groq_key)
        
    google_key = st.text_input("Google API Key", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
    if google_key and google_key != os.environ.get("GOOGLE_API_KEY", ""):
        update_env_key("GOOGLE_API_KEY", google_key)
    
    st.markdown("---")
    is_running = st.session_state.run_status == "running"
    if is_running:
        st.warning("⏳ Pipeline is running. Please wait for it to finish before resetting.")
        st.button("Reset Session", disabled=True)
    else:
        if st.button("Reset Session"):
            st.session_state.thread_id = None
            st.session_state.run_status = "idle"
            st.session_state.current_result = None
            st.session_state.error_message = None
            st.rerun()

# 2. Main Content
if st.session_state.run_status == "idle":
    if os.path.exists("assets/Oracle-Partner.png"):
        st.image("assets/Oracle-Partner.png", use_container_width=True)
    st.title("📄 RD.011 Future Process Model Generator")
    action = st.radio("Choose Action:", ["Start New Generation", "Resume Existing Session"], horizontal=True)
    st.markdown("---")
    
    if action == "Start New Generation":
        st.markdown("Upload your project documents below to generate the RD.011 document.")
        
        with st.form("upload_form"):
            mom_files = st.file_uploader("Upload Minutes of Meeting (MoM) .docx (Mandatory)", type=["docx"], accept_multiple_files=True)
            scope_file = st.file_uploader("Upload Scope of Solution .docx (Optional)", type=["docx"])
            q_files = st.file_uploader("Upload Questionnaires .xlsx (Optional)", type=["xlsx"], accept_multiple_files=True)
            
            submitted = st.form_submit_button("Generate RD.011 Document", type="primary")
            
            if submitted:
                if not mom_files:
                    st.error("Please upload at least one mandatory MoM document.")
                elif not os.environ.get("GROQ_API_KEY") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
                    st.error("Please provide API keys in the sidebar.")
                else:
                    with st.spinner("Saving files..."):
                        mom_paths = save_uploaded_files(mom_files, "mom_")
                        scope_path = save_uploaded_files([scope_file] if scope_file else [], "scope_")
                        q_paths = save_uploaded_files(q_files, "q_")
                        
                        input_files = mom_paths + scope_path + q_paths
                        
                        st.session_state.thread_id = str(uuid.uuid4())[:8]
                        st.session_state.run_status = "running"
                        
                        initial_state = {
                            "thread_id": st.session_state.thread_id,
                            "input_files": input_files,
                            "raw_texts": {},
                            "extraction_result": None,
                            "document_plan": None,
                            "issue_report": None,
                            "consultant_approved": False,
                            "consultant_feedback": "",
                            "approval_iteration": 0,
                            "approval_maxed": False,
                            "intro_content": None,
                            "section_queue": [],
                            "current_section_index": 0,
                            "generated_sections": {},
                            "failed_sections": [],
                            "diagram_registry": {},
                            "output_path": None,
                            "errors": [],
                            "last_completed_node": "",
                        }
                        st.session_state.current_input = initial_state
                        st.rerun()

    else:
        # Resume Session Logic
        with st.form("resume_form"):
            thread_input = st.text_input("Enter Thread ID to resume:")
            resume_submitted = st.form_submit_button("Resume Session", type="primary")
            
            if resume_submitted:
                if not thread_input.strip():
                    st.error("Please enter a valid Thread ID.")
                else:
                    config_dict = {"configurable": {"thread_id": thread_input.strip()}}
                    snapshot = get_graph().get_state(config_dict)
                    
                    if not snapshot.tasks and not snapshot.values:
                        st.error(f"No checkpoint found for Thread ID: {thread_input}")
                    else:
                        st.session_state.thread_id = thread_input.strip()
                        st.session_state.current_result = snapshot.values
                        
                        if any(t.interrupts for t in snapshot.tasks):
                            st.session_state.run_status = "waiting_approval"
                        else:
                            st.session_state.run_status = "completed"
                        st.rerun()

elif st.session_state.run_status == "running":
    st.info(f"Pipeline is running... (Thread ID: {st.session_state.thread_id})")
    
    config_dict = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = st.session_state.get("current_input")
    graph = get_graph()
    
    try:
        with st.status("Executing agentic workflow...", expanded=True) as status:
            progress_placeholder = st.empty()
            
            for event in graph.stream(input_data, config=config_dict, stream_mode="updates"):
                for node_name, state_update in event.items():
                    if node_name == "generate_section":
                        snapshot = graph.get_state(config_dict)
                        curr_idx = snapshot.values.get("current_section_index", 0)
                        queue_len = len(snapshot.values.get("section_queue", []))
                        # Show the next section being worked on (or completed)
                        display_idx = min(curr_idx + 1, queue_len) if queue_len > 0 else 1
                        progress_placeholder.info(f"⏳ Generating section {display_idx} of {queue_len}...")
                    else:
                        st.write(f"✅ Completed step: **{node_name}**")
            
            progress_placeholder.empty()
            
            status.update(label="Workflow paused/completed!", state="complete", expanded=False)
            
            snapshot = graph.get_state(config_dict)
            st.session_state.current_result = snapshot.values
            
            if any(t.interrupts for t in snapshot.tasks):
                st.session_state.run_status = "waiting_approval"
            else:
                st.session_state.run_status = "completed"
            st.rerun()

    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        logger.error(traceback.format_exc())
        st.session_state.run_status = "error"
        st.session_state.error_message = str(e)
        st.rerun()

elif st.session_state.run_status == "waiting_approval":
    st.warning("⚠️ Consultant Approval Required")
    st.markdown("The agent has drafted a document plan based on the inputs. Please review it below.")
    
    state_values = st.session_state.current_result
    
    tab1, tab2 = st.tabs(["Document Plan", "Issue Report"])
    
    with tab1:
        if state_values.get("document_plan"):
            st.markdown(format_document_plan(state_values["document_plan"]))
        else:
            st.write("No document plan generated.")
            
    with tab2:
        if state_values.get("issue_report"):
            st.markdown(format_issue_report(state_values["issue_report"]))
        else:
            st.write("No issues detected.")
    
    st.markdown("---")
    st.subheader("Provide Feedback or Approve")
    
    feedback = st.text_area(
        "Feedback (leave blank if approving):", 
        key="feedback_box",
        help="If the plan needs changes, write instructions here."
    )
    
    has_feedback = bool(feedback.strip())
    
    col1, col2 = st.columns(2)
    with col1:
        approve_btn = st.button("✅ Approve Plan", type="primary", disabled=has_feedback)
    with col2:
        revise_btn = st.button("🔄 Submit Feedback for Revision", disabled=not has_feedback)
        
    if approve_btn:
        st.session_state.run_status = "running"
        from langgraph.types import Command
        st.session_state.current_input = Command(resume="APPROVE")
        st.session_state.feedback_box = ""
        st.rerun()
        
    elif revise_btn:
        st.session_state.run_status = "running"
        from langgraph.types import Command
        st.session_state.current_input = Command(resume=feedback)
        st.session_state.feedback_box = ""
        st.rerun()

elif st.session_state.run_status == "completed":
    st.balloons()
    st.toast("🎉 Document generation is complete! It is safe to close this window.", icon="✅")
    st.success("✅ RD.011 Document Generation Complete!")
    
    result = st.session_state.current_result
    output_path = result.get("output_path")
    
    if output_path and os.path.exists(str(output_path)):
        with open(str(output_path), "rb") as f:
            file_data = f.read()
            
        st.download_button(
            label="⬇️ Download RD.011 Document",
            data=file_data,
            file_name=os.path.basename(str(output_path)),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )
    else:
        st.error(f"Output file not found at: {output_path}. Check errors below.")
        
    if result.get("failed_sections"):
        st.warning(f"Failed sections (check manually): {result['failed_sections']}")
    if result.get("errors"):
        st.error(f"Errors encountered: {len(result['errors'])}")
        for err in result["errors"]:
            st.write(f"- {err}")

elif st.session_state.run_status == "error":
    st.error("❌ An error occurred during execution.")
    st.code(st.session_state.error_message)
    if st.button("Retry / Reset"):
        st.session_state.run_status = "idle"
        st.rerun()
