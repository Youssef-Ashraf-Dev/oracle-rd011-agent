import streamlit as st
import nest_asyncio
# nest_asyncio.apply()  # Allow nested event loops (fixes asyncio/Streamlit threading conflict)
import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key
import traceback

# Streamlit Page Config
st.set_page_config(page_title="RD.011 Generator", page_icon="📄", layout="wide")

# --- CUSTOM CSS TO HIDE FULLSCREEN BUTTON ---
st.markdown(
    """
    <style>
    /* This targets the button that appears on hover over images */
    button[title="View fullscreen"] {
        display: none;
    }
    /* Hide navigator buttons for images/pdfs */
    button[title="Previous page"],
    button[title="Next page"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
# NOTE: Do NOT cache the graph with @st.cache_resource because it contains
# a live SQLite connection (SqliteSaver). Caching would cause connection issues.
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

# --- Session Recovery Detection ---
# Runs ONCE per browser session (tracked by _recovery_checked flag).
# Only sets a flag — the UI below handles the user's choice.
if "_recovery_checked" not in st.session_state:
    st.session_state._recovery_checked = False
if "recoverable_thread" not in st.session_state:
    st.session_state.recoverable_thread = None
if "_recoverable_status" not in st.session_state:
    st.session_state._recoverable_status = None

if not st.session_state._recovery_checked and st.session_state.thread_id is None:
    st.session_state._recovery_checked = True  # never run again this browser session
    try:
        import sqlite3 as _sqlite3
        _db_path = config.CHECKPOINT_DB_PATH
        if os.path.exists(_db_path):
            _conn = _sqlite3.connect(_db_path)
            _cursor = _conn.execute(
                "SELECT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT 1"
            )
            _row = _cursor.fetchone()
            _conn.close()
            if _row:
                _last_thread = _row[0]
                _graph = get_graph()
                _snapshot = _graph.get_state({"configurable": {"thread_id": _last_thread}})
                if _snapshot and _snapshot.values:
                    if any(t.interrupts for t in (_snapshot.tasks or [])):
                        st.session_state.recoverable_thread = _last_thread
                        st.session_state._recoverable_status = "waiting_approval"
                    elif _snapshot.values.get("output_path"):
                        st.session_state.recoverable_thread = _last_thread
                        st.session_state._recoverable_status = "completed"
    except Exception as _recovery_err:
        logger.debug("Session recovery detection skipped: %s", _recovery_err)



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
    if os.path.exists("assets/logo_square.jpg"):
        st.image("assets/logo_square.jpg", width="stretch")
    st.title("⚙️ Configuration")
    
    # Check if API keys are configured
    api_keys_configured = all([
        os.environ.get("OPENROUTER_API_KEY"),
        os.environ.get("GROQ_API_KEY")
    ])
    
    if not api_keys_configured:
        st.warning("⚠️ **API Keys Not Configured**")
        st.markdown("""
**For Streamlit Cloud Deployment:**
1. In your Streamlit Cloud dashboard, go to your app's settings.
2. Click on the "Secrets" tab.
3. Add the following secrets:
   - `OPENROUTER_API_KEY`
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY` (optional)

**For Local Development:**
1. Create `.env` file with your keys
2. Restart the app
        """)
    else:
        st.success("✅ API Keys configured")
    
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
            # Reset recovery detection so next run can detect a fresh session
            st.session_state._recovery_checked = False
            st.session_state.recoverable_thread = None
            st.session_state._recoverable_status = None
            # Reset feedback state for a clean approval UI on the next run
            st.session_state.feedback_key_counter = 0
            st.session_state._feedback_has_content = False
            st.rerun()


# 2. Main Content
if os.path.exists("assets/logo_wide.png"):
    st.image("assets/logo_wide.png", width="stretch")
st.title("📄 RD.011 Future Process Model Generator")

if st.session_state.run_status == "idle":
    action = st.radio("Choose Action:", ["Start New Generation", "Resume Existing Session"], horizontal=True)
    st.markdown("---")
    
    # Notify about a recoverable session — Resume button only.
    # User can ignore this and upload a new document below.
    if st.session_state.recoverable_thread:
        _rec_status = st.session_state.get("_recoverable_status", "interrupted")
        _status_label = "waiting for your approval" if _rec_status == "waiting_approval" else "completed (document ready)"
        st.info(
            f"📌 Previous session found (Thread: **{st.session_state.recoverable_thread}**, "
            f"Status: {_status_label}). Click Resume to continue it, or upload new files below to start fresh."
        )
        if st.button("▶ Resume Previous Session", type="primary"):
            _thread = st.session_state.recoverable_thread
            _graph = get_graph()
            _snapshot = _graph.get_state({"configurable": {"thread_id": _thread}})
            st.session_state.thread_id = _thread
            st.session_state.current_result = _snapshot.values
            st.session_state.run_status = st.session_state._recoverable_status
            st.session_state.recoverable_thread = None
            logger.info("User resumed session: thread_id=%s", _thread)
            st.rerun()
        st.markdown("---")
    
    if action == "Start New Generation":
        st.markdown("Upload your project documents below to generate the RD.011 document.")
        
        with st.form("upload_form"):
            mom_files = st.file_uploader("Upload Minutes of Meeting (MoM) .docx (Mandatory)", type=["docx"], accept_multiple_files=True)
            q_files = st.file_uploader("Upload Questionnaires .xlsx (Optional)", type=["xlsx"], accept_multiple_files=True)
            scope_file = st.file_uploader("Upload Scope of Solution .docx (Optional)", type=["docx"])
            
            submitted = st.form_submit_button("Generate RD.011 Document", type="primary")
            
            if submitted:
                if not mom_files:
                    st.error("Please upload at least one mandatory MoM document.")
                elif not os.environ.get("GROQ_API_KEY") and not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
                    st.error("Please provide API keys in the sidebar.")
                else:
                    # Check for duplicate filenames across all uploaders
                    all_files = list(mom_files or [])
                    if scope_file:
                        all_files.append(scope_file)
                    all_files.extend(q_files or [])
                    
                    seen_names = {}
                    duplicates = []
                    for f in all_files:
                        if f.name in seen_names:
                            if f.name not in duplicates:
                                duplicates.append(f.name)
                        else:
                            seen_names[f.name] = True
                    
                    if duplicates:
                        st.error(f"⚠️ Duplicate files detected: **{', '.join(duplicates)}**. Please remove duplicates before generating.")
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
    
    import threading
    import time
    
    # Initialize background job tracker if this is the first poll
    if "job_done" not in st.session_state:
        st.session_state.job_done = False
        st.session_state.job_result = None
        st.session_state.job_status = None
        st.session_state.job_error = None

        # CRITICAL FIX: Extract data from session state BEFORE spawning thread
        # Do NOT access st.session_state from within the worker thread
        thread_id = st.session_state.thread_id
        input_data = st.session_state.get("current_input")

        def _run_graph_in_background(thread_id, input_data):
            """Run the graph in a background thread without accessing st.session_state"""
            try:
                graph = get_graph()
                config_dict = {"configurable": {"thread_id": thread_id}}

                # IMPORTANT: Log all graph events for visibility into execution
                logger.info("Starting graph execution for thread_id=%s", thread_id)
                for event in graph.stream(input_data, config=config_dict, stream_mode="updates"):
                    logger.info("GRAPH EVENT: %s", str(event)[:1000])
                
                # Get the final state
                snapshot = graph.get_state(config_dict)
                st.session_state.job_result = snapshot.values
                
                # Determine post-run status
                if any(t.interrupts for t in snapshot.tasks):
                    st.session_state.job_status = "waiting_approval"
                else:
                    st.session_state.job_status = "completed"
                    
            except Exception as e:
                logger.error("Graph execution failed: %s", str(e), exc_info=True)
                st.session_state.job_error = str(e)
                st.session_state.job_status = "error"
            finally:
                st.session_state.job_done = True

        # Start the background thread with data passed as arguments (only once per session)
        threading.Thread(target=_run_graph_in_background, args=(thread_id, input_data), daemon=True).start()

    # Polling loop: check if the background job is done
    if not st.session_state.job_done:
        with st.status("Executing agentic workflow...", expanded=True) as status:
            st.info("⏳ Processing in the background. This page auto-refreshes every 5 seconds.")
            time.sleep(5)
        st.rerun()
    else:
        # Job is complete — propagate the result back to the main session state
        if st.session_state.job_status == "error":
            st.session_state.run_status = "error"
            st.session_state.error_message = st.session_state.job_error
        else:
            st.session_state.current_result = st.session_state.job_result
            st.session_state.run_status = st.session_state.job_status
        
        # Clean up background job flags to avoid stale state on the next run
        del st.session_state.job_done
        del st.session_state.job_result
        del st.session_state.job_status
        if "job_error" in st.session_state:
            del st.session_state.job_error
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
    
    # Key counter forces widget recreation (clears the box cleanly after submit)
    if "feedback_key_counter" not in st.session_state:
        st.session_state.feedback_key_counter = 0
    
    _feedback_key = f"feedback_box_{st.session_state.feedback_key_counter}"

    feedback = st.text_area(
        "Feedback (leave blank if approving):",
        key=_feedback_key,
        help="If the plan needs changes, write your revision instructions here.",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        approve_btn = st.button("✅ Approve Plan", type="primary")
    with col2:
        revise_btn = st.button("🔄 Submit Feedback for Revision")
        
    if approve_btn:
        _feedback_text = st.session_state.get(_feedback_key, "").strip()
        if _feedback_text:
            st.warning("⚠️ You have written feedback. Clear the feedback box before approving, or click 'Submit Feedback for Revision' instead.")
        else:
            from langgraph.types import Command
            st.session_state.current_input = Command(resume="APPROVE")
            st.session_state.run_status = "running"
            st.session_state.feedback_key_counter += 1
            st.rerun()
        
    elif revise_btn:
        _feedback_text = st.session_state.get(_feedback_key, "").strip()
        if not _feedback_text:
            st.warning("⚠️ Please write your feedback in the text box before submitting.")
        else:
            from langgraph.types import Command
            st.session_state.current_input = Command(resume=_feedback_text)
            st.session_state.run_status = "running"
            st.session_state.feedback_key_counter += 1
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
        