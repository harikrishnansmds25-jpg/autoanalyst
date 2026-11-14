"""Streamlit UI: upload, plan approval, model selection, report.

    streamlit run streamlit_app.py
"""

import time

import pandas as pd
import streamlit as st

from autoanalyst.config import get_settings
from autoanalyst.ingest import IngestError
from autoanalyst.models import DEFAULT_MODEL, models_for
from autoanalyst.orchestrator import InvalidAction, Job, JobBusy, JobRunner, PlanNotValid, Stage
from autoanalyst.orchestrator.jobs import cleanup_runs
from autoanalyst.orchestrator.state import WORKING
from autoanalyst.toolbox.registry import ALL_ACTIONS
from autoanalyst.training.trainer import METRIC_NAMES
from autoanalyst.ui.plan_table import ALL_ROWS, frame_plan, plan_frame

STEPS = ["1. Upload", "2. Approve plan", "3. Choose model", "4. Report"]
STEP_OF = {
    Stage.UPLOADED: 0, Stage.PLANNING: 1, Stage.AWAITING_PLAN: 1, Stage.EXECUTING: 1,
    Stage.AWAITING_MODEL: 2, Stage.TRAINING: 3, Stage.REPORTING: 3, Stage.DONE: 3, Stage.FAILED: 3,
}
RETRY_STEP = {Stage.PLANNING: "plan", Stage.EXECUTING: "execute", Stage.TRAINING: "train_and_report"}
POLL_SECONDS = 1.5


@st.cache_resource
def get_runner() -> JobRunner:
    settings = get_settings()
    cleanup_runs(settings.runs_dir, settings.retention_days)
    return JobRunner(settings, inline=settings.inline_jobs)


def current_job() -> Job | None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return None
    try:
        return Job.open(job_id, get_settings())
    except FileNotFoundError:
        st.session_state.pop("job_id", None)
        return None


def run_step(job: Job, step: str, **kwargs) -> None:
    try:
        get_runner().submit(job.state.job_id, step, **kwargs)
    except JobBusy as exc:
        st.session_state["ui_error"] = str(exc)
    st.rerun()


def header(job: Job | None) -> None:
    st.title("AutoAnalyst")
    active = STEP_OF[job.state.stage] if job else 0
    st.markdown("  ·  ".join(f"**{s}**" if i == active else s for i, s in enumerate(STEPS)))
    if job:
        st.caption(f"Status: {job.state.stage.value.replace('_', ' ')}. {job.state.message}")
    else:
        st.caption("Status: waiting for a CSV file.")
    if job and job.state.error:
        st.error(job.state.error)
    if error := st.session_state.pop("ui_error", None):
        st.error(error)


def wait_while_busy(job: Job) -> bool:
    """Show progress and poll while a background step runs. Returns True if the page should stop here."""
    if get_runner().is_busy(job.state.job_id):
        st.info(job.state.message or "Working...")
        time.sleep(POLL_SECONDS)
        st.rerun()
        return True
    if job.state.stage in WORKING:
        st.warning("This step was interrupted (for example by a restart).")
        if st.button("Retry step"):
            if job.state.stage == Stage.REPORTING:
                job.state.stage = Stage.TRAINING
                job.save()
            run_step(job, RETRY_STEP.get(job.state.stage, "train_and_report"))
        return True
    return False


def upload_step(job: Job | None) -> None:
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None and st.session_state.get("upload_id") != upload.file_id:
        st.session_state["upload_id"] = upload.file_id
        try:
            job = Job.create(upload.getvalue(), upload.name, get_settings())
        except IngestError as exc:
            st.session_state.pop("job_id", None)
            st.error("The file cannot be used:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
            return
        st.session_state["job_id"] = job.state.job_id
        st.rerun()
    if job is None:
        return

    profile = job.profile()
    st.subheader(f"{job.state.file_name}: {job.state.n_rows:,} rows, {job.state.n_cols} columns")
    st.dataframe(job.data().df.head(10), width="stretch")
    st.dataframe(
        pd.DataFrame([
            {"column": c.name, "type": c.kind, "dtype": c.dtype, "missing": c.n_missing,
             "unique": c.n_unique, "flags": ", ".join(c.flags)}
            for c in profile.columns
        ]),
        hide_index=True, width="stretch",
    )
    for warning in profile.warnings:
        st.warning(warning)

    columns = [c.name for c in profile.columns]
    target = st.selectbox("Target column", columns, index=columns.index(job.state.target))
    if st.button("Plan preprocessing", type="primary"):
        job.set_target(target)
        run_step(job, "plan")


def approval_step(job: Job) -> None:
    if job.state.execution_explanation:
        st.error(job.state.execution_explanation)
    if job.state.open_issues:
        st.warning("Open issues from the debate:\n\n" + "\n".join(f"- {i}" for i in job.state.open_issues))
    if job.state.critic_remarks:
        with st.expander("Critic remarks", expanded=True):
            for remark in job.state.critic_remarks:
                st.write(f"- {remark}")

    proposed = job.proposed_plan()
    columns = [c.name for c in job.profile().columns]
    edited = st.data_editor(
        plan_frame(proposed),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"plan_editor_{job.state.job_id}_{len(job.state.decision_log)}",
        column_config={
            "column": st.column_config.SelectboxColumn("Column", options=[ALL_ROWS] + columns, required=True),
            "action": st.column_config.SelectboxColumn("Action", options=sorted(ALL_ACTIONS), required=True),
            "reason": st.column_config.TextColumn("Reason", width="large"),
            "factor": st.column_config.NumberColumn("Factor", min_value=1.0, max_value=10.0),
            "dayfirst": st.column_config.CheckboxColumn("Day first"),
        },
    )
    plan, problem = frame_plan(edited, job.state.target)
    if problem:
        st.error(f"The edited plan is not valid: {problem}")
        return
    checks = [c for c in job.check(plan) if not c.passed]
    for check in checks:
        (st.error if check.severity == "error" else st.warning)(f"{check.name}: {check.message}")
    has_errors = any(c.severity == "error" for c in checks)

    left, right = st.columns(2)
    with left:
        if st.button("Approve plan", type="primary", disabled=has_errors):
            try:
                job.approve_plan(plan)
            except (PlanNotValid, InvalidAction) as exc:
                st.session_state["ui_error"] = str(exc)
                st.rerun()
            run_step(job, "execute")
    with right:
        comment = st.text_input("Reason for rejecting (sent to the Planner)")
        if st.button("Reject plan"):
            feedback = job.reject_plan(comment)
            run_step(job, "plan", feedback=feedback)


def model_step(job: Job) -> None:
    report = job.validation()
    eda = job.eda()
    if report:
        st.dataframe(
            pd.DataFrame([{"check": c.name, "result": "pass" if c.passed else c.severity, "detail": c.message}
                          for c in report.checks]),
            hide_index=True, width="stretch",
        )
    if eda:
        for warning in eda.warnings:
            st.warning(warning)

    columns = [c.name for c in job.profile().columns]
    target = st.selectbox("Target column", columns, index=columns.index(job.state.target))
    if target != job.state.target:
        st.info("Changing the target means a new cleaning plan, which you will approve again.")
    tasks = ["classification", "regression"]
    default_task = job.state.task or job.state.suggested_task or "classification"
    task = st.radio("Task type (confirm)", tasks, index=tasks.index(default_task), horizontal=True)
    models = models_for(task)
    keys = list(models)
    default_key = job.state.model_key if job.state.model_key in models else DEFAULT_MODEL[task]
    model_key = st.selectbox("Model", keys, index=keys.index(default_key),
                             format_func=lambda k: models[k].label, key=f"model_{task}")
    if st.button("Train and write report", type="primary"):
        next_step = job.select_model(target, task, model_key)
        run_step(job, "plan" if next_step == "replan" else "train_and_report")


def report_step(job: Job) -> None:
    result = job.train_result()
    if result:
        metric = result.primary_metric
        st.write(
            f"{result.model_label}: test {METRIC_NAMES[metric]} {result.test[metric]:.3f}, "
            f"baseline {result.test_baseline[metric]:.3f}. "
            + ("Beats the baseline." if result.beats_baseline else "Does not clearly beat the baseline.")
        )
    pdf = job.report_path("pdf")
    docx = job.report_path("docx")
    left, right = st.columns(2)
    if docx:
        left.download_button("Download DOCX", docx.read_bytes(), file_name=f"{job.state.job_id}.docx",
                             mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if pdf:
        right.download_button("Download PDF", pdf.read_bytes(), file_name=f"{job.state.job_id}.pdf",
                              mime="application/pdf")
        st.pdf(pdf.read_bytes(), height=900)
    elif docx:
        st.info("No PDF preview is available; download the DOCX.")


def main() -> None:
    st.set_page_config(page_title="AutoAnalyst", layout="wide")
    job = current_job()
    header(job)
    if job and wait_while_busy(job):
        return

    stage = job.state.stage if job else Stage.UPLOADED
    if stage == Stage.UPLOADED:
        upload_step(job)
    elif stage == Stage.AWAITING_PLAN:
        approval_step(job)
    elif stage == Stage.AWAITING_MODEL:
        model_step(job)
    elif stage == Stage.DONE:
        report_step(job)

    if job and stage != Stage.UPLOADED:
        st.divider()
        if st.button("Start a new analysis"):
            for key in ("job_id", "upload_id"):
                st.session_state.pop(key, None)
            st.rerun()


if __name__ == "__main__":
    main()
