"""Drive the real Streamlit script with AppTest: upload, approve, choose model, report."""

from pathlib import Path

import pytest
import streamlit as st
from conftest import make_messy_frame
from streamlit.testing.v1 import AppTest

from autoanalyst.config import get_settings
from autoanalyst.orchestrator import Job, Stage
from refdata import DATA

APP = str(Path(__file__).parents[1] / "streamlit_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    env = {
        "AUTOANALYST_RUNS_DIR": str(tmp_path / "runs"),
        "AUTOANALYST_LLM_PROVIDER": "mock",
        "AUTOANALYST_INLINE_JOBS": "true",
        "AUTOANALYST_CONVERT_PDF": "false",
        "AUTOANALYST_CV_FOLDS": "3",
        "AUTOANALYST_SEARCH_ITER": "2",
        "AUTOANALYST_N_JOBS": "1",
        "AUTOANALYST_IMPORTANCE_REPEATS": "2",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    yield at
    get_settings.cache_clear()
    st.cache_resource.clear()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _job(at) -> Job:
    return Job.open(at.session_state["job_id"], get_settings())


def test_empty_page(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "AutoAnalyst"
    assert "waiting for a CSV" in app.caption[0].value
    assert len(app.file_uploader) == 1


def test_rejected_upload_shows_errors(app):
    app.run()
    app.file_uploader[0].upload("tiny.csv", b"a,b\n1,2\n3,4\n", "text/csv").run()
    assert not app.exception
    assert any("at least 20 are needed" in e.value for e in app.error)
    assert "job_id" not in app.session_state


def test_four_steps_end_to_end(app):
    app.run()
    app.file_uploader[0].upload("messy.csv", make_messy_frame().to_csv(index=False).encode(), "text/csv").run()
    assert not app.exception
    assert "205 rows" in app.subheader[0].value
    assert app.selectbox[0].value == "churn"
    assert any("random train/test split" in w.value for w in app.warning)

    _button(app, "Plan preprocessing").click().run()
    assert not app.exception
    assert _job(app).state.stage == Stage.AWAITING_PLAN
    assert "Approve plan" in [b.label for b in app.button]
    assert any("Critic remarks" in e.label for e in app.expander)

    _button(app, "Approve plan").click().run()
    assert not app.exception
    assert _job(app).state.stage == Stage.AWAITING_MODEL
    assert app.radio[0].value == "classification"
    assert app.selectbox[1].value == "hist_gradient_boosting"

    app.selectbox[1].set_value("decision_tree").run()
    _button(app, "Train and write report").click().run()
    assert not app.exception
    job = _job(app)
    assert job.state.stage == Stage.DONE, job.state.error
    assert "Report ready" in app.caption[0].value
    assert any("Decision Tree: test macro F1" in m.value for m in app.markdown)
    downloads = [e for e in app.get("download_button")]
    assert len(downloads) == 1  # DOCX only: PDF conversion is off in this test
    assert "Start a new analysis" in [b.label for b in app.button]


def test_reject_with_comment_replans(app):
    app.run()
    app.file_uploader[0].upload("iris.csv", (DATA / "iris.csv").read_bytes(), "text/csv").run()
    _button(app, "Plan preprocessing").click().run()
    app.text_input[0].input("Do not drop duplicate rows").run()
    _button(app, "Reject plan").click().run()
    assert not app.exception
    job = _job(app)
    assert job.state.stage == Stage.AWAITING_PLAN
    assert any(e.decision == "Plan rejected" and "duplicate" in e.reason for e in job.state.decision_log)


def test_status_line_and_error_area_show_job_errors(app):
    app.run()
    app.file_uploader[0].upload("iris.csv", (DATA / "iris.csv").read_bytes(), "text/csv").run()
    job = _job(app)
    job.state.stage, job.state.error = Stage.FAILED, "Something broke"
    job.save()
    app.run()
    assert any("Something broke" in e.value for e in app.error)
    assert "failed" in app.caption[0].value


def test_interrupted_step_offers_retry(app):
    app.run()
    app.file_uploader[0].upload("iris.csv", (DATA / "iris.csv").read_bytes(), "text/csv").run()
    job = _job(app)
    job.state.stage = Stage.PLANNING
    job.save()
    app.run()
    assert any("interrupted" in w.value for w in app.warning)
    _button(app, "Retry step").click().run()
    assert _job(app).state.stage == Stage.AWAITING_PLAN
