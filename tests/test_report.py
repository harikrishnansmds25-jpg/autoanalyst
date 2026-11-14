import json

import pytest
from docx import Document
from docx.shared import RGBColor
from pypdf import PdfReader

from autoanalyst.agents import fallback_plan
from autoanalyst.config import Settings
from autoanalyst.eda import run_eda
from autoanalyst.ingest import load_csv
from autoanalyst.profiling import build_profile
from autoanalyst.report import (
    Facts,
    Narrative,
    build_facts,
    build_report,
    template_narrative,
    unsupported_numbers,
)
from autoanalyst.report.pdf import find_soffice
from autoanalyst.schemas import DecisionEntry
from autoanalyst.toolbox.executor import execute
from autoanalyst.training import train_and_evaluate
from autoanalyst.validators import validate_execution
from autoanalyst.viz import make_charts
from refdata import DATA

FAST = Settings(_env_file=None, cv_folds=3, search_iter=2, n_jobs=1, importance_repeats=2)
HEADINGS = [
    "1. Overview and data quality", "2. Decision log", "3. Exploratory analysis",
    "4. Model and evaluation", "5. Feature importance", "6. Warnings and limitations",
    "7. Conclusion", "Appendix: run manifest",
]
needs_soffice = pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")


def _pipeline(name, target, task, model, out_dir, max_rows=None):
    data = load_csv(DATA / f"{name}.csv", settings=FAST)
    if max_rows:
        data.df = data.df.sample(max_rows, random_state=0).reset_index(drop=True)
    profile = build_profile(data, target=target, settings=FAST)
    plan = fallback_plan(profile, FAST)
    result = execute(data.df, plan)
    report = validate_execution(data.df, plan, result, FAST)
    eda = run_eda(result.df, target, task, FAST)
    out = train_and_evaluate(result.df, target, task, model, result.steps_by_feature(), FAST)
    log = [DecisionEntry(stage="plan", actor="fallback", decision="Rule-based plan", reason="test")]
    facts = build_facts(profile, plan, report, len(result.df), eda, out.result, log, {"job_id": "test"})
    charts = make_charts(result.df, profile, eda, out, out_dir / "charts")
    return facts, charts


@pytest.fixture(scope="module")
def titanic(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("titanic")
    facts, charts = _pipeline("titanic", "Survived", "classification", "random_forest", out_dir)
    return facts, charts, out_dir


@pytest.fixture(scope="module")
def california(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("california")
    facts, charts = _pipeline("california_housing", "MedHouseVal", "regression", "ridge", out_dir, 2000)
    return facts, charts, out_dir


def test_facts_roundtrip_and_summary(titanic):
    facts, _, _ = titanic
    assert Facts.model_validate_json(facts.model_dump_json()) == facts
    assert facts.dataset.n_rows == 891 and facts.dataset.n_features_clean == 7
    assert facts.summary.n_dropped_columns == 4
    assert facts.summary.n_checks == facts.summary.n_checks_passed
    assert facts.run["libraries"]["scikit-learn"]


@pytest.mark.parametrize("case", ["titanic", "california"])
def test_template_narrative_uses_only_fact_numbers(case, request):
    facts, _, _ = request.getfixturevalue(case)
    narrative = template_narrative(facts)
    for section, text in narrative.model_dump().items():
        assert text, section
        assert unsupported_numbers(text, facts) == [], (section, text)


def test_narrative_wording(titanic):
    narrative = template_narrative(titanic[0])
    assert "Cramer's V" in narrative.eda
    assert "1 warning in" in narrative.conclusion or "warnings in" in narrative.conclusion
    assert "{" not in " ".join(narrative.model_dump().values())


def test_charts_for_each_task(titanic, california):
    keys_c = {c.key for c in titanic[1]}
    keys_r = {c.key for c in california[1]}
    assert {"missing_values", "confusion_matrix", "feature_importance", "model_vs_baseline"} <= keys_c
    assert {"predicted_vs_actual", "residuals", "correlation_heatmap"} <= keys_r
    assert "missing_values" not in keys_r
    for chart in titanic[1] + california[1]:
        assert chart.path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_docx_structure_and_plain_style(titanic):
    facts, charts, out_dir = titanic
    files = build_report(facts, template_narrative(facts), charts, out_dir / "r1", convert_pdf=False)
    doc = Document(files.docx)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading 1")]
    assert headings == HEADINGS
    assert len(doc.inline_shapes) == len(charts)
    assert all(t.style.name == "Table Grid" for t in doc.tables)
    for name in ("Title", "Heading 1", "Heading 2", "Normal"):
        assert doc.styles[name].font.color.rgb == RGBColor(0, 0, 0)
    assert json.loads(files.facts_json.read_text())["dataset"]["n_rows"] == 891
    assert files.removed_sentences == {}


def test_invented_numbers_are_stripped_before_the_docx(titanic):
    facts, charts, out_dir = titanic
    narrative = template_narrative(facts).model_copy(
        update={"model": "The model is excellent. It reaches 99.9% accuracy on new data."}
    )
    files = build_report(facts, narrative, charts, out_dir / "r2", convert_pdf=False)
    assert files.removed_sentences == {"model": ["It reaches 99.9% accuracy on new data."]}
    text = "\n".join(p.text for p in Document(files.docx).paragraphs)
    assert "99.9" not in text and "The model is excellent." in text


def test_fully_unsupported_section_gets_placeholder(titanic):
    facts, charts, out_dir = titanic
    narrative = Narrative(overview="It has 123456 rows.", data_quality="x", eda="x", model="x",
                          conclusion="x")
    files = build_report(facts, narrative, charts, out_dir / "r3", convert_pdf=False)
    text = "\n".join(p.text for p in Document(files.docx).paragraphs)
    assert "123456" not in text and "No verified statement" in text


def test_missing_soffice_is_reported_not_raised(titanic):
    facts, charts, out_dir = titanic
    settings = FAST.model_copy(update={"soffice_path": str(out_dir / "nope.exe")})
    files = build_report(facts, template_narrative(facts), charts, out_dir / "r4", settings=settings)
    assert files.pdf is None and "not found" in files.pdf_error


@needs_soffice
@pytest.mark.parametrize("case", ["titanic", "california"])
def test_pdf_matches_docx(case, request):
    facts, charts, out_dir = request.getfixturevalue(case)
    files = build_report(facts, template_narrative(facts), charts, out_dir / "pdf", settings=FAST)
    assert files.pdf_error is None and files.pdf.exists()
    reader = PdfReader(files.pdf)
    text = " ".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) >= 3
    for heading in HEADINGS:
        assert heading in text
    assert facts.dataset.data_hash[:16] in text.replace("\n", "")
