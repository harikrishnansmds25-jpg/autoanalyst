"""Assemble the DOCX report (plain Normal template, black and white) and convert it to PDF."""

from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

from autoanalyst.config import Settings, get_settings
from autoanalyst.report.facts import Facts
from autoanalyst.report.narrative import SECTIONS, Narrative
from autoanalyst.report.number_check import strip_unsupported
from autoanalyst.report.pdf import PdfConversionError, docx_to_pdf
from autoanalyst.training.trainer import METRIC_NAMES
from autoanalyst.viz.plots import Chart

BLACK = RGBColor(0, 0, 0)
FONT = "Calibri"
LIMITATIONS = [
    "Rows were split into training and test sets at random. If rows depend on time or come in "
    "groups, the test score may be optimistic.",
    "Results describe associations in this dataset, not cause and effect.",
    "Permutation importance is shared between correlated features, so each one can look less "
    "important than it is.",
    "Only the cleaning steps listed in the decision log were applied; no other feature engineering was done.",
]


@dataclass
class ReportFiles:
    docx: Path
    pdf: Path | None
    facts_json: Path
    removed_sentences: dict[str, list[str]] = field(default_factory=dict)
    pdf_error: str | None = None


def _plain_styles(doc) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = False
        style.font.color.rgb = BLACK
        if name == "Title":
            # The default Title style has a coloured bottom border; remove it.
            ppr = style.element.get_or_add_pPr()
            for border in ppr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pBdr"):
                ppr.remove(border)


def _table(doc, headers: list[str], rows: list[list]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = ""
        cell.paragraphs[0].add_run(str(text)).bold = True
    # Repeat the header row when a table runs onto the next page.
    header_props = table.rows[0]._tr.get_or_add_trPr()
    header_props.append(OxmlElement("w:tblHeader"))
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()


class _Figures:
    def __init__(self, doc, charts: list[Chart]):
        self.doc = doc
        self.by_key = {c.key: c for c in charts}
        self.n = 0

    def add(self, key: str, width: float = 6.0) -> None:
        chart = self.by_key.get(key)
        if chart is None or not Path(chart.path).exists():
            return
        self.n += 1
        self.doc.add_picture(str(chart.path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = self.doc.add_paragraph(f"Figure {self.n}. {chart.caption}")
        caption.runs[0].font.size = Pt(9)
        caption.runs[0].italic = True


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)


def checked_narrative(narrative: Narrative, facts: Facts) -> tuple[Narrative, dict[str, list[str]]]:
    """Remove sentences with numbers that are not in facts. Returns the clean narrative and removals."""
    clean, removed = {}, {}
    for name in SECTIONS:
        text, dropped = strip_unsupported(getattr(narrative, name), facts)
        clean[name] = text or "No verified statement is available for this section."
        if dropped:
            removed[name] = dropped
    return Narrative(**clean), removed


def build_docx(facts: Facts, narrative: Narrative, charts: list[Chart], path: Path) -> Path:
    doc = Document()
    _plain_styles(doc)
    figures = _Figures(doc, charts)
    d, m = facts.dataset, facts.model

    doc.add_heading(f"Analysis report: {d.file_name}", level=0)
    doc.add_paragraph(f"Generated {facts.run['generated_at']}. Target: {d.target}. Task: {d.task}.")

    doc.add_heading("1. Overview and data quality", level=1)
    doc.add_paragraph(narrative.overview)
    _table(doc, ["Item", "Value"], [
        ["Rows", f"{d.n_rows:,}"], ["Columns", d.n_cols], ["Duplicate rows", d.n_duplicates],
        ["Rows after cleaning", f"{d.n_rows_clean:,}"], ["Features used", d.n_features_clean],
    ])
    doc.add_paragraph(narrative.data_quality)
    if facts.column_issues:
        _table(doc, ["Column", "Type", "Missing", "Flags"], [
            [c.column, c.kind, f"{c.n_missing} ({100 * c.missing_frac:.1f}%)", ", ".join(c.flags) or "-"]
            for c in facts.column_issues
        ])
    figures.add("missing_values", 5.5)

    doc.add_heading("2. Decision log", level=1)
    if facts.decision_log:
        _table(doc, ["Stage", "By", "Decision", "Reason"],
               [[e.stage, e.actor, e.decision, e.reason] for e in facts.decision_log])
    doc.add_heading("Cleaning plan", level=2)
    doc.add_paragraph(f"Plan source: {facts.plan.source}.")
    _table(doc, ["Column", "Action", "Reason"],
           [[s.column or "(all rows)", s.action, s.reason] for s in facts.plan.steps])
    doc.add_heading("Validation checks", level=2)
    _table(doc, ["Check", "Result", "Detail"], [
        [c.name, "pass" if c.passed else ("fail" if c.severity == "error" else c.severity), c.message]
        for c in facts.validation
    ])

    doc.add_heading("3. Exploratory analysis", level=1)
    doc.add_paragraph(narrative.eda)
    figures.add("target_distribution", 5.5)
    figures.add("numeric_distributions", 6.0)
    figures.add("correlation_heatmap", 5.0)
    if facts.eda.associations:
        _table(doc, ["Feature", "Measure", "Association with target"],
               [[a.feature, a.measure, f"{a.value:.3f}"] for a in facts.eda.associations[:10]])

    doc.add_heading("4. Model and evaluation", level=1)
    doc.add_paragraph(narrative.model)
    rows = []
    for key in m.cv:
        label = METRIC_NAMES.get(key, key)
        rows.append([label, f"{m.cv[key].mean:.3f} ± {m.cv[key].std:.3f}", f"{m.test.get(key, float('nan')):.3f}",
                     f"{m.cv_baseline[key].mean:.3f}", f"{m.test_baseline.get(key, float('nan')):.3f}"])
    _table(doc, ["Metric", "Model CV (mean ± std)", "Model test", "Baseline CV", "Baseline test"], rows)
    doc.add_paragraph(
        f"Model: {m.model_label}. Training rows: {m.n_train:,}. Test rows: {m.n_test:,}. "
        f"Best parameters: " + ", ".join(f"{k} = {v}" for k, v in m.best_params.items()) + "."
    )
    figures.add("model_vs_baseline", 4.5)
    figures.add("confusion_matrix", 4.0)
    figures.add("predicted_vs_actual", 4.0)
    figures.add("residuals", 5.0)

    doc.add_heading("5. Feature importance", level=1)
    figures.add("feature_importance", 5.5)
    _table(doc, ["Feature", "Importance", "Std"],
           [[f.feature, f"{f.importance:.4f}", f"{f.std:.4f}"] for f in m.feature_importance[:10]])

    doc.add_heading("6. Warnings and limitations", level=1)
    for text in facts.warnings + LIMITATIONS:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_heading("7. Conclusion", level=1)
    doc.add_paragraph(narrative.conclusion)

    doc.add_heading("Appendix: run manifest", level=1)
    manifest = [["Data hash (SHA-256)", d.data_hash]]
    for key, value in facts.run.items():
        if isinstance(value, dict):
            manifest += [[f"{key}: {k}", _fmt(v)] for k, v in value.items()]
        else:
            manifest.append([key, _fmt(value)])
    _table(doc, ["Item", "Value"], manifest)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


def build_report(facts: Facts, narrative: Narrative, charts: list[Chart], out_dir: str | Path,
                 convert_pdf: bool = True, settings: Settings | None = None) -> ReportFiles:
    """Write facts.json, report.docx and (if LibreOffice is available) report.pdf."""
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts_json = out_dir / "facts.json"
    facts_json.write_text(facts.model_dump_json(indent=2), encoding="utf-8")

    narrative, removed = checked_narrative(narrative, facts)
    docx = build_docx(facts, narrative, charts, out_dir / "report.docx")
    files = ReportFiles(docx=docx, pdf=None, facts_json=facts_json, removed_sentences=removed)
    if convert_pdf:
        try:
            files.pdf = docx_to_pdf(docx, out_dir, settings.soffice_path, settings.pdf_timeout)
        except PdfConversionError as exc:
            files.pdf_error = str(exc)
    return files
