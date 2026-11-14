import shutil
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/lib/libreoffice/program/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
]


class PdfConversionError(RuntimeError):
    pass


def find_soffice(configured: str | None = None) -> str | None:
    if configured:
        return configured if Path(configured).exists() else None
    found = shutil.which("soffice")
    if found:
        return found
    return next((c for c in CANDIDATES if Path(c).exists()), None)


def docx_to_pdf(docx: Path, out_dir: Path | None = None, soffice: str | None = None,
                timeout: int = 180) -> Path:
    """Convert with LibreOffice headless so the preview is the same document as the download."""
    soffice = find_soffice(soffice)
    if soffice is None:
        raise PdfConversionError("LibreOffice (soffice) not found; set AUTOANALYST_SOFFICE_PATH")
    docx = Path(docx).resolve()
    out_dir = Path(out_dir or docx.parent).resolve()
    pdf = out_dir / f"{docx.stem}.pdf"
    pdf.unlink(missing_ok=True)
    # A private profile avoids clashing with a LibreOffice window the user has open.
    with tempfile.TemporaryDirectory(prefix="aa_lo_") as profile:
        cmd = [
            soffice, f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise PdfConversionError(f"LibreOffice timed out after {timeout}s") from exc
    if not pdf.exists():
        raise PdfConversionError(f"LibreOffice produced no PDF: {proc.stderr.strip() or proc.stdout.strip()}")
    return pdf
