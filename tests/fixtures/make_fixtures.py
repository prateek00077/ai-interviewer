"""Regenerate the resume fixtures.

The generated .pdf and .docx are committed, so the test suite needs neither
reportlab nor a network. Run this only when the fixture content should change:

    pip install reportlab && python tests/fixtures/make_fixtures.py

The text is deliberately shaped like a real CV -- recognisable section headings,
dated entries, a contact line -- because that is exactly what the parser's
sectioning and the chunker's entry splitting are being asked to handle.
"""

import pathlib

OUT = pathlib.Path(__file__).parent

RESUME_TEXT = """Priya Raman
priya.raman@example.com | +91 98765 43210 | https://github.com/priyaraman

Summary
Backend engineer with eight years building distributed systems.

Experience
2021 - 2024  Senior Engineer, Northwind Systems
Owned the payments ledger service handling 12k requests per second.
Migrated the event bus from RabbitMQ to Kafka with zero downtime.
Mentored four engineers through their first on-call rotations.

2018 - 2021  Engineer, Contoso Cloud
Built the multi-tenant billing pipeline on Postgres and Airflow.
Reduced month-end close from six hours to eleven minutes.

Education
2014 - 2018  B.Tech Computer Science, IIT Madras

Skills
Python, Go, Postgres, Kafka, Kubernetes, Terraform, gRPC
"""


def write_pdf() -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(OUT / "resume.pdf"), pagesize=LETTER)
    y = 750
    for line in RESUME_TEXT.split("\n"):
        c.drawString(60, y, line)
        y -= 14
    c.save()


def write_docx() -> None:
    import docx

    document = docx.Document()
    for line in RESUME_TEXT.split("\n"):
        document.add_paragraph(line)
    document.save(OUT / "resume.docx")


if __name__ == "__main__":
    write_pdf()
    write_docx()
    print(f"fixtures written to {OUT}")
