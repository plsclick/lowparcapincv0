from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from lowparllm.ai.rag.ingestion.base import (
    DocumentProcessingError,
    extract_document,
)
from lowparllm.ai.rag.ingestion.directory_loader import discover_documents


def _pdf_bytes(text: str | None, password: str | None = None) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text is not None:
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font}
                )
            }
        )
        page[NameObject("/Resources")] = resources
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class IngestionTests(unittest.TestCase):
    def test_plain_text_markdown_and_source_code_preserve_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "notes.md": "# Heading\n\nText with paragraphs.\n",
                "source.py": "def f():\n    return 1\n",
                "data.json": '{\n  "key": "value"\n}\n',
            }
            for name, content in files.items():
                path = root / name
                path.write_text(content, encoding="utf-8")
                document = extract_document(path)
                extracted = "\n".join(page.text for page in document.pages)
                self.assertIn(content.strip().splitlines()[0], extracted)
                self.assertEqual(document.file_size, len(content.encode()))
                self.assertEqual(len(document.content_hash), 64)

    def test_html_script_content_is_not_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "page.html"
            path.write_text(
                "<h1>Title</h1><script>do not include</script><p>Visible text</p>",
                encoding="utf-8",
            )
            text = extract_document(path).pages[0].text
        self.assertIn("Visible text", text)
        self.assertNotIn("do not include", text)

    def test_pdf_page_extraction_and_scanned_pdf_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            text_path = Path(temporary) / "text.pdf"
            scan_path = Path(temporary) / "scan.pdf"
            text_path.write_bytes(_pdf_bytes("Hello PDF"))
            scan_path.write_bytes(_pdf_bytes(None))
            parsed = extract_document(text_path)
            self.assertEqual(parsed.pages[0].page_number, 1)
            self.assertIn("Hello PDF", parsed.pages[0].text)
            with self.assertRaisesRegex(DocumentProcessingError, "OCR"):
                extract_document(scan_path)

    def test_encrypted_pdf_and_docx_tables_and_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            encrypted = Path(temporary) / "locked.pdf"
            encrypted.write_bytes(_pdf_bytes("Secret", "password"))
            with self.assertRaisesRegex(DocumentProcessingError, "encrypted"):
                extract_document(encrypted)

            document = Document()
            document.add_heading("Study Notes", level=1)
            document.add_paragraph("Paragraph body")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Term"
            table.cell(0, 1).text = "Definition"
            path = Path(temporary) / "notes.docx"
            document.save(path)
            extracted = extract_document(path)
        text = extracted.pages[0].text
        self.assertIn("# Study Notes", text)
        self.assertIn("Paragraph body", text)
        self.assertIn("Term | Definition", text)

    def test_unsupported_empty_binary_and_size_limit_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsupported = root / "file.exe"
            unsupported.write_bytes(b"binary")
            empty = root / "empty.txt"
            empty.write_bytes(b"")
            binary = root / "binary.txt"
            binary.write_bytes(b"text\x00binary")
            large = root / "large.txt"
            large.write_bytes(b"x" * (1024 * 1024 + 1))
            for path, error in (
                (unsupported, "Unsupported"),
                (empty, "empty"),
                (binary, "Binary"),
                (large, "limit"),
            ):
                with self.subTest(path=path.name), self.assertRaisesRegex(
                    DocumentProcessingError, error
                ):
                    extract_document(path, max_file_size_mb=1 if path == large else 50)

    def test_directory_scan_skips_hidden_generated_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.txt").write_text("ok", encoding="utf-8")
            (root / ".hidden").mkdir()
            (root / ".hidden/secret.txt").write_text("skip", encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv/package.py").write_text("skip", encoding="utf-8")
            (root / "unsupported.bin").write_bytes(b"")
            (root / "nested").mkdir()
            (root / "nested/guide.md").write_text("ok", encoding="utf-8")
            files = discover_documents(root)
            shallow = discover_documents(root, recursive=False)
        self.assertEqual({path.name for path in files}, {"notes.txt", "guide.md"})
        self.assertEqual([path.name for path in shallow], ["notes.txt"])

    def test_directory_scan_respects_gitignore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("private/\nignored.txt\n", encoding="utf-8")
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")
            (root / "credentials.txt").write_text("not for indexing", encoding="utf-8")
            (root / "secrets").mkdir()
            (root / "secrets" / "private.txt").write_text("not for indexing", encoding="utf-8")
            (root / "private").mkdir()
            (root / "private" / "secret.txt").write_text("ignored", encoding="utf-8")

            files = discover_documents(root)

        self.assertEqual([path.name for path in files], ["visible.txt"])

    def test_directory_scan_rejects_lowp_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owned_data = root / "lowp-data"
            owned_data.mkdir()
            with patch(
                "lowparllm.ai.rag.ingestion.directory_loader.data_dir",
                return_value=owned_data,
            ):
                with self.assertRaisesRegex(ValueError, "application-data"):
                    discover_documents(owned_data)


if __name__ == "__main__":
    unittest.main()
