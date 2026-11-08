from pathlib import Path
import unittest

from lowparllm.ai.rag.chunking import chunk_document
from lowparllm.ai.rag.ingestion.base import DocumentPage, ExtractedDocument


class ChunkingTests(unittest.TestCase):
    def test_small_document_is_one_chunk_with_hash_and_metadata(self) -> None:
        document = ExtractedDocument(
            Path("notes.md"),
            (DocumentPage("# Heading\n\nA short paragraph.", page_number=3),),
        )
        chunks = chunk_document(document, chunk_size=20, chunk_overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 3)
        self.assertEqual(chunks[0].section_heading, "Heading")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(len(chunks[0].content_hash), 64)

    def test_large_text_chunks_preserve_paragraph_boundary_and_overlap(self) -> None:
        paragraphs = [f"paragraph{index} " + "word " * 4 for index in range(8)]
        document = ExtractedDocument(
            Path("notes.txt"), (DocumentPage("\n\n".join(paragraphs)),)
        )
        chunks = chunk_document(document, chunk_size=10, chunk_overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertIn("paragraph0", chunks[0].text)
        self.assertTrue(
            any(
                f"paragraph{index}" in chunks[position - 1].text
                and f"paragraph{index}" in chunks[position].text
                for position in range(1, len(chunks))
                for index in range(8)
            )
        )

    def test_long_paragraph_prefers_sentence_boundaries(self) -> None:
        document = ExtractedDocument(
            Path("notes.txt"),
            (DocumentPage("One two three. Four five six. Seven eight nine."),),
        )
        chunks = chunk_document(document, chunk_size=4, chunk_overlap=0)
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["One two three.", "Four five six.", "Seven eight nine."],
        )

    def test_code_chunking_keeps_indentation_and_line_ranges(self) -> None:
        document = ExtractedDocument(
            Path("module.py"),
            (DocumentPage("def greet():\n    print('hello')\n\n    return 1"),),
        )
        chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)
        self.assertIn("    print('hello')", "\n".join(chunk.text for chunk in chunks))
        self.assertEqual(chunks[0].start_line, 1)
        self.assertTrue(all(chunk.start_line <= chunk.end_line for chunk in chunks))

    def test_pdf_pages_are_not_merged_into_one_citation(self) -> None:
        document = ExtractedDocument(
            Path("book.pdf"),
            (
                DocumentPage("First page text.", page_number=1),
                DocumentPage("Second page text.", page_number=2),
            ),
        )
        chunks = chunk_document(document, chunk_size=50, chunk_overlap=5)
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 2])

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(
                ExtractedDocument(Path("a.txt"), (DocumentPage("some text"),)),
                chunk_size=10,
                chunk_overlap=10,
            )


if __name__ == "__main__":
    unittest.main()
