"""Structured result type shared by all techniques.

Each technique's ``run`` returns a ``TechniqueResult`` so the CLI and the
Streamlit UI can render the same data in different ways.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    content: str
    label: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class TechniqueResult:
    title: str
    question: str | None = None
    answer: str | None = None
    # (heading, body) intermediate steps, e.g. sub-questions, decisions.
    steps: list[tuple[str, str]] = field(default_factory=list)
    documents: list[RetrievedDoc] = field(default_factory=list)

    def add_step(self, heading: str, body: object) -> None:
        self.steps.append((heading, str(body)))

    def add_doc(self, content: str, label: str = "", meta: dict | None = None) -> None:
        self.documents.append(RetrievedDoc(content=content, label=label, meta=meta or {}))

    def to_text(self) -> str:
        """Plain-text rendering used by the CLI."""
        lines: list[str] = []
        bar = "=" * len(self.title)
        lines.append(f"\n{bar}\n{self.title}\n{bar}")
        if self.question:
            lines.append(f"Question: {self.question}\n")

        for heading, body in self.steps:
            lines.append(f"--- {heading} ---\n{body}\n")

        if self.documents:
            lines.append(f"Retrieved documents ({len(self.documents)}):")
            for i, doc in enumerate(self.documents, start=1):
                label = f" ({doc.label})" if doc.label else ""
                preview = doc.content[:200].replace("\n", " ")
                lines.append(f"[{i}]{label} {preview}...")
            lines.append("")

        if self.answer is not None:
            lines.append(f"Answer:\n{self.answer}")

        return "\n".join(lines)


def print_result(result: TechniqueResult) -> None:
    print(result.to_text())
