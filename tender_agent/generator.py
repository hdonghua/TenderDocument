from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tender_agent.kb import KnowledgeBase, format_hits
from tender_agent.outline import Outline, SectionSpec, generate_outline
from tender_agent.prompts import SECTION_SYSTEM, ProjectBrief, section_user
from tender_agent.textutil import char_count, number_sort_key, strip_leading_heading


@dataclass
class GenerationResult:
    markdown: str
    outline: Outline
    errors: list[str]


def prepare_outline(project: ProjectBrief, client, kb: KnowledgeBase | None, use_kb: bool) -> Outline:
    references = ""
    if use_kb and kb is not None:
        query = "\n".join(
            part for part in (project.bid_type, project.industry, project.name, project.summary[:800]) if part
        )
        try:
            references = format_hits(kb.search(query, k=6), limit=3000)
        except Exception:
            references = ""
    return generate_outline(project, client, references)


def _retrieve(section: SectionSpec, project: ProjectBrief, kb: KnowledgeBase | None, use_kb: bool) -> str:
    if not use_kb or kb is None:
        section.references = []
        return ""
    query = "\n".join(
        part
        for part in (
            section.title,
            section.points,
            project.industry,
            project.name,
            project.summary[:400],
        )
        if part
    )
    try:
        hits = kb.search(query, k=4)
    except Exception:
        hits = []
    section.references = [f"{hit.source}（相关度 {hit.score:.2f}）" for hit in hits]
    return format_hits(hits, limit=1800)


def _write_section(section: SectionSpec, project: ProjectBrief, client, references: str) -> None:
    last_error: Exception | None = None
    text = ""
    truncated = False
    for attempt in range(2):
        prompt = section_user(
            project,
            number=section.number,
            title=section.title,
            points=section.points,
            target_words=max(int(section.target_words or 800), 150),
            is_parent=section.is_parent,
            child_titles=section.child_titles,
            references=references,
            tighten=attempt > 0,
        )
        try:
            result = client.complete(SECTION_SYSTEM, prompt)
            text = strip_leading_heading(result.text, section.title)
            truncated = result.truncated
            if not text:
                raise RuntimeError("模型返回空内容")
            minimum = max(120, int(section.target_words * 0.35))
            if char_count(text) < minimum and attempt == 0:
                continue
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            text = ""
    if not text:
        raise RuntimeError(str(last_error) if last_error else "生成失败")
    section.content = text
    section.error = ""
    if truncated:
        section.error = "内容可能被 max_tokens 截断。调高该模型的 max_tokens 后，可只重写此节。"


def fill_sections(
    outline: Outline,
    project: ProjectBrief,
    client,
    kb: KnowledgeBase | None,
    *,
    use_kb: bool,
    workers: int,
    only_missing: bool,
    on_progress,
) -> list[str]:
    from tender_agent.outline import mark_parents

    mark_parents(outline.sections)
    targets = []
    for section in outline.sections:
        if only_missing and section.content.strip() and not section.error:
            continue
        targets.append(section)
    if not targets:
        return []

    errors: list[str] = []
    total = len(targets)
    done = 0

    def job(section: SectionSpec) -> SectionSpec:
        references = _retrieve(section, project, kb, use_kb)
        _write_section(section, project, client, references)
        return section

    worker_count = max(1, min(int(workers), len(targets)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(job, section): section for section in targets}
        for future in as_completed(futures):
            section = futures[future]
            done += 1
            try:
                future.result()
                message = f"已完成 {section.number} {section.title}"
            except Exception as exc:
                section.content = f"【生成失败】{exc}"
                section.error = str(exc)
                errors.append(f"{section.number} {section.title}：{exc}")
                message = f"失败 {section.number} {section.title}"
            if on_progress:
                on_progress(done, total, message)
    outline.sections.sort(key=lambda item: number_sort_key(item.number))
    return errors


def assemble_markdown(outline: Outline) -> str:
    lines = [f"# {outline.title}", "", "## 目录", ""]
    for section in outline.sections:
        indent = "  " * max(section.level - 1, 0)
        lines.append(f"{indent}- {section.number} {section.title}")
    lines.extend(["", "---", ""])
    for section in outline.sections:
        hashes = "#" * min(section.level + 1, 4)
        lines.append(f"{hashes} {section.number} {section.title}")
        lines.append("")
        lines.append((section.content or "【待生成】").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def generate_document(
    outline: Outline,
    project: ProjectBrief,
    client,
    kb: KnowledgeBase | None,
    *,
    use_kb: bool,
    workers: int,
    only_missing: bool,
    on_progress,
) -> GenerationResult:
    errors = fill_sections(
        outline,
        project,
        client,
        kb,
        use_kb=use_kb,
        workers=workers,
        only_missing=only_missing,
        on_progress=on_progress,
    )
    return GenerationResult(markdown=assemble_markdown(outline), outline=outline, errors=errors)
