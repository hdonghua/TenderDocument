from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from tender_agent.config import (
    KNOWLEDGE_DIR,
    OUTPUT_DIR,
    PROVIDERS,
    ModelProfile,
    ensure_dirs,
    has_api_key,
    load_settings,
    load_secrets,
    resolve_api_key,
    save_secrets,
    save_settings,
    unique_id,
)
from tender_agent.export import export_docx, export_markdown, make_stem
from tender_agent.generator import assemble_markdown, generate_document, prepare_outline
from tender_agent.kb import SUPPORTED_SUFFIXES, KnowledgeBase
from tender_agent.llm import build_client
from tender_agent.outline import Outline, SectionSpec, mark_parents
from tender_agent.prompts import ProjectBrief
from tender_agent.textutil import infer_level, number_sort_key, safe_filename

ensure_dirs()
KB = KnowledgeBase(KNOWLEDGE_DIR)

st.set_page_config(page_title="标书生成智能体", layout="wide")


def _ensure_index() -> None:
    if st.session_state.get("use_kb") and not KB.ready() and KB.list_sources():
        KB.build()


def _profile_map() -> tuple[list[ModelProfile], str, dict[str, ModelProfile]]:
    profiles, default_id = load_settings()
    return profiles, default_id, {item.id: item for item in profiles}


def _prefer(profiles: list[ModelProfile], *ids: str) -> str:
    existing = {item.id for item in profiles}
    for profile_id in ids:
        if profile_id in existing:
            return profile_id
    return profiles[0].id


def _project_from_inputs() -> ProjectBrief:
    return ProjectBrief(
        name=st.session_state.get("project_name", "").strip(),
        purchaser=st.session_state.get("purchaser", ""),
        bidder=st.session_state.get("bidder", ""),
        industry=st.session_state.get("industry", ""),
        bid_type=st.session_state.get("bid_type", "技术标"),
        summary=st.session_state.get("summary", ""),
        scoring=st.session_state.get("scoring", ""),
        extra=st.session_state.get("extra", ""),
        target_words=int(st.session_state.get("target_words", 12000)),
    )


def _outline_frame(outline: Outline) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "编号": section.number,
                "层级": int(section.level),
                "标题": section.title,
                "要点": section.points,
                "目标字数": int(section.target_words),
            }
            for section in outline.sections
        ]
    )


def _sections_from_frame(frame: pd.DataFrame, previous: list[SectionSpec]) -> list[SectionSpec]:
    saved = {(section.number, section.title): section for section in previous}
    sections: list[SectionSpec] = []
    for _, row in frame.iterrows():
        title = str(row.get("标题") or "").strip()
        if not title or title.lower() == "nan":
            continue
        number = str(row.get("编号") or "").strip()
        if number.endswith(".0"):
            number = number[:-2]
        if not number or number.lower() == "nan":
            number = str(len(sections) + 1)
        try:
            level = int(row.get("层级") or 1)
        except (TypeError, ValueError):
            level = 1
        try:
            words = int(float(row.get("目标字数") or 800))
        except (TypeError, ValueError):
            words = 800
        points = str(row.get("要点") or "")
        if points.lower() == "nan":
            points = ""
        section = SectionSpec(
            number=number,
            level=infer_level(number, level),
            title=title,
            points=points.strip(),
            target_words=max(150, words),
        )
        old = saved.get((section.number, section.title))
        if old:
            section.content = old.content
            section.error = old.error
            section.references = list(old.references)
        sections.append(section)
    sections.sort(key=lambda item: number_sort_key(item.number))
    mark_parents(sections)
    return sections


def page_generate() -> None:
    st.title("标书生成")
    st.caption("先出大纲，再按小节并发生成。每一节单独调用模型，避免整篇标书超出上下文。")

    profiles, _, profile_map = _profile_map()
    if not profiles:
        st.error("还没有模型配置。请先打开「模型配置」。")
        return

    labels = {item.id: f"{item.label}  ·  {item.model}" for item in profiles}
    left, right = st.columns(2)
    with left:
        st.text_input("项目名称", key="project_name", placeholder="例如：市级政务服务平台建设项目")
        st.text_input("招标人", key="purchaser")
        st.text_input("投标人", key="bidder", help="不知道就留空，正文会写成【待补充】，不会编造公司信息。")
        st.text_input("行业 / 项目类型", key="industry", placeholder="例如：政务信息化")
    with right:
        st.selectbox("标书类型", ["技术标", "商务标", "完整投标文件"], key="bid_type")
        st.slider("全文目标字数", min_value=3000, max_value=50000, step=1000, value=12000, key="target_words")
        outline_id = st.selectbox(
            "大纲模型",
            options=[item.id for item in profiles],
            index=[item.id for item in profiles].index(_prefer(profiles, "deepseek-reasoner", "deepseek-chat")),
            format_func=lambda item: labels[item],
            key="outline_model",
        )
        body_id = st.selectbox(
            "正文模型",
            options=[item.id for item in profiles],
            index=[item.id for item in profiles].index(_prefer(profiles, "deepseek-chat", "deepseek-reasoner")),
            format_func=lambda item: labels[item],
            key="body_model",
        )

    st.text_area("项目概要", height=180, key="summary", placeholder="建设背景、范围、周期、必须响应的功能。越具体，标书越不容易写成空话。")
    st.text_area("招标要求 / 评分办法摘录", height=140, key="scoring", placeholder="可选。粘贴评分项后，大纲会尽量把得分点拆进对应小节。")
    st.text_area("补充要求", height=80, key="extra", placeholder="可选。例如：强调国产化、等保、与现有系统对接。")

    option_col, worker_col = st.columns(2)
    with option_col:
        st.checkbox("参考知识库中的历史标书", value=True, key="use_kb")
        st.radio("生成范围", ["全部章节", "仅空白或失败章节"], horizontal=True, key="gen_scope")
    with worker_col:
        st.number_input("并发生成数", min_value=1, max_value=8, value=4, step=1, key="workers")
        body_profile = profile_map[st.session_state.get("body_model", body_id)]
        if body_profile.provider == "cursor":
            st.caption("Cursor 每写一节都会启动一次本地 Agent，耗时更长。章节多时把并发降到 1。")
        elif not KB.ready() and st.session_state.get("use_kb", True):
            st.caption("知识库还没有索引。可以先生成，或到「知识库」页放入历史标书后重建索引。")

    if st.button("生成大纲", type="primary"):
        project = _project_from_inputs()
        if not project.name or not project.summary:
            st.error("请先填写项目名称和项目概要。")
        else:
            try:
                client = build_client(profile_map[outline_id])
                with st.spinner("正在生成大纲…"):
                    _ensure_index()
                    outline = prepare_outline(project, client, KB, bool(st.session_state.get("use_kb")))
                st.session_state.outline = outline
                st.session_state.outline_rev = st.session_state.get("outline_rev", 0) + 1
                st.session_state.pop("markdown", None)
            except Exception as exc:
                st.error(str(exc))

    outline: Outline | None = st.session_state.get("outline")
    edited = None
    if outline is not None:
        if outline.warning:
            st.warning(outline.warning)
        source = "模型生成" if outline.source == "model" else "内置模板"
        st.subheader(f"大纲 · {outline.title}")
        st.caption(f"来源：{source}。可以直接改标题、要点和字数，也可以增删行。有子节的章只会写短导语。")
        edited = st.data_editor(
            _outline_frame(outline),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"outline_editor_{st.session_state.get('outline_rev', 0)}",
            column_config={
                "编号": st.column_config.TextColumn(required=True, width="small"),
                "层级": st.column_config.SelectboxColumn("层级", options=[1, 2, 3], required=True, width="small"),
                "标题": st.column_config.TextColumn(required=True, width="medium"),
                "要点": st.column_config.TextColumn(width="large"),
                "目标字数": st.column_config.NumberColumn(min_value=150, max_value=4000, step=50, width="small"),
            },
        )
        planned = int(pd.to_numeric(edited["目标字数"], errors="coerce").fillna(0).sum()) if len(edited) else 0
        st.caption(f"当前大纲合计约 {planned} 字，共 {len(edited)} 节。单节建议不超过 2000 字，更长的内容请拆开。")

    if st.button("根据大纲生成正文", type="primary", disabled=outline is None):
        project = _project_from_inputs()
        if outline is None or edited is None:
            st.error("请先生成大纲。")
        elif not project.name or not project.summary:
            st.error("请先填写项目名称和项目概要。")
        else:
            sections = _sections_from_frame(edited, outline.sections)
            if not sections:
                st.error("大纲至少需要一个标题。")
            else:
                outline.title = outline.title or f"{project.name}{project.bid_type}"
                outline.sections = sections
                progress = st.progress(0, text="准备生成…")
                log = st.empty()
                messages: list[str] = []

                def on_progress(done: int, total: int, message: str) -> None:
                    messages.append(message)
                    progress.progress(done / total, text=f"{done}/{total}  {message}")
                    log.caption("\n".join(messages[-8:]))

                try:
                    client = build_client(profile_map[st.session_state.get("body_model", body_id)])
                    _ensure_index()
                    result = generate_document(
                        outline,
                        project,
                        client,
                        KB,
                        use_kb=bool(st.session_state.get("use_kb")),
                        workers=int(st.session_state.get("workers", 4)),
                        only_missing=st.session_state.get("gen_scope") == "仅空白或失败章节",
                        on_progress=on_progress,
                    )
                    stem = make_stem(project.name)
                    md_path = export_markdown(result.markdown, OUTPUT_DIR, stem)
                    docx_path = export_docx(result.outline, OUTPUT_DIR, stem)
                    st.session_state.outline = result.outline
                    st.session_state.markdown = result.markdown
                    st.session_state.exports = (str(md_path), str(docx_path))
                    st.session_state.gen_errors = result.errors
                    progress.progress(1.0, text="生成结束")
                except Exception as exc:
                    st.error(str(exc))

    markdown = st.session_state.get("markdown")
    if markdown:
        errors = st.session_state.get("gen_errors") or []
        if errors:
            st.error("有章节生成失败，可把生成范围改成「仅空白或失败章节」后再次生成。\n\n" + "\n".join(errors))
        export_paths = st.session_state.get("exports")
        if export_paths:
            md_path, docx_path = (Path(export_paths[0]), Path(export_paths[1]))
            download_md, download_docx = st.columns(2)
            with download_md:
                st.download_button(
                    "下载 Markdown",
                    data=md_path.read_bytes(),
                    file_name=md_path.name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with download_docx:
                st.download_button(
                    "下载 Word",
                    data=docx_path.read_bytes(),
                    file_name=docx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            st.caption(f"已保存到 {md_path.parent}")
        preview, source, status = st.tabs(["预览", "Markdown", "章节状态"])
        with preview:
            st.markdown(markdown)
        with source:
            st.text_area("Markdown 原文", markdown, height=480)
        with status:
            rows = []
            for section in outline.sections if outline else []:
                rows.append(
                    {
                        "编号": section.number,
                        "标题": section.title,
                        "状态": "失败" if section.error else ("已写" if section.content else "空白"),
                        "约字数": len("".join(section.content.split())),
                        "说明": section.error,
                        "参考片段": "；".join(section.references),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_knowledge() -> None:
    st.title("知识库")
    st.write(
        "把脱敏后的历史标书放进来，生成时按小节检索相近片段，只借鉴写法和结构。"
        "证书编号、报价、人员信息不要放进知识库。"
    )
    uploaded = st.file_uploader(
        "上传历史标书",
        type=["md", "txt", "docx", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("保存到知识库", type="primary"):
        saved = 0
        for item in uploaded:
            suffix = Path(item.name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                st.warning(f"已跳过不支持的文件：{item.name}")
                continue
            target = KB.raw_dir / safe_filename(item.name)
            target.write_bytes(item.getbuffer())
            saved += 1
        st.success(f"已保存 {saved} 个文件。需要重建索引后才会被检索。")

    sources = KB.list_sources()
    st.subheader(f"资料目录（{len(sources)} 个文件）")
    st.caption(str(KB.raw_dir))
    if not sources:
        st.info("目录是空的。仓库里带了一份示例技术方案，若列表为空请确认 knowledge/raw 是否被移走。")
    else:
        for path in sources:
            label = path.relative_to(KB.raw_dir).as_posix()
            column, action = st.columns([8, 1])
            column.write(label)
            if action.button("删除", key=f"delete-{label}"):
                path.unlink(missing_ok=True)
                st.rerun()

    if st.button("重建索引", type="primary"):
        with st.spinner("正在分词并建立索引，首次会稍慢…"):
            report = KB.build()
        st.success(f"已处理 {report.files} 个文件，生成 {report.chunks} 个片段。")
        for warning in report.warnings:
            st.warning(warning)

    if KB.ready():
        st.caption("索引已就绪。")
    else:
        st.caption("还没有索引。")

    query = st.text_input("试检索", placeholder="例如：办件流转 对象存储")
    if query:
        hits = KB.search(query, k=5)
        if not hits:
            st.info("没有匹配片段。若刚放入文件，请先重建索引。")
        for hit in hits:
            with st.expander(f"{hit.source}  ·  相关度 {hit.score:.2f}"):
                st.write(hit.text)


def _empty_profile(provider: str = "deepseek") -> ModelProfile:
    if provider == "cursor":
        return ModelProfile(
            id="",
            label="",
            provider="cursor",
            model="composer-2.5",
            api_key_env="CURSOR_API_KEY",
            temperature=0.3,
            max_tokens=4096,
        )
    if provider == "openai_compat":
        return ModelProfile(
            id="",
            label="",
            provider="openai_compat",
            model="",
            base_url="",
            api_key_env="DEEPSEEK_API_KEY",
            temperature=0.5,
            max_tokens=8192,
        )
    return ModelProfile(
        id="",
        label="",
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.5,
        max_tokens=8192,
    )


def page_models() -> None:
    st.title("模型配置")
    st.write("可以添加多个 DeepSeek 配置，分别指定不同模型；也可以添加 Cursor，或任意 OpenAI 兼容接口。")
    profiles, default_id, _ = _profile_map()
    if not profiles:
        st.warning("配置文件是空的，保存一条后会写回 config/models.yaml。")

    rows = []
    for profile in profiles:
        rows.append(
            {
                "名称": profile.label,
                "提供商": PROVIDERS.get(profile.provider, profile.provider),
                "模型": profile.model,
                "密钥": "已配置" if has_api_key(profile) else "未配置",
                "默认": "是" if profile.id == default_id else "",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    choices = ["新建配置"] + [f"{item.label}（{item.id}）" for item in profiles]
    selected = st.selectbox("编辑", choices)
    current = _empty_profile()
    if selected != "新建配置":
        current = profiles[choices.index(selected) - 1]

    with st.form(f"model_form_{current.id or 'new'}"):
        label = st.text_input("显示名称", value=current.label, placeholder="例如：DeepSeek Chat 账号二")
        provider = st.selectbox(
            "提供商",
            list(PROVIDERS.keys()),
            index=list(PROVIDERS.keys()).index(current.provider if current.provider in PROVIDERS else "deepseek"),
            format_func=lambda item: PROVIDERS[item],
        )
        model = st.text_input("模型名", value=current.model, placeholder="deepseek-chat / deepseek-reasoner / composer-2.5")
        base_url = st.text_input(
            "Base URL",
            value=current.base_url,
            help="DeepSeek 默认 https://api.deepseek.com。Cursor 不使用这个地址。OpenAI 兼容接口填到 /v1 或服务商文档要求的前缀。",
        )
        api_key_env = st.text_input(
            "环境变量名（可选）",
            value=current.api_key_env,
            help="留空密钥时，会按这个变量名从环境或 .env 读取。",
        )
        api_key = st.text_input("API Key", type="password", placeholder="留空表示不修改已保存的密钥")
        clear_key = st.checkbox("清除已单独保存的密钥")
        temperature = st.number_input("温度", min_value=0.0, max_value=2.0, value=float(current.temperature), step=0.1)
        max_tokens = st.number_input("max_tokens", min_value=256, max_value=32768, value=int(current.max_tokens), step=256)
        make_default = st.checkbox("设为默认模型", value=current.id == default_id or selected == "新建配置")
        save = st.form_submit_button("保存", type="primary")

    if save:
        if not label.strip() or not model.strip():
            st.error("显示名称和模型名都不能为空。")
        elif provider == "openai_compat" and not base_url.strip():
            st.error("OpenAI 兼容接口必须填写 Base URL。")
        else:
            profile_id = current.id or unique_id(label, {item.id for item in profiles})
            profile = ModelProfile(
                id=profile_id,
                label=label.strip(),
                provider=provider,
                model=model.strip(),
                base_url=base_url.strip(),
                api_key_env=api_key_env.strip(),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            updated = [item for item in profiles if item.id != profile_id]
            updated.append(profile)
            new_default = profile_id if make_default else default_id
            save_settings(updated, new_default)
            secrets = load_secrets()
            if clear_key:
                secrets.pop(profile_id, None)
            if api_key.strip():
                secrets[profile_id] = api_key.strip()
            save_secrets(secrets)
            st.success("已保存。密钥写在本机 config/secrets.yaml，不会放进模型清单。")
            st.rerun()

    if current.id:
        column_test, column_delete = st.columns(2)
        with column_test:
            if st.button("测试连接"):
                try:
                    client = build_client(current)
                    with st.spinner("正在调用模型…"):
                        result = client.complete("你是连接测试助手。", "只回复 OK")
                    if result.text:
                        st.success(f"连接成功：{result.text[:200]}")
                    else:
                        st.warning("调用完成，但没有返回文本。")
                except Exception as exc:
                    st.error(str(exc))
        with column_delete:
            if st.button("删除此配置"):
                remaining = [item for item in profiles if item.id != current.id]
                if not remaining:
                    st.error("至少保留一个模型配置。")
                else:
                    save_settings(remaining, default_id if default_id != current.id else remaining[0].id)
                    secrets = load_secrets()
                    secrets.pop(current.id, None)
                    save_secrets(secrets)
                    st.rerun()

    st.info(
        "同一把 DeepSeek Key 可以配多条记录，例如 deepseek-chat 写正文、deepseek-reasoner 写大纲。"
        "换账号时新建一条并单独填写 API Key。Cursor 需要先执行 pip install -r requirements-cursor.txt。"
    )


def page_help() -> None:
    st.title("使用说明")
    st.markdown(
        """
### 生成一篇长标书

1. 在「模型配置」里确认 DeepSeek 或 Cursor 的密钥可用，点一次「测试连接」。
2. 在「知识库」放入脱敏后的历史标书，点「重建索引」。没有历史标书也可以直接生成。
3. 在「生成标书」填写项目名称和概要。评分办法建议一并贴上。
4. 先「生成大纲」，检查章节和字数，再「根据大纲生成正文」。
5. 某一节失败时，把生成范围改成「仅空白或失败章节」再跑一次。

正文按小节并发生成。每节只携带项目概要、该节要点和少量历史片段，所以全文可以到数万字，而不会把整篇标书塞进一次上下文。

### 知识库里放什么

支持 Markdown、TXT、DOCX、PDF。扫描版 PDF 提取不到文字，需要先做文字识别。旧版 `.doc` 请另存为 `.docx`。

放进来之前删掉报价、身份证号、手机号、未公开的客户信息和证书编号。生成结果里的未知事实会写成【待补充】，程序不会替你编造资质和业绩。

更完整的搭建步骤写在项目根目录的 `README.md`。
        """
    )


pages = {
    "生成标书": page_generate,
    "知识库": page_knowledge,
    "模型配置": page_models,
    "使用说明": page_help,
}
choice = st.sidebar.radio("功能", list(pages))
st.sidebar.caption("密钥只保存在本机。未知的资质、报价和人员信息会留成【待补充】。")
pages[choice]()
