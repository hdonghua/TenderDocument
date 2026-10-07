from __future__ import annotations

from dataclasses import dataclass, field

from tender_agent.prompts import OUTLINE_SYSTEM, ProjectBrief, outline_user
from tender_agent.textutil import extract_json, infer_level, number_sort_key

MAX_SECTIONS = 40
MAX_LEAF_WORDS = 2500


@dataclass
class SectionSpec:
    number: str
    level: int
    title: str
    points: str = ""
    target_words: int = 0
    content: str = ""
    error: str = ""
    is_parent: bool = False
    child_titles: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class Outline:
    title: str
    sections: list[SectionSpec]
    source: str
    warning: str = ""


Node = tuple[str, str, tuple]


TECH_TREE: tuple[Node, ...] = (
    (
        "项目理解与需求分析",
        "说明对项目背景、目标和需求边界的理解",
        (
            ("项目背景与建设目标", "政策背景、现状问题、建设目标", ()),
            ("需求理解", "功能需求、性能与非功能需求", ()),
            ("重点难点与应对思路", "关键约束、风险和应对原则", ()),
        ),
    ),
    (
        "总体技术方案",
        "给出总体路线，不展开到实现细节",
        (
            ("设计原则与技术路线", "设计原则、总体路线", ()),
            ("总体架构", "逻辑架构、部署架构、数据流", ()),
            ("技术选型说明", "关键技术选择及理由，不编造未确定的产品型号", ()),
        ),
    ),
    (
        "详细实现方案",
        "按概要中的业务范围写实现思路",
        (
            ("业务功能方案", "覆盖概要中的主要功能", ()),
            ("数据、接口与集成", "数据、接口、与现有系统的集成", ()),
            ("部署与运行环境", "环境、部署步骤、运行维护", ()),
        ),
    ),
    (
        "项目实施计划",
        "实施组织与计划",
        (
            ("组织与分工", "项目组织、角色职责，人员姓名用占位符", ()),
            ("进度与里程碑", "阶段划分、交付物、验收衔接", ()),
            ("风险管理", "进度、需求、协作风险及措施", ()),
        ),
    ),
    (
        "质量、安全与服务",
        "质量、安全和售后承诺的总述",
        (
            ("质量管理", "过程质量控制、评审与测试", ()),
            ("安全保障", "安全制度、权限、数据和运维安全", ()),
            ("培训、验收与售后", "培训、验收配合、售后响应", ()),
        ),
    ),
)

BIZ_TREE: tuple[Node, ...] = (
    (
        "投标综述",
        "商务标开篇",
        (
            ("投标函要点", "投标意愿与有效期，不编造法定代表人信息", ()),
            ("对招标文件的响应说明", "实质性条款响应原则", ()),
        ),
    ),
    (
        "投标人概况与资质",
        "公司情况只写已知信息",
        (
            ("公司简介", "未知的注册信息、地址、人员规模用占位符", ()),
            ("资质与认证", "没有提供的证书一律用【待补充】，禁止编造编号", ()),
        ),
    ),
    (
        "类似业绩",
        "没有提供业绩时明确占位，禁止虚构合同",
        (
            ("业绩概述", "业绩名称、金额、时间未知时占位", ()),
            ("与本项目的相关性", "说明可借鉴经验，不张冠李戴", ()),
        ),
    ),
    (
        "商务响应与服务承诺",
        "商务条款与服务",
        (
            ("商务条款响应", "付款、工期、质保等，未知数字用占位符", ()),
            ("服务承诺与偏离说明", "服务级别与无偏离/有偏离的写法", ()),
        ),
    ),
)


def tree_for(bid_type: str) -> tuple[Node, ...]:
    if bid_type == "商务标":
        return BIZ_TREE
    if bid_type == "完整投标文件":
        return TECH_TREE + BIZ_TREE
    return TECH_TREE


def flatten_tree(nodes: tuple[Node, ...] | list[Node], prefix: str = "") -> list[SectionSpec]:
    sections: list[SectionSpec] = []
    for index, (title, points, children) in enumerate(nodes, start=1):
        number = str(index) if not prefix else f"{prefix}.{index}"
        sections.append(
            SectionSpec(
                number=number,
                level=infer_level(number),
                title=title,
                points=points,
            )
        )
        if children:
            sections.extend(flatten_tree(children, number))
    return sections


def fallback_outline(project: ProjectBrief) -> Outline:
    sections = flatten_tree(tree_for(project.bid_type))
    warning = allocate_words(sections, project.target_words)
    return Outline(
        title=f"{project.name}{project.bid_type}",
        sections=sections,
        source="fallback",
        warning=warning,
    )


def mark_parents(sections: list[SectionSpec]) -> None:
    for section in sections:
        prefix = f"{section.number}."
        parent_dots = section.number.count(".")
        children = [
            item
            for item in sections
            if item.number.startswith(prefix) and item.number.count(".") == parent_dots + 1
        ]
        section.is_parent = bool(children)
        section.child_titles = [item.title for item in children]


def allocate_words(sections: list[SectionSpec], target: int) -> str:
    """按目标总字数分配各节字数。模型给出的 target_words 只作为权重。"""
    if not sections:
        return ""
    target = max(int(target), 300)
    mark_parents(sections)
    parents = [item for item in sections if item.is_parent]
    leaves = [item for item in sections if not item.is_parent]
    if not leaves:
        leaves = list(sections)
        parents = []
        for item in leaves:
            item.is_parent = False
            item.child_titles = []

    parent_each = 220 if target < 15000 else 320
    leaf_floor = 300 if target < 12000 else 500
    for parent in parents:
        parent.target_words = parent_each
    budget = target - parent_each * len(parents)
    if budget < leaf_floor * len(leaves):
        parent_each = 0 if target < 200 * max(len(leaves), 1) else 160
        for parent in parents:
            parent.target_words = parent_each
        budget = max(target - parent_each * len(parents), len(leaves) * 180)
        leaf_floor = 180

    weights = [item.target_words if item.target_words > 0 else 1000 for item in leaves]
    weight_sum = sum(weights) or 1
    for item, weight in zip(leaves, weights):
        item.target_words = max(leaf_floor, int(round(budget * weight / weight_sum)))

    leaf_total = sum(item.target_words for item in leaves)
    if leaf_total > budget * 1.05 and leaf_total > 0:
        scale = budget / leaf_total
        for item in leaves:
            item.target_words = max(180, int(item.target_words * scale))

    warning = ""
    overflow = 0
    adjustable: list[SectionSpec] = []
    for item in leaves:
        if item.target_words > MAX_LEAF_WORDS:
            overflow += item.target_words - MAX_LEAF_WORDS
            item.target_words = MAX_LEAF_WORDS
        else:
            adjustable.append(item)
    if overflow and adjustable:
        capacity = sum(MAX_LEAF_WORDS - item.target_words for item in adjustable)
        if capacity > 0:
            give = min(overflow, capacity)
            rooms = [MAX_LEAF_WORDS - item.target_words for item in adjustable]
            room_sum = sum(rooms) or 1
            for item, room in zip(adjustable, rooms):
                item.target_words += int(give * room / room_sum)
            overflow = max(0, overflow - give)
    if overflow > 0:
        warning = (
            f"目标字数较高，单节上限 {MAX_LEAF_WORDS} 字，仍有约 {overflow} 字未能分配。"
            "请在大纲中增加小节后再生成。"
        )

    if not warning and leaves:
        drift = target - sum(item.target_words for item in sections)
        if drift:
            leaves[-1].target_words = max(180, leaves[-1].target_words + drift)
            if leaves[-1].target_words > MAX_LEAF_WORDS:
                leftover = leaves[-1].target_words - MAX_LEAF_WORDS
                leaves[-1].target_words = MAX_LEAF_WORDS
                warning = (
                    f"目标字数较高，单节上限 {MAX_LEAF_WORDS} 字，仍有约 {leftover} 字未能分配。"
                    "请在大纲中增加小节后再生成。"
                )
    return warning


def _pick(data: dict, *keys, default=None):
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def sections_from_json(data: dict) -> tuple[str, list[SectionSpec]]:
    title = str(_pick(data, "title", "标题", default="投标文件")).strip() or "投标文件"
    raw_sections = _pick(data, "sections", "章节", default=[])
    if not isinstance(raw_sections, list):
        raise ValueError("大纲缺少 sections 数组")
    sections: list[SectionSpec] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(raw_sections, start=1):
        if not isinstance(item, dict):
            continue
        title_text = str(_pick(item, "title", "标题", default="")).strip()
        if not title_text:
            continue
        number = str(_pick(item, "number", "编号", default=str(index))).strip() or str(index)
        if number in seen:
            seen[number] += 1
            number = f"{number}-{seen[number]}"
        else:
            seen[number] = 1
        level_raw = _pick(item, "level", "层级", default=1)
        try:
            words = int(_pick(item, "target_words", "目标字数", default=0) or 0)
        except (TypeError, ValueError):
            words = 0
        sections.append(
            SectionSpec(
                number=number,
                level=infer_level(number, level_raw),
                title=title_text,
                points=str(_pick(item, "points", "要点", default="") or "").strip(),
                target_words=max(0, words),
            )
        )
    if not sections:
        raise ValueError("大纲里没有可用章节")
    sections.sort(key=lambda item: number_sort_key(item.number))
    return title, sections


def parse_outline(text: str, project: ProjectBrief) -> Outline:
    title, sections = sections_from_json(extract_json(text))
    warning = ""
    if len(sections) > MAX_SECTIONS:
        sections = sections[:MAX_SECTIONS]
        warning = f"大纲超过 {MAX_SECTIONS} 节，已只保留前 {MAX_SECTIONS} 节。"
    alloc_warning = allocate_words(sections, project.target_words)
    warning = " ".join(part for part in (warning, alloc_warning) if part)
    return Outline(title=title, sections=sections, source="model", warning=warning)


def generate_outline(project: ProjectBrief, client, references: str = "") -> Outline:
    reminder = ""
    last_error: Exception | None = None
    for _ in range(2):
        try:
            result = client.complete(
                OUTLINE_SYSTEM,
                outline_user(project, references, reminder),
                json_mode=True,
            )
            if not result.text:
                raise ValueError("模型返回空大纲")
            return parse_outline(result.text, project)
        except Exception as exc:
            last_error = exc
            reminder = "上一次输出无法解析。请只输出 JSON 对象，sections 为非空数组。"
    fallback = fallback_outline(project)
    fallback.warning = f"大纲生成失败，已改用内置模板。原因：{last_error}"
    return fallback
