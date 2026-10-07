from __future__ import annotations

from tender_agent.textutil import clip


class ProjectBrief:
    def __init__(
        self,
        name: str,
        purchaser: str = "",
        bidder: str = "",
        industry: str = "",
        bid_type: str = "技术标",
        summary: str = "",
        scoring: str = "",
        extra: str = "",
        target_words: int = 12000,
    ):
        self.name = name.strip()
        self.purchaser = purchaser.strip()
        self.bidder = bidder.strip()
        self.industry = industry.strip()
        self.bid_type = bid_type.strip() or "技术标"
        self.summary = summary.strip()
        self.scoring = scoring.strip()
        self.extra = extra.strip()
        self.target_words = int(target_words)


def project_block(project: ProjectBrief, *, summary_limit: int, scoring_limit: int) -> str:
    bidder = project.bidder or "【待补充：投标人名称】"
    purchaser = project.purchaser or "【待补充：招标人】"
    lines = [
        f"项目名称：{project.name}",
        f"招标人：{purchaser}",
        f"投标人：{bidder}",
        f"行业/类型：{project.industry or '未填写'}",
        f"标书类型：{project.bid_type}",
        f"全文目标字数：约 {project.target_words} 字",
        "",
        "项目概要：",
        clip(project.summary, summary_limit) or "（未填写）",
    ]
    if project.scoring:
        lines.extend(["", "招标要求 / 评分办法摘录：", clip(project.scoring, scoring_limit)])
    if project.extra:
        lines.extend(["", "补充要求：", clip(project.extra, 2000)])
    return "\n".join(lines)


OUTLINE_SYSTEM = """你是资深投标文件编制专家。根据项目概要编写标书大纲。
只输出一个 JSON 对象，不要输出解释，不要使用 Markdown 代码块。
JSON 结构如下：
{
  "title": "投标文件标题",
  "sections": [
    {
      "number": "1.1",
      "level": 2,
      "title": "小节标题",
      "points": "这一节必须覆盖的要点，用一句话写清",
      "target_words": 1200
    }
  ]
}
要求：
- 章节要针对本项目，不要用放之四海的空标题。
- 同时给出章（level 1）和可直接撰写的小节（level 2 或 3）。
- 有子节的章只写导语，target_words 设为 200 到 400。
- 叶子小节 target_words 一般 600 到 2000，全部 target_words 之和接近全文目标字数。
- 小节总数不要超过 30 个。
- 不得编造资质、业绩金额、人员身份和报价。未知事项留在要点里提示用【待补充】占位。
"""


def outline_user(project: ProjectBrief, references: str, reminder: str = "") -> str:
    parts = [
        project_block(project, summary_limit=6000, scoring_limit=4000),
        "",
        "请结合上述项目编写大纲。评分办法里的得分点要在对应小节的要点中体现。",
    ]
    if references:
        parts.extend(
            [
                "",
                "以下历史标书片段只用来参考结构与写法，禁止照搬其中的项目名称、金额、人员和证书编号：",
                references,
            ]
        )
    if reminder:
        parts.extend(["", reminder])
    return "\n".join(parts)


SECTION_SYSTEM = """你是资深投标文件编制专家，正在撰写标书中的一个小节。
写作要求：
- 使用正式、具体的中文，紧扣本项目概要，避免空泛套话。
- 直接输出正文，不要复述小节标题，不要写“以下是正文”之类的说明。
- 需要分层时只用 Markdown 三级及以下标题（###、####）。
- 不得编造资质证书编号、合同金额、人员身份证号、联系方式、奖项和报价。
- 材料里没有的事实写成【待补充：需要补充的内容】。
- 历史标书只可借鉴表述方式，不得把其他项目的名称和数据写进本文。
- 字数接近要求即可，写完即止，不要为了凑字数重复。
"""


def section_user(
    project: ProjectBrief,
    *,
    number: str,
    title: str,
    points: str,
    target_words: int,
    is_parent: bool,
    child_titles: list[str],
    references: str,
    tighten: bool,
) -> str:
    role = "这是章首导语。" if is_parent else "这是需要展开撰写的正文小节。"
    parts = [
        project_block(project, summary_limit=2500, scoring_limit=2000),
        "",
        f"当前小节：{number} {title}",
        role,
        f"目标字数：约 {target_words} 字。",
        f"必须覆盖的要点：{points or '结合项目概要展开'}",
    ]
    if is_parent and child_titles:
        joined = "、".join(child_titles)
        parts.append(f"下属小节将另行撰写，导语只做总述，不要展开这些小节的细节：{joined}")
    if references:
        parts.extend(
            [
                "",
                "历史标书参考片段（仅借鉴写法，禁止照搬事实）：",
                references,
            ]
        )
    if tighten:
        parts.append(
            f"\n上一稿明显偏短。请重新撰写，篇幅接近 {target_words} 字，并写清可检查的做法、步骤和响应点。"
        )
    return "\n".join(parts)
