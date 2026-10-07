import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tender_agent.export import export_docx, export_markdown
from tender_agent.generator import assemble_markdown
from tender_agent.kb import KnowledgeBase, chunk_text
from tender_agent.llm import CursorClient, OpenAICompatClient
from tender_agent.outline import SectionSpec, fallback_outline, parse_outline
from tender_agent.prompts import ProjectBrief
from tender_agent.textutil import char_count, extract_json, strip_leading_heading


class TextUtilTests(unittest.TestCase):
    def test_extract_json_from_fence(self):
        text = """说明文字
```json
{"title": "示例标", "sections": [{"number": "1", "title": "背景"}]}
```
"""
        data = extract_json(text)
        self.assertEqual(data["title"], "示例标")

    def test_extract_json_repairs_trailing_comma(self):
        data = extract_json('{"title": "A", "sections": [{"title": "B",},],}')
        self.assertEqual(data["title"], "A")

    def test_strip_repeated_heading(self):
        content = "## 1.1 项目背景\n\n正文从这里开始。"
        self.assertEqual(strip_leading_heading(content, "项目背景"), "正文从这里开始。")

    def test_char_count_ignores_whitespace(self):
        self.assertEqual(char_count("甲 乙\n丙"), 3)


class OutlineTests(unittest.TestCase):
    def _project(self, words=12000, bid_type="技术标"):
        return ProjectBrief(name="政务服务平台", bid_type=bid_type, summary="统一受理", target_words=words)

    def test_fallback_marks_parents_and_tracks_budget(self):
        outline = fallback_outline(self._project(12000))
        parents = [item for item in outline.sections if item.is_parent]
        leaves = [item for item in outline.sections if not item.is_parent]
        self.assertGreaterEqual(len(parents), 5)
        self.assertGreaterEqual(len(leaves), 10)
        total = sum(item.target_words for item in outline.sections)
        self.assertLess(abs(total - 12000), 80)
        self.assertTrue(any(item.number == "1.1" for item in leaves))

    def test_full_bid_contains_business_chapter(self):
        outline = fallback_outline(self._project(20000, "完整投标文件"))
        titles = " ".join(item.title for item in outline.sections)
        self.assertIn("资质与认证", titles)
        self.assertIn("总体架构", titles)

    def test_parse_model_json_uses_points_as_content_plan(self):
        raw = """
        {
          "title": "政务服务平台技术标",
          "sections": [
            {"number": "1", "level": 1, "title": "需求理解", "points": "总述", "target_words": 200},
            {"number": "1.2", "level": 2, "title": "性能需求", "points": "并发与时限", "target_words": 400},
            {"number": "1.1", "level": 2, "title": "业务需求", "points": "受理与出件", "target_words": 1600}
          ]
        }
        """
        outline = parse_outline(raw, self._project(3000))
        numbers = [item.number for item in outline.sections]
        self.assertEqual(numbers, ["1", "1.1", "1.2"])
        parent = outline.sections[0]
        self.assertTrue(parent.is_parent)
        self.assertIn("业务需求", parent.child_titles)
        self.assertEqual(outline.source, "model")


class KnowledgeTests(unittest.TestCase):
    def test_chunk_and_search_prefers_matching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "架构方案.md").write_text(
                "总体架构采用分层设计。接入层、应用层、数据层和集成层分开部署。"
                "办件中心负责受理、补正、流转和办结。材料放入对象存储，不进业务库。" * 8,
                encoding="utf-8",
            )
            (raw / "食堂采购.md").write_text(
                "食堂食材采购按周报价。蔬菜、肉类和粮油分开验收，留存检验报告和配送温度记录。" * 12,
                encoding="utf-8",
            )
            kb = KnowledgeBase(root)
            report = kb.build()
            self.assertGreaterEqual(report.chunks, 2)
            hits = kb.search("办件流转 总体架构 对象存储", k=2)
            self.assertTrue(hits)
            self.assertIn("架构方案.md", hits[0].source)

    def test_chunk_text_splits_long_paragraph(self):
        chunks = chunk_text("甲" * 2000)
        self.assertGreater(len(chunks), 1)


class AssembleTests(unittest.TestCase):
    def test_markdown_and_docx(self):
        section = SectionSpec(number="1", level=1, title="需求理解", points="", target_words=300, content="正文内容")
        child = SectionSpec(number="1.1", level=2, title="业务需求", points="", target_words=800, content="### 受理\n\n- 支持补正")
        from tender_agent.outline import Outline, mark_parents

        outline = Outline(title="示例技术标", sections=[section, child], source="test")
        mark_parents(outline.sections)
        markdown = assemble_markdown(outline)
        self.assertIn("# 示例技术标", markdown)
        self.assertIn("## 1 需求理解", markdown)
        self.assertIn("### 1.1 业务需求", markdown)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            md_path = export_markdown(markdown, directory, "示例")
            docx_path = export_docx(outline, directory, "示例")
            self.assertTrue(md_path.read_text(encoding="utf-8").startswith("# 示例技术标"))
            self.assertGreater(docx_path.stat().st_size, 1000)


class ClientTests(unittest.TestCase):
    def test_cursor_sdk_is_not_imported_by_default(self):
        self.assertNotIn("cursor_sdk", sys.modules)

    def test_cursor_client_explains_missing_sdk(self):
        import builtins

        from tender_agent.config import ModelProfile

        profile = ModelProfile(id="cursor", label="Cursor", provider="cursor", model="composer-2.5")
        client = CursorClient(profile, "test-key")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cursor_sdk":
                raise ImportError("missing")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", fake_import):
            with self.assertRaises(RuntimeError) as caught:
                client.complete("系统", "用户")
        self.assertIn("Cursor SDK", str(caught.exception))

    @patch("openai.OpenAI")
    def test_deepseek_chat_returns_content(self, openai_cls):
        from tender_agent.config import ModelProfile

        message = MagicMock()
        message.content = "这是正文"
        message.reasoning_content = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        openai_cls.return_value.chat.completions.create.return_value.choices = [choice]
        profile = ModelProfile(
            id="deepseek-chat",
            label="DeepSeek Chat",
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            max_tokens=1024,
            temperature=0.4,
        )
        client = OpenAICompatClient(profile, "sk-test")
        result = client.complete("系统", "写一段")
        self.assertEqual(result.text, "这是正文")
        kwargs = openai_cls.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["temperature"], 0.4)

    @patch("openai.OpenAI")
    def test_reasoner_omits_temperature(self, openai_cls):
        from tender_agent.config import ModelProfile

        message = MagicMock()
        message.content = "{}"
        message.reasoning_content = "思考"
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        openai_cls.return_value.chat.completions.create.return_value.choices = [choice]
        profile = ModelProfile(
            id="deepseek-reasoner",
            label="Reasoner",
            provider="deepseek",
            model="deepseek-reasoner",
            max_tokens=1024,
        )
        OpenAICompatClient(profile, "sk-test").complete("系统", "大纲", json_mode=True)
        kwargs = openai_cls.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
