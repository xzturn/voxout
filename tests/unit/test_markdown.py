"""markdown → 可朗读文本。

处理的构件由真实语料（63 Short-form Posts 两篇）决定，不做投机泛化。
"""

import pytest

from voxout.markdown import to_speakable


def test_frontmatter_is_stripped_whole() -> None:
    md = '---\ntitle: "留意波动"\nsource: "[[某本书]]"\n---\n\n正文在这里。'
    assert to_speakable(md) == "正文在这里。"


def test_heading_becomes_its_own_sentence() -> None:
    """标题没有句末标点，直接与下一段相连会被读成一句话。"""
    assert to_speakable("# 留意波动\n\n正文。") == "留意波动。\n正文。"


def test_heading_with_terminal_punctuation_is_left_alone() -> None:
    assert to_speakable("# 真的吗？\n\n正文。") == "真的吗？\n正文。"


def test_wiki_link_with_alias_reads_the_alias() -> None:
    assert to_speakable("我读了[[2026-07-22 留意波动|那篇短帖]]。") == "我读了那篇短帖。"


def test_wiki_link_without_alias_reads_the_target() -> None:
    assert to_speakable("我读了[[逆龄生长]]。") == "我读了逆龄生长。"


def test_markdown_link_reads_the_text_not_the_url() -> None:
    assert to_speakable("见[这篇文章](https://example.com/a?b=c)。") == "见这篇文章。"


def test_fenced_code_block_is_dropped_entirely() -> None:
    md = "前面。\n\n```python\nprint('不要读我')\n```\n\n后面。"
    assert to_speakable(md) == "前面。\n后面。"


def test_inline_code_keeps_its_content() -> None:
    assert to_speakable("参数是 `max_tokens` 这个。") == "参数是 max_tokens 这个。"


def test_emphasis_markers_are_removed() -> None:
    assert to_speakable("这**很重要**也*挺有趣*。") == "这很重要也挺有趣。"


def test_blockquote_marker_is_removed() -> None:
    assert to_speakable("> 引用的一句话。") == "引用的一句话。"


def test_list_markers_are_removed() -> None:
    md = "- 第一条。\n- 第二条。\n1. 第三条。"
    assert to_speakable(md) == "第一条。\n第二条。\n第三条。"


def test_image_is_dropped_but_surrounding_text_survives() -> None:
    assert to_speakable("看图![示意图](x.png)就懂了。") == "看图就懂了。"


def test_html_comment_is_dropped() -> None:
    assert to_speakable("正文。<!-- 这是给人看的备注 -->") == "正文。"


def test_horizontal_rule_is_dropped() -> None:
    assert to_speakable("上半。\n\n---\n\n下半。") == "上半。\n下半。"


def test_paragraph_boundaries_survive_as_newlines() -> None:
    """chunker 依赖 \\n 作为切段点，清洗不能把段落界丢掉。"""
    assert to_speakable("第一段。\n\n第二段。\n\n第三段。") == "第一段。\n第二段。\n第三段。"


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="空"):
        to_speakable("")


def test_input_with_nothing_speakable_is_rejected() -> None:
    """只有 frontmatter 和代码块——不能悄悄产出一个静音文件（设计 §9）。"""
    md = "---\ntitle: x\n---\n\n```python\npass\n```\n"
    with pytest.raises(ValueError, match="可朗读"):
        to_speakable(md)
