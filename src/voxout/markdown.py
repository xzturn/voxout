"""markdown → 可朗读文本。

只处理标定语料里实际出现的构件：frontmatter、H1、
wiki 链接、行内代码、引号类标点。不做投机泛化——表格、脚注、HTML 标签
等等在语料里不存在，等真出现了再加，届时有真实样本可对着写测试。

段落界用 `\\n` 保留，chunker 依赖它作为切段点。
"""

import re

_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n", re.DOTALL)
_FENCED_CODE = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_WIKI_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"\*([^*]+)\*")
_HORIZONTAL_RULE = re.compile(r"\A(?:[-*_][ \t]*){3,}\Z")
_HEADING = re.compile(r"\A#{1,6}[ \t]+(.*)\Z")
_BLOCKQUOTE = re.compile(r"\A>[ \t]*")
_LIST_MARKER = re.compile(r"\A[ \t]*(?:[-*+][ \t]+|\d+\.[ \t]+)")

_TERMINAL_PUNCTUATION = "。！？；…”」》.!?:："


def _clean_block_syntax(md: str) -> str:
    """先干掉整块构件，再逐行处理——顺序反了会把代码块里的 # 当标题。"""
    text = _FRONTMATTER.sub("", md)
    text = _FENCED_CODE.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    return _IMAGE.sub("", text)


def _clean_inline_syntax(text: str) -> str:
    """行内构件。wiki 链接必须在普通链接之前——`[[a|b]]` 的内层会被 `[x](y)` 误吃。"""
    text = _WIKI_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _MD_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    return _ITALIC.sub(r"\1", text)


def _speakable_line(line: str) -> str:
    """单行 → 可朗读文本；返回空串表示这一行不该被读。"""
    stripped = line.strip()
    if not stripped or _HORIZONTAL_RULE.match(stripped):
        return ""

    heading = _HEADING.match(stripped)
    if heading:
        title = heading.group(1).strip()
        # 标题没有句末标点，与下一段相连会被读成一句话，故补上句号。
        return title if title.endswith(tuple(_TERMINAL_PUNCTUATION)) else f"{title}。"

    stripped = _BLOCKQUOTE.sub("", stripped)
    return _LIST_MARKER.sub("", stripped).strip()


def to_speakable(md: str) -> str:
    """把 markdown 正文清洗成可直接送进 TTS 的纯文本。

    Raises:
        ValueError: 输入为空，或清洗后没有任何可朗读内容。宁可明确失败，
            也不要产出一个静音 wav 让人以为渲染成功了（设计 §9）。
    """
    if not md.strip():
        raise ValueError("输入为空：没有可朗读的内容")

    cleaned = _clean_inline_syntax(_clean_block_syntax(md))
    lines = [spoken for line in cleaned.splitlines() if (spoken := _speakable_line(line))]
    if not lines:
        raise ValueError("清洗后没有可朗读文本：整篇只有 frontmatter、代码块或图片")

    return "\n".join(lines)
