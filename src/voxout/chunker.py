"""长文切段。

**段落是原子单位。** 一次生成里绝不放两个段落，哪怕两段加起来只有十几个汉字。

这条规则来自 T13 的实测，也是这个模块存在的主要理由。原先的规则是「按 token 上限
打包，900 汉字以内一次送出」，依据是「实测 1050 汉字单次调用零截断」。那次探测测
的是**截断**——音频有没有提前结束；它测不到**跳读**——音频时长正常、听起来完整，
中间少了一整句。当时闸门 1（ASR 回环）还不存在，跳读因此从未被测量过。

补测的读数（house 音色，两篇真实短帖）：

    帖1 213 汉字单段    覆盖率 55.40%   0.1390 秒每汉字   ← 跳掉约四成
    帖1 切到 ≤80 汉字   覆盖率 99.06%   0.2325–0.2533 秒每汉字
    帖2 540 汉字单段    覆盖率 85.00%   按念到的字算 0.2499 秒每汉字
    帖2 首段 80 汉字（标题+两段落合并）  覆盖率 68.75%

「秒每汉字」是与覆盖率完全独立的旁证：跳读时音频时长会同比例变短，而只要按**实际
念出的**字数折算就又回到正常区间——两个指标一致指向内容真的没被念出来。凡是跳读的
生成都含不止一个段落，凡是单段落生成都只剩同音字级别的零星偏差。

所以 token 上限（budget.py）不再是切段的驱动力，只是它的上界；`CHUNK_THRESHOLD_HAN`
退化成「超长单个段落的切分粒度」。第二条纪律不变：**切段不是丢内容的许可**。无标点
的超长句宁可整段送出并告警，也不许在字中间硬切——硬切会同时破坏韵律和闸门 1 的可读性。
"""

import logging

from voxout.config import CHUNK_THRESHOLD_HAN
from voxout.textmatch import han_only

_LOG = logging.getLogger(__name__)

_PARAGRAPH_BREAK = "\n"
"""markdown 清洗后的段落界。段落先分开，之后才轮到句末标点。"""

_SENTENCE_ENDS = "。！？；"
"""段落内部的切分点。`\\n` **不在**其中：它是更硬的界，由 `_paragraphs` 处理。"""


def _split_sentences(text: str) -> tuple[str, ...]:
    """按句末标点切成句子，标点跟在前一句尾部（韵律属于前一句）。"""
    sentences: list[str] = []
    current: list[str] = []

    for char in text:
        current.append(char)
        if char in _SENTENCE_ENDS:
            sentences.append("".join(current))
            current = []

    if current:
        sentences.append("".join(current))
    return tuple(sentences)


def _pack(sentences: tuple[str, ...], threshold_han: int) -> tuple[str, ...]:
    """贪心打包：句子装得下就装，装不下才开新段。"""
    segments: list[str] = []
    current: list[str] = []
    current_han = 0

    for index, sentence in enumerate(sentences):
        sentence_han = len(han_only(sentence))
        if current and current_han + sentence_han > threshold_han:
            segments.append("".join(current))
            current, current_han = [], 0

        if not current and sentence_han > threshold_han:
            _LOG.warning(
                "第 %d 句共 %d 汉字，超过阈值 %d 且内部无句末标点可切；整句送出以免丢字",
                index + 1,
                sentence_han,
                threshold_han,
            )

        current.append(sentence)
        current_han += sentence_han

    if current:
        segments.append("".join(current))
    return tuple(segments)


def _paragraphs(text: str) -> tuple[str, ...]:
    """按 `\\n` 分段并丢掉空段落。

    空段落必须丢：markdown 常留空行，而把空串送进模型会生成一段静音——那是
    「有正确时长的错误产物」，正是闸门体系要防的那类失败。
    """
    return tuple(
        stripped for line in text.split(_PARAGRAPH_BREAK) if (stripped := line.strip())
    )


def chunk(text: str, threshold_han: int = CHUNK_THRESHOLD_HAN) -> tuple[str, ...]:
    """把文本切成逐段落、且每段不超过 `threshold_han` 汉字的片段。

    段落永不合并（模块文档：跨段落生成会跳读）。段落自身未超阈值时原样返回，
    不做任何重新拼装。无句末标点可切的超长段落会超出阈值并告警。

    Raises:
        ValueError: 文本为空。
    """
    paragraphs = _paragraphs(text)
    if not paragraphs:
        raise ValueError("待切段的文本为空")

    segments: list[str] = []
    for paragraph in paragraphs:
        if len(han_only(paragraph)) <= threshold_han:
            segments.append(paragraph)
            continue
        segments.extend(
            stripped
            for packed in _pack(_split_sentences(paragraph), threshold_han)
            if (stripped := packed.strip())
        )

    return tuple(segments)
