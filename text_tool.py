from server import mcp
from typing import Any, Dict, List


def text_word_count(text: str) -> int:
    """
    Count words in text using simple whitespace splitting.

    Args:
        text: Input text.

    Returns:
        Integer word count.

    Semantic use:
        Useful for heuristics and quick metadata extraction for documents.
    """
    return len([w for w in text.split() if w.strip()])


def text_summary_simple(text: str, max_sentences: int = 2) -> str:
    """
    Produce a very simple extractive summary by taking the first `max_sentences` sentences.

    Args:
        text: Input text to summarize.
        max_sentences: Number of sentences to return (default 2).

    Returns:
        Extractive summary string.

    Note:
        Lightweight; not an ML summarizer, but useful as a quick heuristic in pipelines.
    """
    import re
    sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    return " ".join(sentences[:max_sentences])


def text_keywords_simple(text: str, top_n: int = 5) -> List[str]:
    """
    Extract simple keyword candidates by frequency after lowercasing and removing short tokens.

    Args:
        text: Input text.
        top_n: Number of top tokens to return.

    Returns:
        List of keyword strings.

    Note:
        Simple heuristic; suitable for quick tagging or pre-filtering before heavy NLP.
    """
    import re
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9']+", text)]
    tokens = [t for t in tokens if len(t) > 2]
    from collections import Counter
    freq = Counter(tokens)
    return [t for t, _ in freq.most_common(top_n)]


@mcp.tool()
def text_tool(action: str, text: str = "", top_n: int = 5) -> Dict[str, Any]:
    """
    Text processing tool with simple operations: count, summary, keywords.

    Args:
        action: One of ["count", "summary", "keywords"].
        text: Input text to process.
        top_n: For keywords action, number of keywords to return.

    Returns:
        Dictionary with action result.

    Semantic description:
        Lightweight NLP helpers used for metadata extraction and quick embeddings.
    """
    action = action.lower()
    if action == "count":
        return {"word_count": text_word_count(text)}
    if action == "summary":
        return {"summary": text_summary_simple(text)}
    if action == "keywords":
        return {"keywords": text_keywords_simple(text, top_n)}
    raise ValueError("Unsupported action. Use count/summary/keywords.")
