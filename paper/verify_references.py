#!/usr/bin/env python3
"""
参考文献验证脚本
用于验证 BibTeX 文件中的参考文献信息是否准确
使用 CrossRef / Semantic Scholar / DBLP API 进行查询

修复点（相对你 Downloads 里的版本）：
- 更稳健的 BibTeX 字段解析：支持 title = "..."、title = {...}、多行、以及包含 {C}ode{A}gent 这类花括号片段
- 报告输出路径改为当前目录，避免写入不存在的 /mnt/... 路径
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# API 配置
CROSSREF_API = "https://api.crossref.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DBLP_API = "https://dblp.org/search/publ/api"


@dataclass
class BibEntry:
    """BibTeX 条目"""

    key: str
    entry_type: str
    title: str
    authors: List[str]
    year: str
    journal: Optional[str] = None
    booktitle: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    raw: str = ""


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _extract_bibtex_value(body: str, field_name: str) -> Optional[str]:
    """
    从条目 body 中提取字段值，支持：
    - field = { ... }（支持嵌套括号计数）
    - field = " ... "（支持转义引号）
    - field = bareword（直到逗号/换行）
    """
    # 找到 "field_name =" 的起点（忽略大小写，不要求行首）
    m = re.search(rf"(?im)\b{re.escape(field_name)}\s*=", body)
    if not m:
        return None

    i = m.end()
    n = len(body)

    # 跳过空白
    while i < n and body[i].isspace():
        i += 1
    if i >= n:
        return None

    # 解析值
    if body[i] == "{":
        i += 1
        start = i
        depth = 1
        while i < n:
            ch = body[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return _collapse_ws(body[start:i])
            i += 1
        # 不平衡括号：退化为 None
        return None

    if body[i] == '"':
        i += 1
        start = i
        escaped = False
        while i < n:
            ch = body[i]
            if escaped:
                escaped = False
            else:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    return _collapse_ws(body[start:i])
            i += 1
        return None

    # bareword：读到逗号或换行
    start = i
    while i < n and body[i] not in ",\n\r":
        i += 1
    return _collapse_ws(body[start:i]) or None


def parse_bib_file(bib_path: str) -> List[BibEntry]:
    """解析 BibTeX 文件（轻量实现，足以覆盖常见 bib 写法）"""
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries: List[BibEntry] = []

    # 匹配 BibTeX 条目：找到 @type{key, ... } 的结构
    # 使用括号计数来正确匹配整个条目
    i = 0
    n = len(content)
    
    while i < n:
        # 找到 @type{ 的开始
        match = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", content[i:], re.IGNORECASE)
        if not match:
            break
        
        start_pos = i + match.start()
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        
        # 从 key 后面的逗号开始，找到条目的结束位置（匹配括号）
        body_start = i + match.end()
        depth = 1  # 已经有一个左括号
        j = body_start
        
        while j < n and depth > 0:
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
            j += 1
        
        if depth == 0:
            # 找到了完整的条目
            body = content[body_start:j-1]  # j-1 是最后一个 } 的位置
            i = j  # 继续从下一个位置搜索
        else:
            # 括号不匹配，跳过这个条目
            i = body_start
            continue

        title = _extract_bibtex_value(body, "title") or ""
        author_str = _extract_bibtex_value(body, "author") or ""
        year = _extract_bibtex_value(body, "year") or ""
        journal = _extract_bibtex_value(body, "journal")
        booktitle = _extract_bibtex_value(body, "booktitle")
        volume = _extract_bibtex_value(body, "volume")
        pages = _extract_bibtex_value(body, "pages")
        doi = _extract_bibtex_value(body, "doi")

        # 提取 arXiv ID（兼容 eprint / url / journal）
        arxiv_id = None
        eprint = _extract_bibtex_value(body, "eprint")
        if eprint:
            arxiv_id = eprint
        else:
            url = _extract_bibtex_value(body, "url") or ""
            arxiv_match = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
            elif journal and "arxiv" in journal.lower():
                arxiv_match = re.search(r"arXiv:(\d+\.\d+)", journal)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)

        # 解析作者（保持你原逻辑）
        authors: List[str] = []
        if author_str:
            author_str = re.sub(
                r"\s+and\s+others\s*$", "", author_str, flags=re.IGNORECASE
            )
            for author in author_str.split(" and "):
                author = author.strip()
                if author:
                    authors.append(author)

        entry = BibEntry(
            key=key,
            entry_type=entry_type,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            booktitle=booktitle,
            volume=volume,
            pages=pages,
            arxiv_id=arxiv_id,
            doi=doi,
            raw=match.group(0),
        )
        entries.append(entry)

    return entries


def normalize_text(text: str) -> str:
    """规范化文本用于比较"""
    if not text:
        return ""
    text = re.sub(r"[{}\\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_words(text: str) -> List[str]:
    """从标题中提取单词，处理 {R}eflec{T}ool 这种情况"""
    if not text:
        return []
    # 移除花括号（保留内容），然后提取单词
    # {R}eflec{T}ool -> ReflecTool
    text = re.sub(r"\{([^}]+)\}", r"\1", text)
    # 移除其他特殊字符，保留字母数字和连字符
    text = re.sub(r"[^\w\s-]", " ", text)
    # 分割成单词（保留连字符，如 "Tool-Augmented"）
    words = re.findall(r"\b[\w-]+\b", text)
    # 转小写并过滤空字符串
    return [w.lower() for w in words if w.strip()]


def compare_titles_word_by_word(local_title: str, online_title: str) -> Dict:
    """逐词比较两个标题，返回详细的差异信息"""
    local_words = extract_words(local_title)
    online_words = extract_words(online_title)
    
    local_set = set(local_words)
    online_set = set(online_words)
    
    missing_words = local_set - online_set  # 本地有但在线没有的单词
    extra_words = online_set - local_set     # 在线有但本地没有的单词
    common_words = local_set & online_set    # 共同单词
    
    return {
        "local_words": local_words,
        "online_words": online_words,
        "missing_words": sorted(missing_words),
        "extra_words": sorted(extra_words),
        "common_words": sorted(common_words),
        "word_overlap_ratio": len(common_words) / max(len(local_set), len(online_set), 1)
    }


def _http_get_json(
    url: str, params: Dict[str, str], headers: Dict[str, str], timeout: int = 15
) -> Tuple[int, Optional[Dict]]:
    """标准库 GET + JSON 解析（避免 requests 依赖）"""
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    full_url = f"{url}?{qs}" if qs else url
    req = urllib.request.Request(full_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return status, json.loads(raw)
            except Exception:
                return status, None
    except Exception:
        return 0, None


def query_crossref(title: str, author: str = None) -> Optional[Dict]:
    """通过 CrossRef API 查询"""
    try:
        params = {"query.title": title, "rows": 3}
        if author:
            params["query.author"] = (
                author.split(",")[0] if "," in author else author.split()[-1]
            )

        headers = {"User-Agent": "RefChecker/1.0 (mailto:test@example.com)"}
        status, data = _http_get_json(CROSSREF_API, params=params, headers=headers, timeout=15)
        if status != 200 or not data:
            return None
        items = data.get("message", {}).get("items", [])

        best_match = None
        best_score = 0.0
        for item in items:
            item_title = item.get("title", [""])[0] if item.get("title") else ""
            score = similarity(title, item_title)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = item

        if not best_match:
            return None

        authors = []
        for auth in best_match.get("author", []):
            name = auth.get("literal") or f"{auth.get('given', '')} {auth.get('family', '')}".strip()
            if name:
                authors.append(name)

        year = None
        issued = best_match.get("issued", {})
        if issued and "date-parts" in issued and issued["date-parts"]:
            date_parts = issued["date-parts"][0]
            if date_parts:
                year = str(date_parts[0])

        return {
            "title": best_match.get("title", [""])[0],
            "authors": authors,
            "year": year,
            "journal": best_match.get("container-title", [""])[0]
            if best_match.get("container-title")
            else None,
            "volume": best_match.get("volume"),
            "pages": best_match.get("page"),
            "doi": best_match.get("DOI"),
            "source": "CrossRef",
            "match_score": best_score,
        }
    except Exception as e:
        print(f"  CrossRef 查询错误: {e}")
        return None


def query_semantic_scholar(title: str) -> Optional[Dict]:
    """通过 Semantic Scholar API 查询"""
    try:
        headers = {"User-Agent": "RefChecker/1.0"}
        params = {
            "query": title,
            "limit": 3,
            "fields": "title,authors,year,venue,publicationVenue,externalIds",
        }
        status, data = _http_get_json(S2_API, params=params, headers=headers, timeout=15)
        if status != 200 or not data:
            return None
        papers = data.get("data", [])

        best_match = None
        best_score = 0.0
        for paper in papers:
            paper_title = paper.get("title", "")
            score = similarity(title, paper_title)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = paper

        if not best_match:
            return None

        authors = [a.get("name", "") for a in best_match.get("authors", [])]
        return {
            "title": best_match.get("title"),
            "authors": authors,
            "year": str(best_match.get("year")) if best_match.get("year") else None,
            "journal": best_match.get("venue")
            or (best_match.get("publicationVenue", {}) or {}).get("name"),
            "arxiv_id": (best_match.get("externalIds", {}) or {}).get("ArXiv"),
            "source": "SemanticScholar",
            "match_score": best_score,
        }
    except Exception as e:
        print(f"  Semantic Scholar 查询错误: {e}")
        return None


def query_dblp(title: str) -> Optional[Dict]:
    """通过 DBLP API 查询 - 对计算机科学文献特别有效"""
    try:
        params = {"q": title, "format": "json", "h": 5}
        headers = {"User-Agent": "RefChecker/1.0"}
        status, data = _http_get_json(DBLP_API, params=params, headers=headers, timeout=15)
        if status != 200 or not data:
            return None
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return None

        best_match = None
        best_score = 0.0
        for hit in hits:
            info = hit.get("info", {})
            hit_title = (info.get("title", "") or "").rstrip(".")
            score = similarity(title, hit_title)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = info

        if not best_match:
            return None

        authors = []
        authors_info = best_match.get("authors", {}).get("author", [])
        if isinstance(authors_info, dict):
            authors_info = [authors_info]
        for author in authors_info:
            if isinstance(author, dict):
                authors.append(author.get("text", ""))
            elif isinstance(author, str):
                authors.append(author)

        venue = best_match.get("venue", "")
        if isinstance(venue, list):
            venue = venue[0] if venue else ""

        return {
            "title": (best_match.get("title", "") or "").rstrip("."),
            "authors": authors,
            "year": best_match.get("year"),
            "journal": venue,
            "volume": best_match.get("volume"),
            "pages": best_match.get("pages"),
            "doi": best_match.get("doi"),
            "dblp_url": best_match.get("url"),
            "source": "DBLP",
            "match_score": best_score,
        }
    except Exception as e:
        print(f"  DBLP 查询错误: {e}")
        return None


def verify_entry(entry: BibEntry) -> Dict:
    """验证单个条目"""
    result = {
        "key": entry.key,
        "local_title": entry.title,
        "local_authors": entry.authors,
        "local_year": entry.year,
        "issues": [],
        "warnings": [],
        "online_info": None,
        "status": "unknown",
    }

    if not entry.title:
        result["issues"].append("缺少标题（通常是 BibTeX 解析未覆盖 title 的写法，比如 title=\"...\"）")
        result["status"] = "error"
        return result

    # 先尝试 DBLP（对计算机科学文献最全面）
    online = query_dblp(entry.title)
    if not online:
        time.sleep(0.5)
        online = query_semantic_scholar(entry.title)
    if not online:
        time.sleep(0.5)
        first_author = entry.authors[0] if entry.authors else None
        online = query_crossref(entry.title, first_author)

    if not online:
        result["warnings"].append("在 DBLP、Semantic Scholar 和 CrossRef 中均未找到匹配的文献")
        result["status"] = "not_found"
        return result

    result["online_info"] = online

    # 标题检查：字符串比较（规范化后）
    online_title = online.get("title", "")
    local_title_norm = normalize_text(entry.title)
    online_title_norm = normalize_text(online_title)
    
    if local_title_norm != online_title_norm:
        result["issues"].append("标题不一致")
        result["issues"].append(f"  本地: {entry.title}")
        result["issues"].append(f"  在线: {online_title}")

    if entry.year and online.get("year"):
        if str(entry.year).strip() != str(online["year"]).strip():
            result["issues"].append(f"年份不一致: 本地={entry.year}, 在线={online['year']}")

    # 检查并打印 journal 字段（即使值为空字符串也打印，只要字段存在）
    if entry.journal is not None or online.get("journal"):
        result["warnings"].append("Journal 信息:")
        result["warnings"].append(f"  本地: {entry.journal if entry.journal else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('journal', '(无)')}")

    # 检查并打印 booktitle 字段
    if entry.booktitle is not None or online.get("booktitle"):
        result["warnings"].append("Booktitle 信息:")
        result["warnings"].append(f"  本地: {entry.booktitle if entry.booktitle else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('booktitle', '(无)')}")

    # 检查并打印 volume 字段
    if entry.volume is not None or online.get("volume"):
        result["warnings"].append("Volume 信息:")
        result["warnings"].append(f"  本地: {entry.volume if entry.volume else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('volume', '(无)')}")

    # 检查并打印 pages 字段
    if entry.pages is not None or online.get("pages"):
        result["warnings"].append("Pages 信息:")
        result["warnings"].append(f"  本地: {entry.pages if entry.pages else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('pages', '(无)')}")

    # 检查并打印 DOI 字段
    if entry.doi is not None or online.get("doi"):
        result["warnings"].append("DOI 信息:")
        result["warnings"].append(f"  本地: {entry.doi if entry.doi else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('doi', '(无)')}")

    # 检查并打印 arXiv ID 字段
    if entry.arxiv_id is not None or online.get("arxiv_id"):
        result["warnings"].append("arXiv ID 信息:")
        result["warnings"].append(f"  本地: {entry.arxiv_id if entry.arxiv_id else '(无)'}")
        result["warnings"].append(f"  在线: {online.get('arxiv_id', '(无)')}")

    if entry.authors and online.get("authors"):
        def normalize_author_name(name: str) -> str:
            """规范化作者名称：移除花括号、反斜杠、数字后缀（如 0002）"""
            if not name:
                return ""
            # 移除 LaTeX 特殊字符
            name = re.sub(r"[{}\\]", "", name)
            # 移除末尾的数字后缀（如 "Zhi Gao 0002" -> "Zhi Gao"）
            name = re.sub(r"\s+\d+$", "", name)
            return name.strip()

        def author_to_word_set(name: str) -> set:
            """将作者名称转换为单词集合（用于比较，防止姓和名顺序不一致）"""
            name = normalize_author_name(name)
            # 转小写，移除标点，统一空格
            name = re.sub(r"[^\w\s]", " ", name)
            name = re.sub(r"\s+", " ", name)
            # 按空格split并转小写
            words = [w.lower() for w in name.split() if w.strip()]
            return set(words)

        # 规范化所有作者名称（用于显示）
        local_authors_norm = [normalize_author_name(a) for a in entry.authors]
        online_authors_norm = [normalize_author_name(a) for a in online["authors"]]

        # 将每个作者转换为单词集合
        local_author_sets = [author_to_word_set(a) for a in entry.authors]
        online_author_sets = [author_to_word_set(a) for a in online["authors"]]

        # 检查作者数量
        if len(local_author_sets) != len(online_author_sets):
            result["warnings"].append(
                f"作者数量不一致: 本地={len(local_author_sets)}, 在线={len(online_author_sets)}"
            )

        # 检查每个作者是否匹配（使用set比较）
        max_len = max(len(local_author_sets), len(online_author_sets))
        mismatched_authors = []
        
        for i in range(max_len):
            local_set = local_author_sets[i] if i < len(local_author_sets) else set()
            online_set = online_author_sets[i] if i < len(online_author_sets) else set()
            local_author = local_authors_norm[i] if i < len(local_authors_norm) else None
            online_author = online_authors_norm[i] if i < len(online_authors_norm) else None
            
            if not local_set:
                mismatched_authors.append(f"位置 {i+1}: 本地缺失，在线={online_author}")
            elif not online_set:
                mismatched_authors.append(f"位置 {i+1}: 本地={local_author}，在线缺失")
            elif local_set != online_set:
                mismatched_authors.append(
                    f"位置 {i+1}: 本地={entry.authors[i]}，在线={online['authors'][i]}"
                )

        if mismatched_authors:
            result["warnings"].append("作者列表不一致:")
            for mismatch in mismatched_authors:
                result["warnings"].append(f"  - {mismatch}")

    result["status"] = "has_issues" if result["issues"] else "ok"
    return result


def main():
    bib_file = "/Users/xiaocan/Toolmem/bib.bib"

    print("=" * 80)
    print("参考文献验证报告")
    print("=" * 80)
    print(f"\nBibTeX 文件: {bib_file}\n")

    entries = parse_bib_file(bib_file)
    print(f"共找到 {len(entries)} 个参考文献条目\n")
    print("-" * 80)

    results = []
    ok_count = 0
    issue_count = 0
    not_found_count = 0
    error_count = 0

    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] 正在验证: {entry.key}")
        print(f"  标题: {entry.title[:60]}..." if len(entry.title) > 60 else f"  标题: {entry.title}")

        result = verify_entry(entry)
        results.append(result)

        if result["status"] == "ok":
            print("  ✓ 验证通过")
            ok_count += 1
        elif result["status"] == "not_found":
            print("  ? 未找到在线记录")
            not_found_count += 1
            for w in result["warnings"]:
                print(f"    - {w}")
        elif result["status"] == "has_issues":
            print("  ✗ 发现问题:")
            issue_count += 1
            for issue in result["issues"]:
                print(f"    - {issue}")
            for w in result["warnings"]:
                print(f"    警告: {w}")
        else:
            print("  ✗ 解析/数据错误:")
            error_count += 1
            for issue in result["issues"]:
                print(f"    - {issue}")

        time.sleep(1.0)  # 避免 API 限制

    print("\n" + "=" * 80)
    print("验证摘要")
    print("=" * 80)
    print(f"总条目数: {len(entries)}")
    print(f"验证通过: {ok_count}")
    print(f"存在问题: {issue_count}")
    print(f"未找到记录: {not_found_count}")
    print(f"解析/数据错误: {error_count}")

    report_file = "./reference_verification_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存到: {report_file}")


if __name__ == "__main__":
    main()

