"""research_runner.py — Deep research engine for Resynth.

Architecture:
1. Decompose user query into sub-queries
2. Search all sub-queries in parallel via DuckDuckGo
3. Scrape top sources for full content
4. Synthesize into structured report with citations using qwen-max
5. Stream the report back with SSE
"""
import os
import json
import re
import datetime
import concurrent.futures

try:
    from openai import OpenAI as _OAI
    _client = _OAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
except Exception:
    _client = None

# Fallback to Groq
_groq = None
try:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        from groq import Groq
        _groq = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
except Exception:
    pass

PRIMARY_MODEL = "qwen3.5-omni-plus"
GROQ_MODEL    = "llama-3.3-70b-versatile"
FAST_MODEL    = "qwen3-omni-flash"

RESEARCH_SYSTEM_PROMPT = """You are Resynth Research, an advanced AI research assistant.

STRUCTURE YOUR RESPONSE DYNAMICALLY:
- Read the user's query and adapt your length. Simple questions = short answers.
- Use headings (##) and emojis for complex topics.
- ALWAYS cite sources INLINE: [Source Name](URL).
- DO NOT use generic numbers like [1].
- VISUAL DIAGRAMS: Use Mermaid syntax (```mermaid ... ```) to create flowcharts or diagrams for complex processes.
- MATH: Use LaTeX for all mathematical expressions. Use $ ... $ for inline and $$ ... $$ for block math.

CONCISENESS:
- Unless 'Deep Dive' is active, be concise (2-4 paragraphs max).
- If it's a greeting or simple fact, be very brief.

IDENTITY:
- You are Resynth Research by Resynth Inc. Reply in the user's language. Do not repeat introductions. """

LIT_REVIEW_PROMPT = """You are performing a formal Literature Review. 
Your goal is to synthesize the provided research papers and web sources into a structured academic overview.

CRITICAL INSTRUCTIONS:
1. THEMATIC SYNTHESIS: Do not just list summaries. Organize your response by themes, concepts, or conflicting findings across different sources.
2. SOURCE COMPARISON: Explicitly mention where sources agree or disagree. Example: "[Source A] suggests X, however, [Source B] found Y."
3. GAP IDENTIFICATION: Point out any missing information or "gaps" in the current research as represented by the sources.
4. STRUCTURE: Use academic headings like ## Executive Summary, ## Current State of Research, ## Comparative Analysis, ## Key Methodologies (if applicable), and ## Conclusion.
5. CITATIONS: Use the source names as clickable inline links.
6. TONE: Maintain a high-level, objective, and scholarly tone.

Reply in the same language as the user query."""

DECOMPOSE_PROMPT = """Break this research query into 3 focused sub-queries for comprehensive research.
If the user's query contains typos, misspellings, or bad grammar, automatically correct them in your mind before creating the sub-queries.
Return ONLY a JSON array of 3 strings. Example: ["sub-query 1", "sub-query 2", "sub-query 3"]
Query: """

CLASSIFY_PROMPT = """You are a research tool classifier. Decide how to handle this query.

RESEARCH: Complex topics, academic subjects, coding, news, or any topic requiring a multi-source synthesis.
GREETING: "hi", "hello", "good morning", "how are you".
GENERAL: Simple facts, quick definitions, math, or direct questions that can be answered in 1-2 sentences without web research.

Reply with ONLY one word: RESEARCH, GREETING, or GENERAL.
Query: """

OFF_TOPIC_RESPONSES = [
    "I'm Resynth Research — built for deep research and learning. Try asking me something like *\"What are the effects of climate change on agriculture?\"* or *\"Explain quantum entanglement\"*.",
    "That's a bit outside my research scope! I'm specialized for academic and factual research. Ask me about any topic — science, history, technology, current events — and I'll find you real sources.",
    "I'm a research assistant, so I'm best at finding and synthesizing information from the web. Try a research question and I'll pull from multiple sources for you! 📚",
]

import random
import re as _re

FACT_CHECK_PROMPT = """You are a meticulous Research Auditor. 
Your task is to review the following research report against the provided raw source data.

CHECK FOR:
1. HALLUCINATIONS: Are there any claims in the report that are NOT supported by the sources?
2. MISCITATIONS: Does the text cite Source A but the information actually came from Source B (or nowhere)?
3. CONTRADICTIONS: Are there areas where the sources disagree, but the report ignored the conflict?

Return a JSON object with:
{
  "status": "VERIFIED" or "CLARIFIED",
  "notes": "A brief summary of your verification (e.g., 'All 12 claims verified' or 'Found minor contradiction regarding X').",
  "corrections": "Any specific sentences that should be updated."
}

Report to Audit: {report}
Sources: {sources}
"""

_conversations: dict = {}

# ── URL fetching ────────────────────────────────────────────────────────────────
URL_PATTERN = _re.compile(r'https?://[^\s<>"]+', _re.IGNORECASE)

def _fetch_url_content(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a URL. Handles arxiv & pubmed specially."""
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResynthBot/1.0)"}

    # Arxiv: convert /abs/ links to /pdf/ or use the API
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("/abs/")[-1].split("?")[0].strip()
        try:
            api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
            r = requests.get(api_url, timeout=10, headers=headers)
            soup = BeautifulSoup(r.text, "xml")
            entry = soup.find("entry")
            if entry:
                title   = entry.find("title").text.strip() if entry.find("title") else ""
                summary = entry.find("summary").text.strip() if entry.find("summary") else ""
                authors = ", ".join(a.find("name").text for a in entry.find_all("author") if a.find("name"))
                return f"**Paper:** {title}\n**Authors:** {authors}\n\n**Abstract:**\n{summary}"
        except Exception as e:
            print(f"[URL] ArXiv API failed: {e}")

    # PubMed: extract PMID and use eutils
    pubmed_match = _re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', url)
    if pubmed_match:
        pmid = pubmed_match.group(1)
        try:
            efetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
            r = requests.get(efetch, timeout=10, headers=headers)
            if r.status_code == 200:
                return r.text.strip()[:max_chars]
        except Exception as e:
            print(f"[URL] PubMed fetch failed: {e}")

    # Generic URL — fetch and parse HTML
    try:
        r = requests.get(url, timeout=12, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        # Try main content areas first
        for selector in ["article", "main", ".content", "#content", ".post", ".entry-content"]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(separator="\n", strip=True)[:max_chars]
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception as e:
        return f"[Could not fetch URL: {e}]"


def _get_client():
    if _client and os.environ.get("DASHSCOPE_API_KEY"):
        return _client, PRIMARY_MODEL, FAST_MODEL
    if _groq:
        return _groq, GROQ_MODEL, GROQ_MODEL
    return None, None, None


def _is_research_query(query: str, history: list = None) -> bool:
    """Quick classifier — returns False for off-topic chit-chat."""
    if history and len(history) > 0:
        return True  # If there's context, assume it's a follow-up

    chit_chat = [
        "how are you", "what's up", "tell me a joke", "joke", "hi ", "hello",
        "hey ", "good morning", "good night", "i love you", "you're cute",
        "sing a song", "roast me", "be my friend",
    ]
    identity = ["who created you", "what are you", "are you human", "are you an ai", "who made you"]
    q = query.lower().strip()
    if any(phrase in q for phrase in chit_chat):
        return False
    if any(phrase in q for phrase in identity):
        return True
    if len(q) < 5:
        return False
    # Coding is always allowed
    coding_terms = ["code", "python", "javascript", "function", "bug", "error",
                    "algorithm", "program", "script", "debug", "compile", "syntax"]
    if any(t in q for t in coding_terms):
        return True
    client, _, fast = _get_client()
    if not client:
        return True
    try:
        resp = client.chat.completions.create(
            model=fast,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT + query}],
            max_tokens=5, temperature=0,
        )
        return "RESEARCH" in resp.choices[0].message.content.strip().upper()
    except Exception:
        return True


def _decompose_query(query: str, history: list = None) -> list[str]:
    """Break query into sub-queries for parallel search."""
    client, _, fast_model = _get_client()
    if not client:
        return [query]
        
    prompt_text = DECOMPOSE_PROMPT + f"\nCRITICAL: The current year is {datetime.datetime.now().year}. If the query is about current events, news, or time-sensitive topics, implicitly append '{datetime.datetime.now().year}' to your sub-queries to get the latest information."
    if history:
        # Add the last 2 messages for context so the LLM knows what "it" or "they" refers to
        context = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in history[-2:]])
        prompt_text += f"\n\nContext:\n{context}\n\nCurrent Query: {query}"
    else:
        prompt_text += "\n\nCurrent Query: " + query

    try:
        resp = client.chat.completions.create(
            model=fast_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=200,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("Decompose error:", e)
    return [query, query + " overview", query + " recent developments"]


def _needs_search(query: str, history: list = None) -> bool:
    """Decide if a query requires live web search or if it's a simple/personal question."""
    client, _, fast_model = _get_client()
    if not client:
        return True
        
    prompt = (
        "Does the following user query require searching the live internet for facts, "
        "news, research, or recent events? Reply ONLY with 'YES' or 'NO'.\n"
        "If it is a personal question ('how are you'), a simple programming question ('teach me python'), "
        "or general knowledge that an AI already knows perfectly, reply 'NO'.\n\n"
        f"Query: {query}"
    )
    
    try:
        resp = client.chat.completions.create(
            model=fast_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        return "YES" in resp.choices[0].message.content.strip().upper()
    except Exception:
        return True


def _search_one(query: str) -> list[dict]:
    """Perform a single web search with fallback strategies."""
    query_lower = query.lower()
    is_academic = any(w in query_lower for w in ["arxiv", "paper", "pubmed", "scholar", "study", "journal", "research"])
    
    if is_academic:
        # 1a. Semantic Scholar (Very High Quality)
        try:
            import requests
            ss_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3&fields=title,url,abstract,authors,year,citationCount"
            r_ss = requests.get(ss_url, timeout=10)
            if r_ss.status_code == 200:
                ss_data = r_ss.json().get("data", [])
                results = []
                for paper in ss_data:
                    title = paper.get("title")
                    abstract = paper.get("abstract") or "No abstract available."
                    url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                    authors = ", ".join([a.get("name") for a in paper.get("authors", [])])
                    year = paper.get("year", "n.d.")
                    citation = f"{authors} ({year}). {title}."
                    results.append({
                        "title": title, 
                        "url": url, 
                        "body": abstract,
                        "citation": citation,
                        "meta": f"Citations: {paper.get('citationCount', 0)}"
                    })
                if results: return results
        except Exception as e_ss:
            print(f"[Search] Semantic Scholar failed: {e_ss}")

        # 1b. Try Google Scholar via scholarly
        try:
            from scholarly import scholarly as _scholarly
            results = []
            for pub in _scholarly.search_pubs(query):
                bib = pub.get("bib", {})
                title = bib.get("title", "")
                abstract = bib.get("abstract", "")
                pub_url = pub.get("pub_url", "")
                author = bib.get("author", ["Unknown"])[0]
                year = bib.get("pub_year", "n.d.")
                if title and abstract:
                    results.append({
                        "title": title, 
                        "url": pub_url, 
                        "body": abstract,
                        "citation": f"{author} ({year}). {title}."
                    })
                if len(results) >= 3:
                    break
            if results:
                return results
        except Exception as e_scholar:
            print(f"[Search] Google Scholar failed: {e_scholar}")

        # 1b. ArXiv / PubMed
        try:
            from langchain_community.utilities import ArxivAPIWrapper, PubMedAPIWrapper
            results = []
            if "pubmed" in query_lower or "medical" in query_lower or "health" in query_lower or "biology" in query_lower:
                pubmed = PubMedAPIWrapper(top_k_results=3)
                res = pubmed.run(query)
                if res and "No good PubMed Result" not in res:
                    results.append({"title": f"PubMed Results for '{query}'", "url": "https://pubmed.ncbi.nlm.nih.gov/", "body": res})
            
            if not results or any(w in query_lower for w in ["arxiv", "physics", "math", "computer science", "paper"]):
                arxiv = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
                res = arxiv.run(query)
                if res and "No good Arxiv Result" not in res:
                    results.append({"title": f"ArXiv Results for '{query}'", "url": "https://arxiv.org/", "body": res})
            
            if results:
                return results
        except Exception as e_acad:
            print(f"[Search] Academic APIs failed: {e_acad}")

    # Strategy 1: Direct DuckDuckGo HTML Scraper (bulletproof, no API rate limits)
    try:
        import requests, urllib.parse
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://html.duckduckgo.com/html/"
        r = requests.post(url, headers=headers, data={'q': query}, timeout=8)
        
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            results = []
            for result in soup.find_all('div', class_='result'):
                title_a = result.find('a', class_='result__a')
                snippet_div = result.find('a', class_='result__snippet')
                if title_a and snippet_div:
                    results.append({
                        "title": title_a.text.strip(),
                        "url": title_a.get('href', ''),
                        "body": snippet_div.text.strip()
                    })
                if len(results) >= 5:
                    break
            
            if results:
                return results
    except Exception as e1:
        print(f"[Search] HTML Scraper failed: {e1}")

    # Strategy 2: DuckDuckGo HTML API via requests
    try:
        import requests, urllib.parse
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"}
        r = requests.get("https://api.duckduckgo.com/", params=params,
                         headers=headers, timeout=8)
        data = r.json()
        results = []
        # Instant answer
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "url":   data.get("AbstractURL", ""),
                "body":  data.get("AbstractText", ""),
            })
        # Related topics
        for topic in data.get("RelatedTopics", [])[:6]:
            if topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url":   topic.get("FirstURL", ""),
                    "body":  topic.get("Text", ""),
                })
        if results:
            return results
    except Exception as e2:
        print(f"[Search] DDG instant API failed: {e2}")

    # Strategy 3: Wikipedia search as reliable last resort
    try:
        import requests, urllib.parse
        encoded = urllib.parse.quote(query)
        r = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded}&format=json&srlimit=5",
            timeout=8
        )
        data = r.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            results.append({
                "title": title,
                "url":   f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                "body":  snippet,
            })
        if results:
            return results
    except Exception as e3:
        print(f"[Search] Wikipedia fallback failed: {e3}")

    return [{"title": "Search unavailable", "url": "", "body": "Could not retrieve search results. Answering from knowledge."}]



def _search_images(query: str, max_images: int = 4) -> list[dict]:
    """Search for relevant images via DuckDuckGo with Wikipedia fallback."""
    images = []
    # Strategy 1: DDGS images
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.images(
                query + " english",
                max_results=max_images,
                safesearch="moderate",
            ):
                if r.get("image") and r.get("title"):
                    images.append({
                        "url":    r["image"],
                        "title":  r.get("title", ""),
                        "source": r.get("url", ""),
                    })
        if images:
            return images
    except Exception as e:
        print(f"[Image Search] DDGS failed: {e}")

    # Strategy 2: Wikipedia images
    try:
        import requests, urllib.parse
        encoded = urllib.parse.quote(query)
        r = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages"
            f"&format=json&piprop=original&titles={encoded}",
            timeout=8
        )
        pages = r.json().get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1" and "original" in page_data:
                images.append({
                    "url": page_data["original"]["source"],
                    "title": page_data.get("title", query),
                    "source": f"https://en.wikipedia.org/wiki/{encoded}"
                })
    except Exception as e:
        print(f"[Image Search] Wikipedia failed: {e}")

    return images


def _wants_images(query: str) -> bool:
    """Detect if query would benefit from images."""
    triggers = [
        "show", "image", "picture", "photo", "diagram", "chart", "map",
        "what does", "what do", "look like", "visual", "illustration",
    ]
    return any(t in query.lower() for t in triggers)


def _wants_pdf(query: str) -> bool:
    """Detect if user wants a PDF export."""
    triggers = ["pdf", "download", "export", "report", "file", "save as", "document"]
    return any(t in query.lower() for t in triggers)


def _parallel_search(queries: list[str]) -> list[dict]:
    """Run multiple searches in parallel and deduplicate results."""
    all_results = []
    seen_urls = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_search_one, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            for r in fut.result():
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    return all_results[:12]  # Cap at 12 sources


def _format_sources(results: list[dict]) -> str:
    """Format search results for injection into prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{r['body']}\n")
    return "\n".join(lines)


def _is_research_query(query: str, history: list) -> str:
    """Classify the user's intent."""
    client, _, fast = _get_client()
    if not client: return "RESEARCH"
    try:
        resp = client.chat.completions.create(
            model=fast,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT + query}],
            max_tokens=10
        )
        return resp.choices[0].message.content.strip().upper()
    except: return "RESEARCH"

def _ai_simple_reply(query: str, mode: str) -> str:
    """Fast, concise reply for greetings or simple facts."""
    client, _, fast = _get_client()
    if not client: return "Hello! How can I help with your research today?"
    
    system = "You are Resynth Research. Be extremely concise."
    if mode == "GREETING":
        system += " Respond with a warm, short greeting and ask how you can help with research."
    else:
        system += " Provide a direct, factual answer in 1-2 sentences. No fluff."
        
    try:
        resp = client.chat.completions.create(
            model=fast,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except: return "I'm here to help with your research. What would you like to explore?"


def stream_research(session_id: str, query: str, jwt_token=None, user_id=None, chat_id=None, deep_dive=False, sb=None, session_doc=None, lit_review=False, image_data=None, **kwargs):
    """SSE generator for deep research queries."""
    client, primary_model, _ = _get_client()
    if not client:
        yield f"data: {json.dumps({'type': 'error', 'text': 'No AI client. Set DASHSCOPE_API_KEY or GROQ_API_KEY.'})}\n\n"
        return

    history = _conversations.get(session_id, [])

    # ── Step 0: Visual Research (Vision Analysis) ──────────────────────────
    vision_context = ""
    if image_data:
        yield f'data: {json.dumps({"type": "status", "id": "vision", "text": "👁️ Analyzing your image for research context..."})}\n\n'
        try:
            # Use Qwen-VL or similar via Dashscope
            resp = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in detail for a research assistant. Identify any charts, text, or objects that might be relevant for a web search."},
                        {"type": "image_url", "image_url": {"url": image_data if image_data.startswith("http") else f"data:image/jpeg;base64,{image_data}"}}
                    ]}
                ],
                max_tokens=300
            )
            vision_context = f"\n\n[IMAGE ANALYSIS]: {resp.choices[0].message.content}\n"
            query = f"{query}\n(Visual Context: {vision_context})"
        except Exception as e:
            print("Vision error:", e)

    # ── Detect and fetch URLs or ArXiv IDs embedded in the query ──────────────────
    urls_in_query = URL_PATTERN.findall(query)
    # Also detect plain ArXiv IDs like 2401.00001
    arxiv_ids = _re.findall(r'\b\d{4}\.\d{4,5}(?:v\d+)?\b', query)
    
    fetched_url_context = ""
    if urls_in_query or arxiv_ids:
        status_msg = "\U0001f517 Fetching linked academic source(s)..."
        yield f'data: {json.dumps({"type": "status", "id": "urls", "text": status_msg})}\n\n'
        
        # Handle plain ArXiv IDs by turning them into URLs
        for aid in arxiv_ids[:2]:
            urls_in_query.append(f"https://arxiv.org/abs/{aid}")

        for u in list(dict.fromkeys(urls_in_query))[:3]:  # dedupe + cap at 3
            content = _fetch_url_content(u)
            if content and not content.startswith("[Could not"):
                fetched_url_context += f"\n\n[FETCHED SOURCE: {u}]\n{content}"

    # ── Guard: Intent Detection ──────────────────────────────────────────────
    intent = _is_research_query(query, history)
    if intent in ("GREETING", "GENERAL"):
        msg = _ai_simple_reply(query, intent)
        # Avoid repetition
        if history and history[-1]['content'].strip() == msg.strip():
            msg = "How else can I assist your research today?"
            
        yield f"data: {json.dumps({'type': 'token', 'text': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Save to Supabase even if off-topic
        if jwt_token and user_id:
            import requests
            headers = {
                "apikey": os.environ.get("SUPABASE_KEY"),
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
            if not headers["Authorization"]:
                # If we couldn't extract the token from auth_sb, we can't reliably insert bypassing RLS.
                pass
                
            if headers["Authorization"] and not chat_id:
                try:
                    title = query[:40] + ("..." if len(query) > 40 else "")
                    url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/chats"
                    r = requests.post(url, headers=headers, json={"user_id": user_id, "title": title})
                    if r.status_code in (200, 201) and r.json():
                        chat_id = r.json()[0]["id"]
                        yield f"data: {json.dumps({'type': 'chat_id', 'id': chat_id})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to insert chat REST: {r.text}'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Create chat error: {str(e)}'})}\n\n"
            
            if headers["Authorization"] and chat_id:
                try:
                    url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/messages"
                    r2 = requests.post(url, headers=headers, json=[
                        {"chat_id": chat_id, "role": "user", "content": query},
                        {"chat_id": chat_id, "role": "assistant", "content": msg},
                    ])
                    if r2.status_code not in (200, 201):
                        yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to insert msgs REST: {r2.text}'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Save message error: {str(e)}'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # Decide if we need to search
    should_search = deep_dive or lit_review or _needs_search(query, history)
    results = []

    if should_search:
        # Step 1: Decompose
        if lit_review:
            yield f"data: {json.dumps({'type': 'status', 'id': 'plan', 'text': '📚 Lit Review mode active. Building scholarly search plan...'})}\n\n"
        elif deep_dive:
            yield f"data: {json.dumps({'type': 'status', 'id': 'plan', 'text': '🌊 Deep Dive enabled. Analyzing research plan...'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'id': 'plan', 'text': '🔬 Analyzing your research query...'})}\n\n"
            
        sub_queries = _decompose_query(query, history)
        if lit_review:
            # Force more academic sub-queries
            sub_queries = [q + " literature review papers" for q in sub_queries]
            
        search_msg = f"🔍 Searching {len(sub_queries)} angles in parallel..."
        yield f'data: {json.dumps({"type": "status", "id": "search", "text": search_msg})}\n\n'

        # Step 2: Parallel search
        results = _parallel_search(sub_queries)
        yield f"data: {json.dumps({'type': 'sources', 'sources': results})}\n\n"
        sources_msg = f"📖 Found {len(results)} sources. Extracting key insights..."
        yield f'data: {json.dumps({"type": "status", "id": "sources", "text": sources_msg})}\n\n'
    else:
        yield f"data: {json.dumps({'type': 'status', 'id': 'plan', 'text': '🧠 Answering from knowledge base...'})}\n\n"

    # Step 3: Check RAG documents
    rag_context = ""
    if sb and user_id:
        try:
            from rag_pipeline import retrieve_relevant_chunks
            chunks = retrieve_relevant_chunks(query, user_id, sb)
            if chunks:
                rag_context = "\n\n[USER DOCUMENTS — RAG]\n" + "\n---\n".join(chunks)
                rag_msg = '📁 Cross-referencing your personal document library...'
                yield f'data: {json.dumps({"type": "status", "id": "rag", "text": rag_msg})}\n\n'
        except Exception as e:
            print("RAG retrieve error:", e)

    # Session document (uploaded this session — kept for follow-ups)
    session_doc_context = ""
    if session_doc:
        session_doc_context = f"\n\n[ATTACHED DOCUMENT — USER UPLOADED THIS SESSION]\n{session_doc[:12000]}"
        doc_msg = '📎 Analyzing your attached paper...'
        yield f'data: {json.dumps({"type": "status", "id": "doc", "text": doc_msg})}\n\n'

    # Step 4: Synthesis ──────────────────────────────────────────────────
    citation_style = kwargs.get('citation_style', 'inline')
    strategy = kwargs.get('strategy', 'balanced')
    is_debate = kwargs.get('debate', False)

    source_text = _format_sources(results)
    augmented = (
        (f"[WEB SEARCH RESULTS — {datetime.date.today()}]\n{source_text}\n\n" if source_text.strip() else "")
        + fetched_url_context
        + rag_context
        + session_doc_context
        + f"\n\nResearch query: {query}"
    )

    synth_msg = '🧠 Synthesizing comprehensive research report...' if not lit_review else '📚 Synthesizing formal literature review...'
    if is_debate: synth_msg = '⚔️ Engaging Multi-Agent Debate Mode...'
    yield f"data: {json.dumps({'type': 'status', 'id': 'synth', 'text': synth_msg})}\n\n"

    base_prompt = LIT_REVIEW_PROMPT if lit_review else RESEARCH_SYSTEM_PROMPT
    
    style_instr = f"\n\nCITATION STYLE: Use {citation_style.upper()} format for all citations." if citation_style != 'inline' else ""
    verbosity_instruction = "\n\nBE EXTREMELY CONCISE: Provide a direct answer in 1-3 paragraphs. No fluff." if not deep_dive and not lit_review else "\n\nDEEP DIVE MODE: Provide an exhaustive, detailed, and comprehensive scholarly report."
    
    dynamic_system_prompt = base_prompt + verbosity_instruction + style_instr + f"\n\nCRITICAL CONTEXT:\nThe current date and time is {datetime.datetime.now().strftime('%A, %B %d, %Y %H:%M')}."

    if is_debate:
        # DEBATE LOGIC: Run two perspectives
        perspectives = ["PRO/SUPPORTIVE Perspective", "CON/CRITICAL Perspective"]
        for p in perspectives:
            debate_header = f"\n\n### {p}\n"
            yield f"data: {json.dumps({'type': 'token', 'text': debate_header})}\n\n"
            p_prompt = dynamic_system_prompt + f"\n\nYou are representing the {p}. Focus on arguments and evidence supporting this specific side."
            messages = [{"role": "system", "content": p_prompt}, {"role": "user", "content": augmented}]
            
            stream = client.chat.completions.create(model=primary_model, messages=messages, max_tokens=2000, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk.choices[0].delta.content})}\n\n"
        return

    messages = [{"role": "system", "content": dynamic_system_prompt}]
    messages += history
    messages.append({"role": "user", "content": augmented})

    full_response = ""
    try:
        stream = client.chat.completions.create(
            model=primary_model,
            messages=messages,
            max_tokens=4096,
            stream=True,
            temperature=0.4,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'text': repr(exc)})}\n\n"
        return

    # ── Step 4: Fact-Check & Verification ───────────────────────────────────
    if results and len(full_response) > 200:
        yield f'data: {json.dumps({"type": "status", "id": "verify", "text": "🛡️ Cross-verifying claims with source data..."})}\n\n'
        try:
            audit_prompt = FACT_CHECK_PROMPT.format(report=full_response[:4000], sources=source_text[:6000])
            audit_resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": audit_prompt}],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            audit_data = json.loads(audit_resp.choices[0].message.content)
            yield f"data: {json.dumps({'type': 'verification', 'data': audit_data})}\n\n"
        except: pass

    # Persist history (store plain query, not augmented)
    entry = _conversations.setdefault(session_id, [])
    entry += [{"role": "user", "content": query}, {"role": "assistant", "content": full_response}]
    if len(entry) > 20:
        _conversations[session_id] = entry[-20:]

    # Save to Supabase chat if logged in
    if jwt_token and user_id:
        import requests
        headers = {
            "apikey": os.environ.get("SUPABASE_KEY"),
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        if headers["Authorization"] and not chat_id:
            try:
                title = query[:40] + ("..." if len(query) > 40 else "")
                url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/chats"
                r = requests.post(url, headers=headers, json={"user_id": user_id, "title": title})
                if r.status_code in (200, 201) and r.json():
                    chat_id = r.json()[0]["id"]
                    # Notify frontend of new chat ID so it reloads the history dropdown
                    yield f"data: {json.dumps({'type': 'chat_id', 'id': chat_id})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to insert chat REST: {r.text}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Create chat error: {str(e)}'})}\n\n"

        if headers["Authorization"] and chat_id:
            try:
                url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/messages"
                r2 = requests.post(url, headers=headers, json=[
                    {"chat_id": chat_id, "role": "user", "content": query},
                    {"chat_id": chat_id, "role": "assistant", "content": full_response},
                ])
                if r2.status_code not in (200, 201):
                    yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to insert msgs REST: {r2.text}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Save error: {str(e)}'})}\n\n"

    # ── Step 5: Image search (if requested or visual topic) ──────────────────
    # ── Step 5: Summary & BibTeX ─────────────────────────────────────────────
    bibtex_data = ""
    if results:
        bibtex_parts = []
        for i, res in enumerate(results[:10]):
            safe_title = re.sub(r'[^a-zA-Z0-9]', '', res.get('title', 'paper')[:20])
            bib = f"@article{{resynth_{i}_{safe_title},\n  title={{{res.get('title', 'Unknown Title')}}},\n  author={{Resynth Research Assistant}},\n  url={{{res.get('url', '')}}},\n  journal={{Web Source}},\n  year={{{datetime.now().year}}}\n}}"
            bibtex_parts.append(bib)
        bibtex_data = "\n\n".join(bibtex_parts)
        yield f"data: {json.dumps({'type': 'bibtex', 'bibtex': bibtex_data})}\n\n"

    if is_deep_dive:
        summary_prompt = f"Summarize the following research in exactly 3 bullet points as an 'Executive Summary':\n\n{full_response[:4000]}"
        try:
            sum_resp = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200
            )
            summary_text = sum_resp.choices[0].message.content.strip()
            yield f"data: {json.dumps({'type': 'summary', 'text': summary_text})}\n\n"
        except: pass

    if _wants_images(query):
        img_status = '🖼️ Fetching relevant images...'
        yield f'data: {json.dumps({"type": "status", "text": img_status})}\n\n'
        images = _search_images(query, max_images=4)
        if images:
            yield f"data: {json.dumps({'type': 'images', 'images': images})}\n\n"

    # ── Step 6: PDF generation (if requested) ────────────────────────────────
    if _wants_pdf(query):
        pdf_status = '📄 Generating PDF report...'
        yield f'data: {json.dumps({"type": "status", "text": pdf_status})}\n\n'
        try:
            from pdf_utils import generate_research_pdf
            pdf_url = generate_research_pdf(
                title=query[:80],
                content=full_response,
                sources=results,
            )
            if pdf_url:
                yield f"data: {json.dumps({'type': 'pdf', 'url': pdf_url, 'title': query[:60]})}\n\n"
        except Exception as e:
            print("PDF error:", e)

    # ── Step 7: Related Questions ──────────────────────────────────────────
    try:
        followup_prompt = f"Based on this research report about '{query}', suggest 3 concise follow-up research questions that a user might want to ask next. Return ONLY a JSON array of 3 strings. Example: [\"question 1\", \"question 2\", \"question 3\"]"
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "system", "content": "You are a research assistant. Output ONLY valid JSON array."}, 
                      {"role": "user", "content": followup_prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        import re
        match = re.search(r'\[.*\]', resp.choices[0].message.content.strip(), re.DOTALL)
        if match:
            followups = json.loads(match.group(0))
            yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
    except Exception as e:
        print("Follow-up error:", e)

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
