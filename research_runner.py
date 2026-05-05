"""research_runner.py — Deep research engine for Xynth.

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

RESEARCH_SYSTEM_PROMPT = """You are Xynth Research, an advanced AI research assistant for students and professionals.

STRUCTURE YOUR RESPONSE DYNAMICALLY:
Do NOT follow a rigid template. Instead, read the user's query and create a beautifully structured, highly readable response using custom headings (##) that make sense for that specific topic. 
- Use bold text, bullet points, and paragraphs to make it scannable.
- DO NOT use markdown tables unless explicitly asked. Tables are hard to read on mobile. Use bullet points instead.
- If it's a coding question, provide clear explanations and code blocks.
- If it's news, structure it chronologically or by theme.
- Add relevant emojis to your headings to make it engaging.

CITING SOURCES (CRITICAL):
- If you were provided web search results, ALWAYS use them and cite them.
- Cite your sources INLINE using markdown hyperlinks with the source's name. 
  Example: "According to [Reuters](https://...), the market..." or "...grew by 50% ([Bloomberg](https://...))."
- DO NOT use generic numbers like [1] or [2] for citations. Use the actual name/title of the source as the clickable link text.
- NEVER output a "References", "Sources", or "Citations" list at the end of your response. The UI will automatically generate a hidden dropdown for this.

LANGUAGE:
- ALWAYS reply in the exact same language that the user typed their prompt in. If they ask in Spanish, reply in Spanish. If in French, reply in French.

Be academically rigorous, precise, and highly readable."""

DECOMPOSE_PROMPT = """Break this research query into 3 focused sub-queries for comprehensive research.
If the user's query contains typos, misspellings, or bad grammar, automatically correct them in your mind before creating the sub-queries.
Return ONLY a JSON array of 3 strings. Example: ["sub-query 1", "sub-query 2", "sub-query 3"]
Query: """

CLASSIFY_PROMPT = """You are a research tool classifier. Decide if this query is a legitimate research/learning/coding topic.

ALLOWED: Science, history, technology, medicine, law, economics, society, news, coding, programming, math, engineering, academic subjects, how-to learn something, current events, analysis of any topic, debugging help, code explanation.
NOT ALLOWED: Personal chit-chat ("how are you", "tell me a joke"), purely creative requests with no educational value ("write me a love poem"), or requests that are harmful.

Reply with ONLY one word: RESEARCH or OFFTOPIC
Query: """

OFF_TOPIC_RESPONSES = [
    "I'm Xynth Research — built for deep research and learning. Try asking me something like *\"What are the effects of climate change on agriculture?\"* or *\"Explain quantum entanglement\"*.",
    "That's a bit outside my research scope! I'm specialized for academic and factual research. Ask me about any topic — science, history, technology, current events — and I'll find you real sources.",
    "I'm a research assistant, so I'm best at finding and synthesizing information from the web. Try a research question and I'll pull from multiple sources for you! 📚",
]

import random

_conversations: dict = {}


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
        "are you human", "are you an ai", "who created you", "what are you",
        "sing a song", "roast me", "be my friend",
    ]
    q = query.lower().strip()
    if any(phrase in q for phrase in chit_chat):
        return False
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
    """Search with multiple fallback strategies — never silently fails."""

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


def _ai_reject(query: str) -> str:
    """Use AI to craft a personalized, polite rejection referencing the specific query."""
    client, _, fast = _get_client()
    if not client:
        return "I'm Xynth Research, focused on research and learning. Try asking me about science, technology, current events, or any topic you'd like to explore!"
    try:
        resp = client.chat.completions.create(
            model=fast,
            messages=[
                {"role": "system", "content": (
                    "You are Xynth Research, a focused AI research assistant for students and professionals. "
                    "The user asked something that isn't a research or learning topic. "
                    "Write a SHORT, friendly, personalized reply (2-3 sentences max) that: "
                    "1) Acknowledges what they specifically asked, "
                    "2) Explains you're focused on research/learning topics, "
                    "3) Suggests they try a research question instead. "
                    "Be warm but clear. Don't be robotic."
                )},
                {"role": "user", "content": query}
            ],
            max_tokens=120,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "I'm built for research and learning — not quite the right tool for that! Try asking me about a topic you'd like to explore: science, history, technology, current events, coding, and more."


def stream_research(session_id: str, query: str, sb=None, user_id=None, chat_id=None, deep_dive=False):
    """SSE generator for deep research queries."""
    client, primary_model, _ = _get_client()
    if not client:
        yield f"data: {json.dumps({'type': 'error', 'text': 'No AI client. Set DASHSCOPE_API_KEY or GROQ_API_KEY.'})}\n\n"
        return

    history = _conversations.get(session_id, [])

    # ── Guard: reject off-topic — AI crafts personalized reply ───────────────
    if not _is_research_query(query, history):
        yield f"data: {json.dumps({'type': 'status', 'text': '🤔 Thinking...'})}\n\n"
        msg = _ai_reject(query)
        yield f"data: {json.dumps({'type': 'token', 'text': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # Decide if we need to search
    should_search = deep_dive or _needs_search(query, history)
    results = []

    if should_search:
        # Step 1: Decompose
        if deep_dive:
            yield f"data: {json.dumps({'type': 'status', 'text': '🌊 Deep Dive enabled. Analyzing query...'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'text': '🔬 Analyzing your research query...'})}\n\n"
            
        sub_queries = _decompose_query(query, history)
        yield f"data: {json.dumps({'type': 'status', 'text': f'🔍 Searching {len(sub_queries)} angles in parallel...'})}\n\n"

        # Step 2: Parallel search
        results = _parallel_search(sub_queries)
        yield f"data: {json.dumps({'type': 'sources', 'sources': results})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'text': f'📖 Found {len(results)} sources. Synthesizing report...'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'text': '🧠 Answering from knowledge base...'})}\n\n"

    # Step 3: Check RAG documents
    rag_context = ""
    if sb and user_id:
        try:
            from rag_pipeline import retrieve_relevant_chunks
            chunks = retrieve_relevant_chunks(query, user_id, sb)
            if chunks:
                rag_context = "\n\n[USER DOCUMENTS]\n" + "\n---\n".join(chunks)
                yield f"data: {json.dumps({'type': 'status', 'text': '📁 Found relevant content in your documents...'})}\n\n"
        except Exception as e:
            print("RAG retrieve error:", e)

    # Step 4: Build prompt and stream report
    source_text = _format_sources(results)
    augmented = (
        f"[WEB SEARCH RESULTS — {datetime.date.today()}]\n{source_text}"
        + rag_context
        + f"\n\nResearch query: {query}"
    )

    dynamic_system_prompt = RESEARCH_SYSTEM_PROMPT + f"\n\nCRITICAL CONTEXT:\nThe current date and time is {datetime.datetime.now().strftime('%A, %B %d, %Y %H:%M')}. Always assume the present year is {datetime.datetime.now().year} and ensure your answers reflect this timeline."
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
        yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
        return

    # Persist history (store plain query, not augmented)
    entry = _conversations.setdefault(session_id, [])
    entry += [{"role": "user", "content": query}, {"role": "assistant", "content": full_response}]
    if len(entry) > 20:
        _conversations[session_id] = entry[-20:]

    # Save to Supabase chat if logged in
    if sb and user_id:
        if not chat_id:
            try:
                title = query[:40] + ("..." if len(query) > 40 else "")
                res = sb.table("chats").insert({"user_id": user_id, "title": title}).execute()
                if res.data:
                    chat_id = res.data[0]["id"]
                    # Notify frontend of new chat ID so it reloads the history dropdown
                    yield f"data: {json.dumps({'type': 'chat_id', 'id': chat_id})}\n\n"
            except Exception as e:
                print("Create chat error:", e)

        if chat_id:
            try:
                sb.table("messages").insert([
                    {"chat_id": chat_id, "role": "user", "content": query},
                    {"chat_id": chat_id, "role": "assistant", "content": full_response},
                ]).execute()
            except Exception as e:
                print("Save error:", e)

    # ── Step 5: Image search (if requested or visual topic) ──────────────────
    if _wants_images(query):
        yield f"data: {json.dumps({'type': 'status', 'text': '🖼️ Fetching relevant images...'})}\n\n"
        images = _search_images(query, max_images=4)
        if images:
            yield f"data: {json.dumps({'type': 'images', 'images': images})}\n\n"

    # ── Step 6: PDF generation (if requested) ────────────────────────────────
    if _wants_pdf(query):
        yield f"data: {json.dumps({'type': 'status', 'text': '📄 Generating PDF report...'})}\n\n"
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

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
