"""
Reddit voice AI GTM monitor (v2).

Finds conversations about voice AI anywhere on Reddit so a human can join in.
Three sources, all via public RSS (no Reddit API key):
  1. Site-wide Reddit search for voice-AI queries  -> catches threads in subreddits
     we don't track (the "go where voice AI is being talked about" part)
  2. /new feeds of the tracked subreddits (subreddits.txt)
  3. /comments feeds of the tracked subreddits (opt-in per sub with COMMENTS_ENABLED)

Pipeline: fetch -> dedupe -> age filter -> keyword scoring -> LLM classify + draft
-> Slack. A post is only marked "seen" once it has been fully handled, so LLM or
Slack outages never silently drop a lead. Replies are posted by humans only.

Env vars: SLACK_WEBHOOK_URL + GEMINI_API_KEY (free) and/or OPENROUTER_API_KEY (unless DRY_RUN=1)
Optional: OPENROUTER_MODELS (comma list), MIN_CONFIDENCE, MAX_ALERTS_PER_RUN,
          MAX_AGE_HOURS, DRY_RUN, COMMENTS_ENABLED
"""

import calendar
import html
import json
import os
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

import feedparser
import requests

HERE = Path(__file__).parent

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER_AGENT = os.environ.get("USER_AGENT", "voice-ai-gtm-monitor/2.0 (by u/PlayfulTest4733)")

DRY_RUN = os.environ.get("DRY_RUN") == "1"
COMMENTS_ENABLED = os.environ.get("COMMENTS_ENABLED", "1") == "1"
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.75"))
MIN_CONFIDENCE_UNTRACKED = float(os.environ.get("MIN_CONFIDENCE_UNTRACKED", "0.85"))  # subs found via search only
MAX_ALERTS_PER_RUN = int(os.environ.get("MAX_ALERTS_PER_RUN", "20"))
MAX_AGE_HOURS = float(os.environ.get("MAX_AGE_HOURS", "36"))   # ignore stale threads
MAX_LLM_CALLS_PER_RUN = int(os.environ.get("MAX_LLM_CALLS_PER_RUN", "6"))   # LLM requests/run
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "8"))                  # posts per request
LLM_PAUSE_SECONDS = float(os.environ.get("LLM_PAUSE_SECONDS", "7"))          # free tier ~10 req/min

# Tried in order; first that returns usable JSON wins. Override via OPENROUTER_MODELS.
OPENROUTER_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_MODELS",
        "openrouter/free",
    ).split(",") if m.strip()
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # free key: aistudio.google.com/apikey
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()]
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

SEEN_IDS_PATH = HERE / "seen_ids.json"
MAX_SEEN_IDS = 8000

BATCH_SIZE = 7   # subs per combined feed (fewer requests = faster runs)
BATCH_DELAY_SECONDS = float(os.environ.get("BATCH_DELAY_SECONDS", "6"))  # floor between requests

# Site-wide Reddit search queries (each is one request). Quoted phrases = exact match.
SEARCH_QUERIES = [
    '"voice ai" OR "voice agent" OR "voice agents" OR voicebot OR "voice bot"',
    '"ai receptionist" OR "virtual receptionist" OR "ai answering" OR "ai calling" OR "ai caller" OR "ai phone agent" OR "ai cold calling"',
    '"outbound calling" OR "call center automation" OR "call centre automation" OR "phone automation" OR "conversational ai" OR "voice automation"',
    '(IVR OR "call center" OR "call centre") AND (ai OR automation OR replace OR alternative)',
    'vapi OR retell OR "bland ai" OR synthflow OR "elevenlabs agents" OR "air ai" OR hinglish',
]

# Strong = clearly voice-AI. One hit is enough to go to the LLM.
STRONG_KEYWORDS = [
    r"voice ai", r"voice agents?", r"voice ?bots?", r"voice assistants?",
    r"ai receptionists?", r"virtual receptionists?", r"ai answering", r"ai phone",
    r"ai calling", r"ai call(?:er|s|ing)", r"ai cold call\w*", r"ai dialer",
    r"outbound (?:calling|calls|voice)", r"inbound (?:calls|voice)",
    r"conversational ai", r"conversational voice", r"voice automation",
    r"phone automation", r"call (?:cent(?:er|re)) automation", r"call automation",
    r"automated calls?", r"auto[- ]?dialer", r"predictive dialer", r"ivr",
    r"speech[- ]to[- ]text", r"text[- ]to[- ]speech", r"asr", r"tts", r"stt",
    r"vapi", r"retell(?: ai)?", r"bland(?:\.ai| ai)", r"synthflow", r"air ai",
    r"elevenlabs", r"deepgram", r"livekit", r"pipecat", r"vocode", r"play\.?ht",
    r"cartesia", r"assembly ?ai", r"twilio", r"exotel", r"knowlarity", r"ozonetel",
    r"haptik", r"yellow\.ai", r"gnani", r"sarvam", r"uniphore", r"cognigy", r"parloa",
    r"poly\.?ai", r"ringcentral ai", r"goodcall", r"smith\.ai", r"ruby receptionists?",
    r"my ?ai ?front ?desk", r"dialpad ai", r"talkdesk",
]
# Weak = only counts when paired with a context word, to cut generic noise.
WEAK_KEYWORDS = [
    r"call cent(?:er|re)s?", r"contact cent(?:er|re)s?", r"cold call(?:ing|s)?",
    r"missed calls?", r"after[- ]hours calls?", r"answering service", r"receptionist",
    r"telecall\w*", r"telemarketing", r"phone calls?", r"customer support calls?",
    r"collections? calls?", r"lead qualification", r"appointment (?:booking|setting)",
    r"hinglish", r"voice", r"voice clon\w*", r"ai voices?",
]
CONTEXT_WORDS = [
    r"ai", r"automat\w+", r"bot", r"agent", r"gpt", r"llm", r"replace", r"scale",
    r"cost", r"latency", r"api", r"workflow", r"integrat\w+", r"cheap", r"alternative",
    r"recommend\w*", r"tool", r"software", r"platform", r"vendor", r"build(?:ing)?",
]

STRONG_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(STRONG_KEYWORDS) + r")(?![a-z0-9])", re.I)
WEAK_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(WEAK_KEYWORDS) + r")(?![a-z0-9])", re.I)
CONTEXT_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(CONTEXT_WORDS) + r")(?![a-z0-9])", re.I)

# Competitor/neutral tool names we want to flag in Slack (not an exclusion list).
COMPETITORS = ["vapi", "retell", "bland", "synthflow", "air ai", "elevenlabs", "haptik",
               "yellow.ai", "gnani", "uniphore", "cognigy", "parloa", "poly.ai", "smith.ai"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_subreddits() -> list[str]:
    path = HERE / "subreddits.txt"
    subs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            subs.append(line.removeprefix("r/").removeprefix("/r/").strip("/"))
    seen, out = set(), []
    for s in subs:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def load_seen_ids() -> list[str]:
    if SEEN_IDS_PATH.exists():
        try:
            return json.loads(SEEN_IDS_PATH.read_text())
        except json.JSONDecodeError:
            log(f"WARNING: {SEEN_IDS_PATH} was corrupt, starting fresh")
    return []


def save_seen_ids(seen_ids: list[str]) -> None:
    SEEN_IDS_PATH.write_text(json.dumps(seen_ids[-MAX_SEEN_IDS:], indent=1))


TAG_RE = re.compile(r"<[^>]+>")


def clean_html(raw: str) -> str:
    text = TAG_RE.sub(" ", raw or "")
    text = html.unescape(text)
    text = re.sub(r"\s*submitted by\s+/u/\S+.*$", "", text, flags=re.I | re.S)
    text = re.sub(r"\[link\]|\[comments\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Reddit fetching
# ---------------------------------------------------------------------------

SUB_FROM_LINK = re.compile(r"/r/([^/]+)/")
HOST = "www.reddit.com"  # old.reddit.com serves an HTML block page for RSS; .json is 403

_next_ok_at = 0.0  # Reddit's anonymous quota is ~1 request per rate-limit window


def _pace(resp: requests.Response | None) -> None:
    """Schedule the next request after the window Reddit reports in its headers."""
    global _next_ok_at
    wait = 0.0
    if resp is not None:
        try:
            if float(resp.headers.get("x-ratelimit-remaining", 1)) < 1:
                wait = float(resp.headers.get("x-ratelimit-reset", 10)) + 1
        except ValueError:
            wait = 10.0
    _next_ok_at = time.time() + max(wait, BATCH_DELAY_SECONDS)


def http_get_feed(path_and_query: str, label: str) -> bytes | None:
    """GET a reddit RSS path, pacing to Reddit's rate-limit headers and retrying
    429s. Returns None (never raises) on failure or non-XML responses."""
    url = f"https://{HOST}{path_and_query}"
    for attempt in range(4):
        delay = _next_ok_at - time.time()
        if delay > 0:
            time.sleep(delay + random.uniform(0, 1.5))
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        except requests.RequestException as e:
            log(f"WARNING: {label}: request failed: {e}")
            _pace(None)
            continue
        _pace(resp)
        if resp.status_code == 200:
            if resp.content.lstrip()[:5] != b"<?xml":
                log(f"WARNING: {label}: 200 but not XML (blocked?)")
                return None
            return resp.content
        if resp.status_code in (429, 503):
            log(f"WARNING: {label}: HTTP {resp.status_code} (attempt {attempt + 1}/4)")
            continue
        log(f"WARNING: {label}: HTTP {resp.status_code}")
        return None
    log(f"WARNING: {label}: gave up after retries")
    return None


def parse_feed(content: bytes, kind: str) -> list[dict]:
    feed = feedparser.parse(content)
    posts = []
    for e in feed.entries:
        link = e.get("link", "")
        m = SUB_FROM_LINK.search(link)
        ts = e.get("updated_parsed") or e.get("published_parsed")
        posts.append({
            "id": e.get("id", link),
            "kind": kind,  # "post" or "comment"
            "title": clean_html(e.get("title", "")),
            "body": clean_html(e.get("summary", "")),
            "link": link,
            "subreddit": m.group(1) if m else "unknown",
            "author": (e.get("author") or "").removeprefix("/u/"),
            "created": calendar.timegm(ts) if ts else None,
        })
    return posts


def fetch_search(query: str) -> list[dict]:
    q = urllib.parse.quote_plus(query)
    content = http_get_feed(f"/search.rss?q={q}&sort=new&t=day&limit=100", f"search {query[:40]}")
    return parse_feed(content, "post") if content else []


def fetch_sub_batch(subs: list[str], feed: str, kind: str) -> list[dict]:
    combined = "+".join(urllib.parse.quote(s) for s in subs)
    content = http_get_feed(f"/r/{combined}/{feed}.rss?limit=100", f"{feed} {subs}")
    return parse_feed(content, kind) if content else []


def fetch_everything(subreddits: list[str]) -> list[dict]:
    jobs: list[tuple] = []
    for q in SEARCH_QUERIES:
        jobs.append(("search", q))
    batches = [subreddits[i:i + BATCH_SIZE] for i in range(0, len(subreddits), BATCH_SIZE)]
    for b in batches:
        jobs.append(("new", b))
    if COMMENTS_ENABLED:
        for b in batches:
            jobs.append(("comments", b))

    out: list[dict] = []
    for i, job in enumerate(jobs):
        if job[0] == "search":
            out.extend(fetch_search(job[1]))
        elif job[0] == "new":
            out.extend(fetch_sub_batch(job[1], "new", "post"))
        else:
            out.extend(fetch_sub_batch(job[1], "comments", "comment"))
    return out


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def keyword_score(item: dict) -> int:
    """0 = drop. 2 = strong keyword hit. 1 = weak keyword + context word."""
    text = f"{item['title']} {item['body']}"
    if STRONG_RE.search(text):
        return 2
    if WEAK_RE.search(text) and CONTEXT_RE.search(text):
        return 1
    return 0


def competitors_in(text: str) -> list[str]:
    low = text.lower()
    return [c for c in COMPETITORS if re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", low)]


# ---------------------------------------------------------------------------
# LLM classification + reply drafting
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are a GTM analyst for Vaani, a voice AI company. Vaani builds \
AI voice agents that handle phone calls: inbound support, outbound calling (sales, \
collections, reminders, lead qualification), AI receptionists and IVR replacement. \
Vaani is strong in Indian languages and code-mixed speech (Hinglish and other \
regional mixes) and in India-market use cases (BFSI, real estate, healthcare, \
e-commerce, D2C), but sells globally.

Decide whether this Reddit {kind} is a place where a knowledgeable person from Vaani \
could jump in and genuinely help.

Mark relevant=true ONLY if there is real conversation about: someone looking for / \
comparing / building / struggling with voice AI, AI calling, voice agents, IVR, AI \
receptionists, call-center automation, or speech tech for phone calls - or a pain \
point voice AI directly solves (missed calls, expensive call centers, slow lead \
follow-up, multilingual phone support).

The topic must be voice AI for real-time CONVERSATIONS (phone calls, voice agents, IVR, receptionists, call centers) or the speech tech (ASR/TTS/telephony) used to build them. Voice AI in music, memes, gaming, celebrity voice filters, deepfakes, audiobooks, dubbing, or personal-assistant gadgets is NOT relevant, even if "AI voice" is mentioned.

Mark relevant=false for: news/hype with no discussion hook, job postings, pure \
self-promo or spam, homework, unrelated uses of the words ("voice" as opinion, \
"call" as in a meeting, text chatbots with no voice/phone angle), threads that are \
old/locked-looking, or where a reply would clearly be off-topic.

Pick intent from: "asking_for_tool" (wants a recommendation), "pain_point", \
"building" (technical build question), "comparing" (evaluating vendors), \
"discussion" (general talk), "competitor_mention", "other".
Pick lead_quality: "hot" (actively seeking/evaluating, buying signal), "warm" \
(clear pain or active discussion), "cold" (tangential, brand visibility only).

Reply draft rules (reply_draft, only when relevant):
- Sound like a practitioner who has actually shipped this. Concrete, specific to \
THEIR situation, 2-5 sentences, plain text, no emojis, no bullet lists, no hype.
- Add real value first: a practical tip, a trade-off, a number, a pitfall (latency, \
interruptions, accents/languages, telephony, compliance/consent, cost per minute).
- Do NOT name Vaani or any product, and do not include links. A human will decide \
whether/how to disclose affiliation.
- If the thread is about Indian languages / Hinglish / regional accents, say \
specifically what tends to break and how to handle it.
- If nothing useful can be said without selling, set reply_draft to null.

Context:
Type: {kind}
Subreddit: r/{subreddit}
Title: {title}
Text: {body}

Respond with ONLY a JSON object, no markdown fences:
{{"relevant": true|false, "confidence": 0.0-1.0, "intent": "...", "lead_quality": "hot|warm|cold", \
"reason": "one short sentence", "reply_draft": "..." or null}}
"""


def _parse_json_leniently(text: str):
    """Parse a JSON object or array out of model output (fences / stray text ok)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _llm_json(prompt: str, validate) -> object | None:
    """Try Gemini free models, then OpenRouter free models. `validate(parsed)`
    must return True for an acceptable answer. None if every model failed."""
    def accept(content: str, who: str):
        parsed = _parse_json_leniently(content)
        if parsed is not None and validate(parsed):
            return parsed
        log(f"WARNING: {who} output not usable: {str(content)[:150]!r}")
        return None

    if GEMINI_API_KEY:
        for model in GEMINI_MODELS:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"temperature": 0.2,
                                               "responseMimeType": "application/json"}},
                    timeout=90,
                )
            except requests.RequestException as e:
                log(f"WARNING: Gemini {model} request failed: {e}")
                continue
            if resp.status_code != 200:
                log(f"WARNING: Gemini {model} HTTP {resp.status_code}: {resp.text[:200]!r}")
                continue   # quota exhausted on this model -> next model has its own quota
            try:
                content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, ValueError, TypeError):
                log(f"WARNING: Gemini {model} unexpected response shape")
                continue
            got = accept(content, f"Gemini {model}")
            if got is not None:
                return got

    if OPENROUTER_API_KEY:
        for model in OPENROUTER_MODELS:
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": model, "temperature": 0.2,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=90,
                )
            except requests.RequestException as e:
                log(f"WARNING: OpenRouter {model} request failed: {e}")
                continue
            if resp.status_code != 200:
                log(f"WARNING: OpenRouter {model} HTTP {resp.status_code}: {resp.text[:200]!r}")
                continue
            try:
                content = resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            got = accept(content, f"OpenRouter {model}")
            if got is not None:
                return got
    return None


BATCH_RULES = CLASSIFY_PROMPT.split("Context:")[0].replace(
    "Decide whether this Reddit {kind} is", "Decide, for EACH item below, whether that Reddit post/comment is")


def classify_batch(items: list[dict]) -> list[dict | None]:
    """One LLM call for several items (saves free-tier quota). Returns a result
    per item, or None for items the model failed to answer (retried next run)."""
    listing = []
    for n, it in enumerate(items):
        listing.append(f"--- ITEM {n} ---\nType: {it['kind']}\nSubreddit: r/{it['subreddit']}\n"
                       f"Title: {it['title'][:250]}\nText: {(it['body'] or '(no body text)')[:1200]}")
    prompt = (BATCH_RULES + "\n" + "\n".join(listing) +
              "\n\nRespond with ONLY a JSON array with exactly one object per item, in order, "
              "no markdown fences. Each object: "
              '{"item": <number>, "relevant": true|false, "confidence": 0.0-1.0, "intent": "...", '
              '"lead_quality": "hot|warm|cold", "reason": "one short sentence", "reply_draft": "..." or null}')

    def ok(parsed):
        return isinstance(parsed, list) and all(isinstance(x, dict) and "relevant" in x for x in parsed)

    parsed = _llm_json(prompt, ok)
    out: list[dict | None] = [None] * len(items)
    if not parsed:
        return out
    for pos, r in enumerate(parsed):
        idx = r.get("item", pos)
        if isinstance(idx, int) and 0 <= idx < len(items):
            out[idx] = r
    return out


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

LEAD_EMOJI = {"hot": ":fire:", "warm": ":sunny:", "cold": ":snowflake:"}


def _age(created: int | None) -> str:
    if not created:
        return "unknown age"
    mins = max(0, int((time.time() - created) / 60))
    return f"{mins}m ago" if mins < 90 else f"{mins // 60}h ago"


def send_slack_alert(item: dict, res: dict, rivals: list[str]) -> bool:
    lead = str(res.get("lead_quality", "warm")).lower()
    conf = float(res.get("confidence", 0) or 0)
    kind = "Comment" if item["kind"] == "comment" else "Post"
    header = (f"{LEAD_EMOJI.get(lead, ':sunny:')} *{lead.upper()}* {kind} in "
              f"r/{item['subreddit']} · {res.get('intent', 'other')} · "
              f"conf {conf:.2f} · {_age(item['created'])}")
    body = item["body"][:500] + ("…" if len(item["body"]) > 500 else "")
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": f"*<{item['link']}|{item['title'][:200]}>*\n{body or '_no body_'}"}},
        {"type": "context", "elements": [{"type": "mrkdwn",
            "text": f"_Why:_ {res.get('reason', '-')}"
                    + (f" · _Tools mentioned:_ {', '.join(rivals)}" if rivals else "")}]},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": f"*Draft reply (edit before posting):*\n```{res.get('reply_draft') or 'no draft'}```"}},
        {"type": "divider"},
    ]
    payload = {"text": f"{lead.upper()} voice AI lead in r/{item['subreddit']}: {item['title'][:100]}",
               "blocks": blocks}
    if DRY_RUN:
        print(json.dumps(payload, indent=1, ensure_ascii=False))
        return True
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=15)
        if r.status_code == 200:
            return True
        log(f"WARNING: Slack HTTP {r.status_code}: {r.text[:200]!r}")
    except requests.RequestException as e:
        log(f"WARNING: Slack request failed: {e}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DRY_RUN and not ((GEMINI_API_KEY or OPENROUTER_API_KEY) and SLACK_WEBHOOK_URL):
        log("ERROR: SLACK_WEBHOOK_URL and GEMINI_API_KEY (or OPENROUTER_API_KEY) must be set, or DRY_RUN=1")
        sys.exit(1)

    subreddits = load_subreddits()
    tracked = {x.lower() for x in subreddits}
    seen_list = load_seen_ids()
    seen = set(seen_list)
    log(f"Tracking {len(subreddits)} subreddits, {len(SEARCH_QUERIES)} search queries, "
        f"comments={'on' if COMMENTS_ENABLED else 'off'}")

    items = fetch_everything(subreddits)
    total_fetched = len(items)

    # dedupe within this run (search + subreddit feeds overlap), drop already handled
    by_id: dict[str, dict] = {}
    for it in items:
        if it["id"] and it["id"] not in seen and it["id"] not in by_id:
            by_id[it["id"]] = it
    # cross-posts: same author + same title in several subs -> keep the newest only
    uniq: dict[tuple, dict] = {}
    for it in by_id.values():
        key = (it["kind"], it["author"], it["title"].lower()) if it["kind"] == "post" else (it["id"],)
        if key not in uniq or (it["created"] or 0) > (uniq[key]["created"] or 0):
            uniq[key] = it
    fresh = list(uniq.values())
    dup_ids = [i for i in by_id if i not in {x["id"] for x in fresh}]

    now = time.time()
    stats = dict(fetched=total_fetched, new=len(fresh), too_old=0, no_kw=0,
                 llm=0, irrelevant=0, low_conf=0, alerts=0, llm_failed=0, deferred=0)
    handled: list[str] = list(dup_ids)

    # newest first so a rate-limited run still serves the freshest leads
    fresh.sort(key=lambda x: x["created"] or 0, reverse=True)
    candidates: list[dict] = []
    for it in fresh:
        if it["created"] and now - it["created"] > MAX_AGE_HOURS * 3600:
            stats["too_old"] += 1
            handled.append(it["id"])
        elif keyword_score(it) == 0:
            stats["no_kw"] += 1
            handled.append(it["id"])
        else:
            candidates.append(it)

    if DRY_RUN and not (GEMINI_API_KEY or OPENROUTER_API_KEY):
        for it in candidates:
            print(f"[dry-run, no LLM] would classify: r/{it['subreddit']} [{it['kind']}] {it['title'][:90]}")
        log(f"{len(candidates)} candidates would go to the LLM in "
            f"{-(-len(candidates) // LLM_BATCH_SIZE)} request(s)")
        return

    # strong keyword hits first, then newest
    candidates.sort(key=lambda x: (keyword_score(x), x["created"] or 0), reverse=True)
    max_items = MAX_LLM_CALLS_PER_RUN * LLM_BATCH_SIZE
    stats["deferred"] = max(0, len(candidates) - max_items)  # not marked seen -> next run
    candidates = candidates[:max_items]

    for i in range(0, len(candidates), LLM_BATCH_SIZE):
        if stats["alerts"] >= MAX_ALERTS_PER_RUN:
            stats["deferred"] += len(candidates) - i
            break
        chunk = candidates[i:i + LLM_BATCH_SIZE]
        stats["llm"] += 1
        results = classify_batch(chunk)
        time.sleep(LLM_PAUSE_SECONDS)
        for it, res in zip(chunk, results):
            if res is None:
                stats["llm_failed"] += 1      # retry next run
                continue
            try:
                conf = float(res.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            if not res.get("relevant"):
                stats["irrelevant"] += 1
                handled.append(it["id"])
            elif conf < (MIN_CONFIDENCE if it["subreddit"].lower() in tracked else MIN_CONFIDENCE_UNTRACKED):
                stats["low_conf"] += 1
                handled.append(it["id"])
            elif send_slack_alert(it, res, competitors_in(f"{it['title']} {it['body']}")):
                stats["alerts"] += 1
                handled.append(it["id"])
            # Slack failure -> not marked seen, retried next run

    if not DRY_RUN:
        save_seen_ids(seen_list + handled)

    log("Run complete: " + ", ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
