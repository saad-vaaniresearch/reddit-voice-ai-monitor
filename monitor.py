"""
Reddit voice AI GTM monitor.

Polls a fixed list of subreddits' /new.rss feeds (no Reddit API key needed),
keyword-filters new posts, sends survivors to an OpenRouter model for a
relevance check + draft reply, and posts genuine hits to Slack.

Run on a schedule (see .github/workflows/monitor.yml). State (which post
IDs we've already alerted on) is kept in seen_ids.json, committed back to
the repo by the workflow after each run.
"""

import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBREDDITS = [
    "CustomerService",
    "callcentres",
    "sysadmin",
    "SaaS",
    "Entrepreneur",
    "smallbusiness",
    "startups",
    "voiceai",
    "VoiceAutomationAI",
    "AI_Agents",
    "indianstartups",
    "AIReceptionists",
    "StartUpIndia",
]

# Cheap pre-filter. Case-insensitive substring match against title + summary.
KEYWORDS = [
    "voice ai",
    "voice agent",
    "voicebot",
    "voice bot",
    "ivr",
    "ai receptionist",
    "ai calling",
    "outbound calling",
    "call center automation",
    "call centre automation",
    "conversational ai",
    "phone automation",
    "voice assistant",
    "text to speech",
    "speech to text",
    "tts",
    "stt",
]

REDDIT_LIMIT = 100  # max items per feed request
USER_AGENT = "voice-ai-gtm-monitor/1.0 (by u/PlayfulTest4733)"

OPENROUTER_MODEL = "inclusionai/ling-3.0-flash:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SEEN_IDS_PATH = Path(__file__).parent / "seen_ids.json"
MAX_SEEN_IDS = 5000  # trim oldest once we exceed this, so the file doesn't grow forever

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_seen_ids() -> list[str]:
    if SEEN_IDS_PATH.exists():
        try:
            return json.loads(SEEN_IDS_PATH.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {SEEN_IDS_PATH} was corrupt, starting fresh", file=sys.stderr)
    return []


def save_seen_ids(seen_ids: list[str]) -> None:
    # Keep only the most recent MAX_SEEN_IDS to bound file growth.
    trimmed = seen_ids[-MAX_SEEN_IDS:]
    SEEN_IDS_PATH.write_text(json.dumps(trimmed, indent=2))


# ---------------------------------------------------------------------------
# Reddit RSS
# ---------------------------------------------------------------------------

SUBREDDIT_FROM_LINK = re.compile(r"/r/([^/]+)/")


def fetch_all_posts() -> list[dict]:
    """Fetch new posts across every configured subreddit in a SINGLE request,
    using Reddit's combined multi-subreddit feed (r/a+b+c/new.rss). This is
    deliberate: one request is far less likely to get anonymous-traffic
    rate-limited than N separate requests back to back, which is what
    happened when we polled subreddits one at a time. Never raises; logs
    and returns an empty list on failure."""
    combined = "+".join(urllib.parse.quote(s) for s in SUBREDDITS)
    url = f"https://www.reddit.com/r/{combined}/new.rss?limit={REDDIT_LIMIT}"

    for attempt in range(3):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        except requests.RequestException as e:
            print(f"WARNING: failed to fetch combined feed: {e}", file=sys.stderr)
            return []

        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"WARNING: rate limited (attempt {attempt + 1}/3), waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(
                f"WARNING: combined feed returned HTTP {resp.status_code}: "
                f"{resp.text[:300]!r}",
                file=sys.stderr,
            )
            return []

        break
    else:
        print("WARNING: still rate limited after 3 attempts, skipping this run", file=sys.stderr)
        return []

    feed = feedparser.parse(resp.content)
    if feed.bozo and not feed.entries:
        print("WARNING: combined feed did not parse, skipping this run", file=sys.stderr)
        return []

    posts = []
    for entry in feed.entries:
        link = entry.get("link", "")
        match = SUBREDDIT_FROM_LINK.search(link)
        subreddit = match.group(1) if match else "unknown"
        posts.append({
            "id": entry.get("id", link),
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "link": link,
            "subreddit": subreddit,
        })
    return posts


# ---------------------------------------------------------------------------
# Keyword pre-filter
# ---------------------------------------------------------------------------

def matches_keywords(post: dict) -> bool:
    haystack = f"{post['title']} {post['summary']}".lower()
    return any(kw in haystack for kw in KEYWORDS)


# ---------------------------------------------------------------------------
# OpenRouter classification + reply drafting
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are screening Reddit posts for a company that sells voice AI \
(AI voice agents for calls, IVR replacement, AI receptionists, outbound calling automation).

Given a Reddit post's title and summary, decide:
1. Is this a genuine conversation where a voice AI product/company could plausibly \
add value or be relevant (someone asking for solutions, complaining about a problem \
voice AI solves, discussing the space, etc)? A stray keyword match in an unrelated \
context (e.g. "voice" meaning someone's opinion, "AI" as a generic buzzword complaint) \
does NOT count as relevant.
2. If relevant, draft a short, genuinely helpful, non-salesy Reddit reply. It should \
read like a knowledgeable person contributing to the discussion, not an ad. Do not \
mention any specific company by name in the draft.

Post title: {title}
Post summary: {summary}
Subreddit: r/{subreddit}

Respond with ONLY a JSON object, no other text, no markdown code fences:
{{"relevant": true or false, "confidence": 0.0 to 1.0, "reply_draft": "..." or null}}
"""


def classify_and_draft(post: dict) -> dict | None:
    """Calls OpenRouter to classify relevance and draft a reply.
    Returns None on any failure (caller should treat as 'skip this post')."""
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return None

    prompt = CLASSIFY_PROMPT.format(
        title=post["title"],
        summary=post["summary"][:1500],  # keep prompt bounded
        subreddit=post["subreddit"],
    )

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(
                f"WARNING: OpenRouter returned HTTP {resp.status_code}: {resp.text[:500]!r}",
                file=sys.stderr,
            )
            return None
    except requests.RequestException as e:
        print(f"WARNING: OpenRouter request failed: {e}", file=sys.stderr)
        return None

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        print(f"WARNING: unexpected OpenRouter response shape: {e}", file=sys.stderr)
        return None

    return _parse_json_leniently(content)


def _parse_json_leniently(text: str) -> dict | None:
    """The model may wrap JSON in markdown fences or add stray text.
    Try direct parse first, then extract the first {...} block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    print(f"WARNING: could not parse model output as JSON: {text[:200]!r}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def send_slack_alert(post: dict, confidence: float, reply_draft: str | None) -> None:
    if not SLACK_WEBHOOK_URL:
        print("ERROR: SLACK_WEBHOOK_URL not set, cannot send alert", file=sys.stderr)
        return

    text = (
        f"*Voice AI mention in r/{post['subreddit']}* (confidence: {confidence:.2f})\n"
        f"*{post['title']}*\n"
        f"<{post['link']}|View thread>\n\n"
        f"*Draft reply:*\n{reply_draft or '_no draft generated_'}"
    )
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
    except requests.RequestException as e:
        print(f"WARNING: failed to send Slack alert: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seen_ids = load_seen_ids()
    seen_set = set(seen_ids)
    new_seen_ids = list(seen_ids)

    total_keyword_matches = 0
    total_alerts = 0

    posts = fetch_all_posts()
    total_fetched = len(posts)

    for post in posts:
        if post["id"] in seen_set:
            continue
        seen_set.add(post["id"])
        new_seen_ids.append(post["id"])

        if not matches_keywords(post):
            continue
        total_keyword_matches += 1

        result = classify_and_draft(post)
        if not result or not result.get("relevant"):
            continue

        confidence = float(result.get("confidence", 0))
        reply_draft = result.get("reply_draft")

        send_slack_alert(post, confidence, reply_draft)
        total_alerts += 1

        time.sleep(1)  # be polite to the OpenRouter free tier

    save_seen_ids(new_seen_ids)

    print(
        f"Run complete. Fetched: {total_fetched}, "
        f"keyword matches: {total_keyword_matches}, alerts sent: {total_alerts}"
    )


if __name__ == "__main__":
    main()
