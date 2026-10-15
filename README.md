# Reddit voice AI GTM monitor

Polls a fixed list of subreddits (via their public `.rss` feeds - no Reddit
API key needed), keyword-filters new posts, sends survivors to a free
OpenRouter model for a relevance check + draft reply, and posts genuine hits
to a Slack channel.

## How it works

1. Every 10 min, GitHub Actions runs `monitor.py`.
2. It pulls Reddit RSS: site-wide search for voice-AI queries, `/new` and
   `/comments` for every sub in `subreddits.txt`.
3. Items are de-duplicated, age-filtered (36h) and keyword-scored
   (word-boundary regex, strong/weak tiers).
4. Survivors go to the LLM in batches of 8 (Gemini free tier first, then
   OpenRouter free) which returns relevance, intent, lead quality and a draft.
5. Hits above `MIN_CONFIDENCE` (0.6) are posted to Slack. Items are marked
   seen only after they are fully handled, so outages never drop leads.

## Setup
Secrets (Settings > Secrets > Actions): `SLACK_WEBHOOK_URL`, `GEMINI_API_KEY`
(free, aistudio.google.com/apikey), optional `OPENROUTER_API_KEY`.
Edit `subreddits.txt` to change tracked subs. Replies are always posted by a
human. Test locally with `DRY_RUN=1 python monitor.py`.
