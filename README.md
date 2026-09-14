# Reddit voice AI GTM monitor

Polls a fixed list of subreddits (via their public `.rss` feeds - no Reddit
API key needed), keyword-filters new posts, sends survivors to a free
OpenRouter model for a relevance check + draft reply, and posts genuine hits
to a Slack channel.

## How it works

1. `monitor.py` fetches `reddit.com/r/{subreddit}/new.rss?limit=100` for each
   subreddit in `SUBREDDITS`.
2. New posts (not in `seen_ids.json`) are checked against a keyword list.
3. Keyword matches are sent to OpenRouter (`inclusionai/ling-3.0-flash:free`)
   which decides if the post is genuinely relevant and, if so, drafts a
   reply.
4. Relevant hits are posted to Slack via an incoming webhook.
5. `seen_ids.json` is updated and committed back to the repo so the next run
   doesn't re-alert on the same posts.

## One-time setup

1. **Create a Slack incoming webhook** (Slack app > Incoming Webhooks >
   Add New Webhook to Workspace). Copy the URL.
2. **Create a free OpenRouter account and API key** at
   https://openrouter.ai/keys.
3. In this GitHub repo, go to **Settings > Secrets and variables > Actions**
   and add two repository secrets:
   - `SLACK_WEBHOOK_URL`
   - `OPENROUTER_API_KEY`
4. Edit `monitor.py`:
   - Update `USER_AGENT` to include your actual Reddit username (Reddit
     wants a descriptive, non-generic user agent string).
   - Adjust `SUBREDDITS` or `KEYWORDS` as needed.
5. Commit and push. The workflow in `.github/workflows/monitor.yml` will
   start running automatically every 30 minutes. You can also trigger a run
   manually from the repo's **Actions** tab (`workflow_dispatch`).

## Tuning

- **Missing posts on very active subreddits**: increase polling frequency
  in the cron schedule for those subreddits, or split them into a separate
  workflow file with a tighter schedule.
- **Too much noise**: tighten `KEYWORDS`, or raise the confidence threshold
  check in `monitor.py` (currently any `relevant: true` triggers an alert
  regardless of confidence - you could require e.g. `confidence >= 0.6`).
- **Rate limits from OpenRouter's free tier**: if you hit limits, consider
  batching multiple posts into one classification call, or switching to a
  paid low-cost model for just the classification step.

## Known limitations

- RSS feeds return at most 100 items per subreddit per fetch. Very high
  volume subreddits polled infrequently could have older posts pushed out
  of the window between polls.
- This monitors **new posts only**, not comments on existing threads.
- Reddit rate-limits anonymous RSS traffic. If you see repeated 429
  warnings in the Actions logs, slow down the schedule.
- The OpenRouter free model doesn't enforce JSON output, so parsing has a
  fallback but can occasionally fail on a malformed response (that post is
  just skipped, logged as a warning).
