# Claude Code as the Fitscan LLM provider

Fitscan can route every Claude call (vision measurement, validation,
correction, height estimation, landmark verification, and style
classification) through one of two backends:

| `FITSCAN_LLM_PROVIDER` | Backend | Billing | Notes |
| --- | --- | --- | --- |
| `api` (default) | Anthropic Messages API | Anthropic API credits | Full control: `temperature=0`, prompt caching, `metadata.user_id`. Unchanged production path. |
| `claude_code` | Claude Code via `claude-agent-sdk` | Your Claude subscription *or* API credits | Runs off `claude login` / a subscription token. See trade-offs below. |

The provider seam lives in [`fitscan/llm/`](../fitscan/llm). Switching is a
single env var; nothing else in the pipeline changes.

> Claude Code is **not** local/on-device inference. Both backends send
> prompts and images to Anthropic's servers. The reason to use Claude Code
> is auth/billing (run off a subscription) and MCP, not data locality.

## What you give up on the `claude_code` path

The Agent SDK does not expose some controls the direct API has. These are
SDK limitations, not Fitscan choices:

- **No `temperature`.** Measurement determinism is weaker (the API path
  uses `temperature=0` so the same photo yields the same numbers).
- **No `metadata.user_id`.** The consent gate still applies (a photo is
  only ever sent when a `subject_id` is present), but Anthropic-side
  retention can't be targeted per user.
- **No prompt-cache `cache_control`.**
- **Images** are delivered by writing them to a temp file and letting
  Claude Code read them with the `Read` tool (the SDK has no inline base64
  image input). Multi-image (front + side) reliability is weaker.

For these reasons the default stays `api`. Use `claude_code` deliberately.

## Install

```bash
# In the fitscan repo, with its venv active:
pip install -e ".[claude-code]"      # adds claude-agent-sdk
```

The `claude` CLI must also be on PATH (installed separately, e.g. via the
official installer). Verify:

```bash
claude --version
```

## Auth on a headless VPS (no browser)

Set `FITSCAN_CLAUDE_CODE_AUTH` to one of three modes. The provider builds
the Claude Code subprocess environment so the chosen credential actually
wins. In particular it **scrubs `ANTHROPIC_API_KEY`** in the subscription
modes, because the API key otherwise silently shadows a subscription token
(precedence is `ANTHROPIC_API_KEY` > `CLAUDE_CODE_OAUTH_TOKEN` > on-disk
login).

### Mode `api_key` (default), API billing

```dotenv
FITSCAN_LLM_PROVIDER=claude_code
FITSCAN_CLAUDE_CODE_AUTH=api_key
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Mode `oauth_token`, your subscription, headless

1. On a machine **with a browser** (your laptop), run once:

   ```bash
   claude setup-token
   ```

   It does the OAuth flow and prints a long-lived (≈1 year) token.

2. On the VPS, set:

   ```dotenv
   FITSCAN_LLM_PROVIDER=claude_code
   FITSCAN_CLAUDE_CODE_AUTH=oauth_token
   FITSCAN_CLAUDE_CODE_OAUTH_TOKEN=<token from step 1>
   ```

   Do **not** also set `ANTHROPIC_API_KEY` for the worker, the provider
   scrubs it anyway, but keeping it out of the env avoids confusion.

### Mode `cli_session`, persistent on-disk login

A prior `claude login` stores credentials at `~/.claude/.credentials.json`
(Linux, mode `0600`); they auto-refresh and survive reboots. Since the
login flow needs a browser, on a UI-less VPS you either run
`claude setup-token` (mode above) or copy the credential file across:

```bash
# on your laptop, after `claude login`
scp ~/.claude/.credentials.json user@vps:~/.claude/.credentials.json
ssh user@vps chmod 600 ~/.claude/.credentials.json
```

```dotenv
FITSCAN_LLM_PROVIDER=claude_code
FITSCAN_CLAUDE_CODE_AUTH=cli_session
```

## Model

The Claude Code path defaults to **Opus 4.8** (`claude-opus-4-8`), set via
`FITSCAN_CLAUDE_CODE_MODEL`. The direct-API path keeps Sonnet (hardcoded per
call). Override if needed:

```dotenv
FITSCAN_CLAUDE_CODE_MODEL=claude-opus-4-8   # default; or claude-opus-4-7, claude-sonnet-4-6
```

On a front+side measurement test, Opus 4.8 and 4.7 agreed with the
deterministic API/Sonnet baseline to ~1.2-1.3 inches mean, and ran faster
through Claude Code than Sonnet did (~55-70s vs ~260s). Note: Claude Code
cannot pin `temperature=0`, so the Opus path is still not repeatable
run-to-run; only the direct-API path is deterministic.

## Measurement stability (ensembling)

Opus cannot pin `temperature=0`, so a single scan varies run-to-run (typically
~1 inch, with the odd ~3 inch spike on hard fields like knee/bust). To get
repeatable numbers, run the scan K times and fuse by per-field **median** (the
median discards single-run outliers):

```dotenv
FITSCAN_MEASURE_ENSEMBLE_SIZE=3   # 1 = single call (default); 3 = recommended for Opus
```

Measured on a real front+side scan, going from 1 to 3 cut scan-to-scan
variation from ~1.06 in mean / 3.0 in max to **~0.33 in mean / 1.8 in max**
(19/25 fields within 0.5 in). The prompt's own "median-of-three" instruction
smooths within one call; this setting ensembles across calls, which is what
actually rejects the spikes. The K runs execute **concurrently**, so wall-clock
is ~one call (~47s at K=3 through Claude Code), not K times; only the token cost
scales with K. The direct-API Sonnet path is already deterministic
(`temperature=0`), so it does not need this.

The measurement prompt also runs a "characterise the body first" step: the model
commits to a `body` description (build, shape, orientation, pose/fit, visibility)
BEFORE measuring, then must keep the numbers consistent with the declared shape
(e.g. an "oval" body must have waist >= bust and hip). Writing the description
first anchors the measurements and improves run-to-run consistency.

## MCP servers (optional)

You can attach MCP servers to every Claude Code request. Set
`FITSCAN_CLAUDE_CODE_MCP_SERVERS` to a JSON string, either the
`{"mcpServers": {...}}` envelope or a bare server-name → config map:

```dotenv
FITSCAN_CLAUDE_CODE_MCP_SERVERS={"mcpServers":{"my-tool":{"command":"npx","args":["-y","@scope/server"]}}}
```

> No MCP improves anthropometric body measurement, MCPs are tool/data
> connectors, not vision enhancers. This is plumbing for servers you
> choose to add; leave it empty to disable MCP.

## Quick switch summary

```dotenv
# Default, unchanged production behaviour
FITSCAN_LLM_PROVIDER=api
ANTHROPIC_API_KEY=sk-ant-api03-...

# Route everything through Claude Code on your subscription
FITSCAN_LLM_PROVIDER=claude_code
FITSCAN_CLAUDE_CODE_AUTH=oauth_token
FITSCAN_CLAUDE_CODE_OAUTH_TOKEN=...
```
