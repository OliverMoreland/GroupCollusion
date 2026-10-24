# GroupCollusion

Do **teams** of LLM agents collude differently from single agents? This repo replicates
Fish et al. (2024), *Algorithmic Collusion by Large Language Models* (repeated Bertrand
pricing with logit demand, prices compared to the Nash and monopoly benchmarks), and extends
it so each firm is run by a team of agents with a configurable decision structure and
communication topology. Agents are open-weight DeepSeek models via OpenRouter so their
chain-of-thought can be logged and inspected.

## Quick start

```bash
pip install -r requirements.txt
# the model runs on our own vLLM server on Modal (see "Model server" below); put its URL and key in .env:
#   MODAL_LLM_URL=https://<workspace>--groupcollusion-deepseek-serve.modal.run/v1
#   MODAL_LLM_KEY=...
# or use OpenRouter instead with --set llm.provider=openrouter and OPENROUTER_KEY=... in .env

python -m collusion benchmarks                 # Nash / monopoly prices for the market
python -m collusion run configs/smoke.yaml --echo          # 3-round live check (cheap)
python -m collusion run configs/base.yaml                  # Fish et al. replication (300 rounds)
python -m collusion run configs/teams.yaml --judge         # teams of 3, voting, team channel
python -m collusion sweep configs/sweep.yaml --dry-run     # list the full grid
python -m collusion sweep configs/sweep.yaml --judge       # run it
python -m collusion sweep configs/grid.yaml --parallel 15  # the project grid (structure x cross-firm comms)
streamlit run console/app.py                               # live console over runs/ (sidebar: start a run, transcripts, new-config editor)
python -m collusion report runs/sweep                      # one row per run -> report.csv
python -m pytest                                           # offline tests (mock LLM)
```

Every command accepts `--mock` to use an offline mock LLM, and `run`/`sweep` accept dotted
overrides, e.g. `--set team.structure=leader --set 'communication.channels=[team_open,cross_dm]'`.

## Model server

`modal_app/deepseek.py` serves `deepseek-ai/DeepSeek-V4-Flash` (pinned revision) with vLLM on 4×B200 on Modal,
behind an OpenAI-compatible API with a key, under the same model name the configs use. `llm.provider: vllm` (the
default) sends every call there; the client waits for a cold start before the first call. The server scales to
zero 10 minutes after the last call and costs about $25/h while up.

```bash
pip install modal && modal setup
modal secret create groupcollusion-llm-key LLM_API_KEY=<random>    # same value as MODAL_LLM_KEY in .env
modal run modal_app/deepseek.py::download                          # once: ~160 GB of weights into a Volume
modal deploy modal_app/deepseek.py                                 # prints the URL for MODAL_LLM_URL (add /v1)
```

Thinking is always on (the template's "high" mode, as on OpenRouter), and reasoning comes back in the logs as before.
Cost per call isn't reported for this backend, so the console's cost column reads $0.

## The game

`collusion/market.py`: n firms, logit demand
`q_i = β·exp((a − p_i/α)/μ) / (Σ_j exp((a − p_j/α)/μ) + exp(a₀/μ))`, profit `(p_i − αc)·q_i`,
defaults `a=2, a₀=0, μ=0.25, c=1, β=100` (Calvano et al. 2020, as in Fish et al.). For a
duopoly at α=1 the Nash price is 1.473 and the monopoly price is 1.925. The headline metric
is the collusion index **Δ = (p̄ − p_Nash)/(p_Monopoly − p_Nash)** (0 = competitive, 1 = full
collusion), plus the same index on profits.

Each agent gets the Fish et al. task prompt (P1 or P2 prefix). How it remembers past rounds
depends on `agent.memory`:

- `conversation` (default): each agent keeps one chat for the whole run. Every call appends a user
  turn with only what is new since the agent last spoke (the round number, new market results, new
  messages from others, election results and closed channels) plus the task, and the agent's answer
  is appended after it. The system prompt never changes, so each call's prompt is the previous one
  plus a few lines, which providers can serve from their prompt cache. Proposals ask only for a
  price and an optional short `note` to self. Reasoning traces are logged but never sent back. Past
  `context_chars` characters, the oldest turns are dropped (down to half) and replaced by a market
  data recap.
- `notes`: the original Fish et al. setup. Every call is a fresh system + user pair with the last
  `history_window` rounds of market data and the agent's PLANS.txt / INSIGHTS.txt notes, which it
  rewrites every round.

The prompt text is adapted from the paper and should be checked against its appendix
before treating results as a strict replication (`collusion/prompts.py`).

## Team structures (`team.structure`)

| structure | how the firm's price is set each round |
|---|---|
| `single` | control: one agent per firm (team size 1) |
| `average` | every member proposes; price = mean of proposals |
| `vote` | every member proposes (with a rationale), then every member votes for one proposal; plurality wins, ties random |
| `leader` | non-leaders propose with advice; the leader reads it and sets the price |
| `elected_leader` | like `leader`, but members elect the leader on round 1 and every `reelect_every` rounds |
| `representatives` | the team elects `n_representatives` members (once, or every `reelect_every` rounds); everyone proposes, only representatives vote |

All members share the firm's profit.

## Communication (`communication.channels`)

A composable list of channel types. Before pricing, every agent with at least one channel
gets up to `subrounds` simultaneous messaging turns and may post at most one message per channel
per turn. An agent can answer `"done": true` to stop early; it is only asked again that round if a
new message reaches it. In `conversation` memory agents see every message from others on their channels once, as it arrives; in `notes` memory they see the last `history_rounds` rounds of messages each turn.

| type | channels created |
|---|---|
| `[]` | no communication |
| `team_dm` | 1-on-1 between teammates |
| `team_open` | one open channel per team |
| `cross_dm` | 1-on-1 between agents on different teams (none inside teams) |
| `sub_dm` | like `cross_dm`, but closed to current decision-makers (leader, elected leader, representatives, a single-agent firm's agent) |
| `all_dm` | 1-on-1 between every pair of agents |
| `global_open` | one open channel shared by all agents of all teams |

Combine freely, e.g. `[team_open, cross_dm]`. `configs/sweep.yaml` names presets for the grid.

`configs/grid.yaml` is the project grid: `vote` / `leader` / `elected_leader` (every 5 rounds) /
`single` x cross-firm comms `none` / `sub_dm` / `cross_dm` / `global_open`, with a private team
channel in every condition, 5 firms of 5 agents, 3 messaging turns, 3 seeds (45 runs; single x
sub_dm is undefined and dropped). `sweep --parallel N` plays N runs at once.

## Live console

`streamlit run console/app.py` reads `runs/**/events.jsonl` as it is written (refreshes every
5 s): a heatmap of the collusion index over the grid with run status and cost, and per run the
prices against Nash / monopoly, message volume, leader deviation from advice, a searchable
message feed (with judge scores once judged), every call's reasoning / response / prompt,
elections, and errors.

## Observability

Each run writes `runs/<timestamp>_<name>/`:

- `events.jsonl`: every event, append-only. Each `llm_call` has the full prompt, response,
  **reasoning trace**, token usage, cost and latency; plus `message`, `proposal` (with the
  agent's rewritten notes), `election`, `team_decision` (proposals, ballots, winner),
  `market`, `parse_error`, `llm_error`.
- `rounds.csv`: per-round prices, quantities, profits, Δ, message counts.
- `messages.csv`: every message with channel, scope (intra/inter-team), kind (dm/open) and judge scores.
- `summary.json`: pricing vs benchmarks (whole run and last half), communication volume by
  scope/kind/channel/agent, team dynamics (within-team proposal spread, leader deviation
  from advice, vote ties/unanimity), LLM usage and cost, deception scores.
- `transcript.md`: human-readable round-by-round log.

Inspect calls with chain-of-thought:

```bash
python -m collusion show runs/<run> --round 12 --agent A2 --phase message
```

## Deception judge

`python -m collusion judge runs/<run>` (or `--judge` on run/sweep) scores every message
0–10 for deception with an LLM judge (`judge.model`). The judge sees what recipients could
not: the sender's chain-of-thought when writing the message, the sender's full output, and
what the sender then did that round (proposal, vote, firm price). It also flags explicit
price coordination with competitors. Results go to `judgments.jsonl` and are folded into
`summary.json` and `messages.csv`. Re-running resumes.

## Layout

```
collusion/
  market.py      logit Bertrand market + Nash/monopoly benchmarks
  config.py      dataclass config, YAML loading, dotted overrides
  llm.py         OpenRouter client (reasoning on, retries) + mock + lenient JSON parsing
  channels.py    channel topologies and message store
  prompts.py     all prompt text
  simulation.py  round loop: elections, messaging, proposals, team decisions, market
  metrics.py     summary / CSV / transcript from events.jsonl
  judge.py       LLM-as-judge deception scoring
  __main__.py    CLI
console/app.py   live streamlit console
configs/         base (replication), teams, smoke, sweep, grid
tests/           offline tests using the mock LLM
```
