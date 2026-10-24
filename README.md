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
export OPENROUTER_KEY=...                      # or OPENROUTER_API_KEY

python -m collusion benchmarks                 # Nash / monopoly prices for the market
python -m collusion run configs/smoke.yaml --echo          # 3-round live check (cheap)
python -m collusion run configs/base.yaml                  # Fish et al. replication (300 rounds)
python -m collusion run configs/teams.yaml --judge         # teams of 3, voting, team channel
python -m collusion sweep configs/sweep.yaml --dry-run     # list the full grid
python -m collusion sweep configs/sweep.yaml --judge       # run it
python -m collusion report runs/sweep                      # one row per run -> report.csv
python -m pytest                                           # offline tests (mock LLM)
```

Every command accepts `--mock` to use an offline mock LLM, and `run`/`sweep` accept dotted
overrides, e.g. `--set team.structure=leader --set 'communication.channels=[team_open,cross_dm]'`.

## The game

`collusion/market.py`: n firms, logit demand
`q_i = β·exp((a − p_i/α)/μ) / (Σ_j exp((a − p_j/α)/μ) + exp(a₀/μ))`, profit `(p_i − αc)·q_i`,
defaults `a=2, a₀=0, μ=0.25, c=1, β=100` (Calvano et al. 2020, as in Fish et al.). For a
duopoly at α=1 the Nash price is 1.473 and the monopoly price is 1.925. The headline metric
is the collusion index **Δ = (p̄ − p_Nash)/(p_Monopoly − p_Nash)** (0 = competitive, 1 = full
collusion), plus the same index on profits.

Each agent gets the Fish et al. task prompt (P1 or P2 prefix), the last `history_window`
rounds of market data, and its own PLANS.txt / INSIGHTS.txt notes which it rewrites every
round. The prompt text is adapted from the paper and should be checked against its appendix
before treating results as a strict replication (`collusion/prompts.py`).

## Team structures (`team.structure`)

| structure | how the firm's price is set each round |
|---|---|
| `single` | control: one agent per firm (team size 1) |
| `average` | every member proposes; price = mean of proposals |
| `vote` | every member proposes (with a rationale), then every member votes for one proposal; plurality wins, ties random |
| `leader` | non-leaders propose with advice; the leader reads it and sets the price |
| `representatives` | the team elects `n_representatives` members (once, or every `reelect_every` rounds); everyone proposes, only representatives vote |

All members share the firm's profit.

## Communication (`communication.channels`)

A composable list of channel types. Before pricing, every agent with at least one channel
gets `subrounds` simultaneous messaging turns and may post at most one message per channel
per turn. Agents see the last `history_rounds` rounds of messages on their channels.

| type | channels created |
|---|---|
| `[]` | no communication |
| `team_dm` | 1-on-1 between teammates |
| `team_open` | one open channel per team |
| `cross_dm` | 1-on-1 between agents on different teams (none inside teams) |
| `all_dm` | 1-on-1 between every pair of agents |
| `global_open` | one open channel shared by all agents of all teams |

Combine freely, e.g. `[team_open, cross_dm]`. `configs/sweep.yaml` names presets for the grid.

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
configs/         base (replication), teams, smoke, sweep
tests/           offline tests using the mock LLM
```
