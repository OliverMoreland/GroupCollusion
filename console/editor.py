"""Config editor page: build an experiment config step by step, see what it means and costs, save it, run it."""
import copy
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the collusion package, wherever streamlit is started from
from collusion.__main__ import expand_sweep
from collusion.config import CHANNEL_TYPES, STRUCTURES, Config, config_from_dict
from collusion.market import Market  # noqa: E402

# rough per-call cost and per-phase latency, from DeepSeek v4 flash smoke runs (36 calls, $0.005-0.007;
# phases take ~10-20 s with hosts pinned to Baidu/GMICloud); prompts grow with history, so long runs cost more per call
COST_PER_CALL = 0.0002
SECONDS_PER_PHASE = 15

STRUCTURE_INFO = {
    "single": ("One agent per firm", "The control condition from Fish et al.: each firm is a single agent who sets its price alone."),
    "average": ("Average of proposals", "Every team member proposes a price and the firm charges the mean. No deliberation step."),
    "vote": ("Team vote", "Every member proposes a price with a rationale, then every member votes for one proposal. Plurality wins; ties are broken at random."),
    "leader": ("Fixed leader", "Members send the leader price advice; the leader reads it and sets the price alone. The leader is fixed for the whole run."),
    "elected_leader": ("Elected leader", "Like a fixed leader, but the team elects its leader in round 1 and, optionally, re-elects every few rounds."),
    "representatives": ("Elected representatives", "The team elects a few representatives. Everyone proposes; only representatives vote on the proposals."),
}
CHANNEL_INFO = {  # must cover CHANNEL_TYPES
    "team_open": ("Team room", "One private group chat per firm. Teammates coordinate in it; competitors can't see it."),
    "team_dm": ("Teammate 1-on-1s", "Private 1-on-1 chats between every pair of teammates."),
    "cross_dm": ("Cross-firm 1-on-1s", "Private 1-on-1 chats between agents of different firms: the main route to explicit collusion."),
    "sub_dm": ("Cross-firm 1-on-1s, rank and file only", "Like cross-firm 1-on-1s, but the firms' decision-makers (leaders, representatives) are shut out. Tests whether collusion travels up through subordinates."),
    "all_dm": ("Everyone 1-on-1", "Private 1-on-1 chats between every pair of agents, inside and across firms."),
    "global_open": ("Market-wide room", "One chat that every agent of every firm can read and post in."),
}
SWEEP_COMMS = {  # named communication conditions offered in sweeps
    "no talk": [],
    "team room": ["team_open"],
    "team room + cross-firm 1-on-1s": ["team_open", "cross_dm"],
    "team room + rank-and-file 1-on-1s": ["team_open", "sub_dm"],
    "team room + market-wide room": ["team_open", "global_open"],
    "cross-firm 1-on-1s only": ["cross_dm"],
    "market-wide room only": ["global_open"],
}
SECTIONS = ["1 · Basics", "2 · Market", "3 · Firms", "4 · Talk", "5 · Agents", "6 · Sweep", "7 · Advanced"]
DEFAULTS = Config().to_dict()


def _load(path):
    raw = yaml.safe_load(open(path)) or {}
    sweep, presets = raw.pop("sweep", None), raw.pop("presets", {})
    data = copy.deepcopy(DEFAULTS)
    for k, v in raw.items():
        data[k] = {**data[k], **v} if isinstance(v, dict) and isinstance(data.get(k), dict) else v
    return data, sweep, presets


def _slug(text):
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "experiment"


def calls_per_round(c):
    """(min, max) LLM calls in a typical round, plus number of sequential phases."""
    n, size, s = c["market"]["n_firms"], c["team"]["size"], c["team"]["structure"]
    agents = n * size
    talk = bool(c["communication"]["channels"])
    sub = c["communication"]["subrounds"] if talk else 0
    lo = hi = agents  # proposals (leaders give one decision call instead of proposing)
    phases = 1
    if s in ("vote", "representatives"):
        voters = size if s == "vote" else c["team"]["n_representatives"]
        lo += n * voters
        hi += n * voters
        phases += 1
    if talk:
        lo += agents
        hi += agents * sub
        phases += sub
    k = c["team"]["reelect_every"]
    if s in ("elected_leader", "representatives"):  # elections, spread over rounds
        per = agents / (k if k else c["rounds"])
        lo += per
        hi += per
    return lo, hi, phases


def short(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,.0f}"


def describe(c):
    m, t, comm, a = c["market"], c["team"], c["communication"], c["agent"]
    firm = "a single agent" if t["structure"] == "single" else f"a team of {t['size']} agents"
    how = {
        "single": "who sets its price alone",
        "average": "whose proposals are averaged",
        "vote": "who propose prices and vote on them",
        "leader": "whose leader sets the price after hearing everyone's advice",
        "elected_leader": "who elect a leader" + (f" every {t['reelect_every']} rounds" if t["reelect_every"] else " once")
                          + "; the leader sets the price after hearing everyone's advice",
        "representatives": f"who elect {t['n_representatives']} representative(s)"
                           + (f" every {t['reelect_every']} rounds" if t["reelect_every"] else "")
                           + " to vote on everyone's proposals",
    }[t["structure"]]
    out = [f"**{m['n_firms']} firms** compete on price for **{c['rounds']} rounds**. Each firm is {firm} {how}.",
           f"Agents are `{a['model']}`" + (" with reasoning logged." if a["reasoning"] else " without reasoning.")]
    if comm["channels"]:
        chans = "; ".join(CHANNEL_INFO[ch][0].lower() for ch in comm["channels"])
        out.append(f"Before each pricing decision they can talk on: {chans}, for up to {comm['subrounds']} "
                   f"turn{'s' if comm['subrounds'] > 1 else ''} per round.")
    else:
        out.append("Agents can't talk to anyone.")
    return " ".join(out)


def problems(c):
    """Hard errors (from the real validator) and softer warnings."""
    errors, warnings = [], []
    try:
        config_from_dict(c)
    except (ValueError, KeyError, TypeError) as e:
        errors.append(str(e))
    t, chans = c["team"], c["communication"]["channels"]
    if t["structure"] == "single" and {"team_open", "team_dm"} & set(chans):
        warnings.append("Teams of one have no teammates, so team channels will be empty.")
    if "sub_dm" in chans and t["structure"] in ("vote", "average"):
        warnings.append("Rank-and-file 1-on-1s only differ from cross-firm 1-on-1s when firms have leaders or representatives.")
    if "sub_dm" in chans and t["structure"] == "single":
        errors.append("Rank-and-file 1-on-1s need teams: a single agent is its firm's decision-maker, so nobody could talk.")
    if c["rounds"] > 100 and c["communication"]["subrounds"] > 2 and t["size"] * c["market"]["n_firms"] > 12:
        warnings.append("This is a big run; try it with a few rounds or the mock LLM first.")
    return errors, warnings


def editor_page(launch):
    repo = Path(__file__).resolve().parents[1]
    st.title("New experiment")
    st.markdown("Start from an existing config, change what you need, and check the summary on the right before you "
                "save. Hover the **?** icons for details. Nothing runs until you press **Save & run**.")

    templates = sorted(str(p.relative_to(repo)) for p in (repo / "configs").glob("*.yaml"))
    tpl = st.selectbox("Start from", templates,
                       index=templates.index("configs/smoke.yaml") if "configs/smoke.yaml" in templates else 0,
                       help="Every field below starts at this config's value. Your edits never change the original file.")
    base, base_sweep, base_presets = _load(repo / tpl)
    key = lambda name: f"{tpl}:{name}"  # widgets reset when the template changes

    left, right = st.columns([3, 2], gap="large")
    c = copy.deepcopy(base)
    with left:
        tabs = st.tabs(SECTIONS)
        with tabs[0]:
            st.caption("What to call this experiment and how long to play.")
            c["name"] = st.text_input("Name", base["name"] if tpl != "configs/smoke.yaml" else "my_experiment", key=key("name"),
                                      help="Used for the file name (configs/<name>.yaml) and the run folder.")
            col = st.columns(2)
            c["rounds"] = col[0].number_input("Rounds", 1, 1000, base["rounds"], key=key("rounds"),
                                              help="Pricing periods. Fish et al. play 300; 3 is enough for a smoke test, "
                                                   "and collusion usually needs dozens to emerge.")
            c["seed"] = col[1].number_input("Seed", 0, 10_000, base["seed"], key=key("seed"),
                                            help="Fixes tie-breaks, leader picks and the mock LLM. The real LLM is still random.")
            c["output_dir"] = st.text_input("Save runs under", "" if tpl == "configs/smoke.yaml" else base["output_dir"],
                                            key=key("out"), placeholder=f"runs/{_slug(c['name'])}",
                                            help="Each run gets a timestamped folder here, which the dashboard reads. "
                                                 "Leave empty to use runs/<name>.") or f"runs/{_slug(c['name'])}"
        with tabs[1]:
            st.caption("The repeated Bertrand market from Calvano et al. (2020) and Fish et al. (2024): firms sell "
                       "differentiated products with logit demand, and each round every firm picks a price.")
            m = c["market"]
            col = st.columns(2)
            m["n_firms"] = col[0].number_input("Firms", 2, 20, base["market"]["n_firms"], key=key("firms"),
                                               help="More firms make tacit collusion harder to sustain.")
            m["alpha"] = col[1].number_input("Price scale α", 0.1, 100.0, float(base["market"]["alpha"]), step=0.1,
                                             key=key("alpha"),
                                             help="Multiplies every price and cost. Fish et al. use 1, 3.2 and 10 to keep "
                                                  "the model from leaning on textbook numbers.")
            with st.expander("Demand and cost details"):
                col = st.columns(3)
                m["cost"] = col[0].number_input("Unit cost", 0.0, 100.0, float(base["market"]["cost"]), key=key("cost"),
                                                help="Marginal cost, in units of α.")
                m["a"] = col[1].number_input("Quality a", 0.0, 10.0, float(base["market"]["a"]), key=key("a"),
                                             help="Product appeal; higher means more demand at a given price.")
                m["a0"] = col[2].number_input("Outside option a₀", -10.0, 10.0, float(base["market"]["a0"]), key=key("a0"),
                                              help="Appeal of buying nothing.")
                col = st.columns(3)
                m["mu"] = col[0].number_input("Differentiation μ", 0.01, 5.0, float(base["market"]["mu"]), key=key("mu"),
                                              help="Higher μ = products are less substitutable, so undercutting steals fewer customers.")
                m["beta"] = col[1].number_input("Market size β", 1.0, 10_000.0, float(base["market"]["beta"]), key=key("beta"),
                                                help="Scales quantities and profits.")
                wtp = col[2].number_input("Max willingness to pay", 0.0, 1000.0, float(base["market"]["max_wtp"] or 0.0),
                                          key=key("wtp"), help="Adds 'no customer would pay more than $X' to the prompt "
                                                               "(in units of α). 0 leaves it out.")
                m["max_wtp"] = wtp or None
            try:
                b = Market(**{k: v for k, v in m.items() if k != "max_wtp"}).benchmarks()
                col = st.columns(2)
                col[0].metric("Competitive (Nash) price", f"{b['nash_price']:.3f}")
                col[1].metric("Full-collusion (monopoly) price", f"{b['monopoly_price']:.3f}")
                st.caption("The collusion index Δ measures where prices land between these two: 0 = competitive, 1 = monopoly.")
            except Exception as e:  # noqa: BLE001 - odd parameter combinations can fail to solve
                st.warning(f"Can't compute benchmarks for these parameters: {e}")
        # the sweep tab is filled first because it changes what the firm and talk tabs show
        with tabs[5]:
            st.caption("Optionally turn this config into a grid: one run for every combination of the choices below. "
                       "Everything else stays as set in the other sections.")
            do_sweep = st.toggle("Make this a sweep", bool(base_sweep), key=key("sweep"))
            sweep, presets = None, {}
            if do_sweep:
                base_structs = (base_sweep or {}).get("team.structure", [base["team"]["structure"]])
                structs = st.multiselect("Decision structures", STRUCTURES, [s for s in base_structs if s in STRUCTURES],
                                         format_func=lambda s: STRUCTURE_INFO[s][0], key=key("sw:struct"))
                options = dict(SWEEP_COMMS)
                options.update({n: v for n, v in base_presets.items()})
                base_comms = (base_sweep or {}).get("communication.channels") or []
                comms = st.multiselect("Communication conditions", list(options),
                                       [x for x in base_comms if isinstance(x, str) and x in options],
                                       key=key("sw:comms"), help="Each condition is a set of channels.")
                seeds = st.number_input("Seeds per combination", 1, 20, len((base_sweep or {}).get("seed", [0])) or 1,
                                        key=key("sw:seeds"), help="Repeat each combination with different seeds to "
                                                                  "measure noise between runs.")
                sweep = {}
                if structs:
                    sweep["team.structure"] = structs
                if comms:
                    sweep["communication.channels"] = comms
                    presets = {n: options[n] for n in comms}
                sweep["seed"] = list(range(int(seeds)))
        with tabs[2]:
            st.caption("Who runs each firm, and how a team turns its members' views into one price.")
            t = c["team"]
            swept = (sweep or {}).get("team.structure")
            if swept:
                st.info("The sweep (tab 6) varies the structure: " + ", ".join(STRUCTURE_INFO[x][0] for x in swept)
                        + ". The settings below apply to each of them where relevant; one-agent firms always get a team of 1.")
                t["structure"] = next((x for x in swept if x != "single"), swept[0])
                relevant = set(swept)
            else:
                t["structure"] = st.radio("Decision structure", STRUCTURES, index=STRUCTURES.index(base["team"]["structure"]),
                                          key=key("structure"), format_func=lambda s: STRUCTURE_INFO[s][0], horizontal=True)
                st.info(STRUCTURE_INFO[t["structure"]][1])
                relevant = {t["structure"]}
            if relevant == {"single"}:
                t["size"] = 1
            else:
                t["size"] = st.slider("Agents per firm", 2, 10, max(2, base["team"]["size"]), key=key("size"),
                                      help="All members share the firm's profit.")
            if "leader" in relevant:
                t["leader"] = st.radio("Who leads", ["first", "random"], index=["first", "random"].index(base["team"]["leader"]),
                                       key=key("leader"), horizontal=True, help="first = agent 1 of each firm; random = drawn by seed.")
            if "representatives" in relevant:
                t["n_representatives"] = st.slider("Representatives per firm", 1, t["size"],
                                                   min(max(1, base["team"]["n_representatives"]), t["size"]), key=key("reps"))
            if relevant & {"elected_leader", "representatives"}:
                t["reelect_every"] = st.number_input("Re-elect every N rounds (0 = elect once)", 0, 1000,
                                                     base["team"]["reelect_every"], key=key("reelect"),
                                                     help="Elections happen before messaging in round 1 and every N rounds after.")
            if relevant & {"vote", "representatives"}:
                t["ballot_shows_rationales"] = st.toggle("Voters see each proposal's rationale",
                                                         base["team"]["ballot_shows_rationales"], key=key("rationales"))
        with tabs[3]:
            st.caption("Who can talk to whom before each pricing decision. Tick any combination; leave all unticked for no talk.")
            comm = c["communication"]
            swept_comms = (sweep or {}).get("communication.channels")
            if swept_comms:
                st.info("The sweep (tab 6) varies who can talk: " + "; ".join(swept_comms)
                        + ". The settings below apply whenever a condition allows talk.")
                chosen = sorted({ch for n in swept_comms for ch in presets[n]})
            else:
                chosen = []
                for ch in CHANNEL_INFO:  # CHANNEL_TYPES, in a friendlier order
                    name, info = CHANNEL_INFO[ch]
                    if st.checkbox(f"**{name}**", ch in base["communication"]["channels"], key=key(f"ch:{ch}")):
                        chosen.append(ch)
                    st.caption(info)
            comm["channels"] = [] if swept_comms else chosen
            if chosen:
                col = st.columns(3)
                comm["subrounds"] = col[0].number_input("Turns per round", 1, 10, base["communication"]["subrounds"],
                                                        key=key("subrounds"),
                                                        help="Messaging turns before pricing. Agents reply to each other "
                                                             "turn by turn and can say they're done early.")
                comm["max_chars"] = col[1].number_input("Max message length", 50, 5000, base["communication"]["max_chars"],
                                                        step=50, key=key("chars"), help="Longer messages are cut off.")
                comm["history_rounds"] = col[2].number_input("Remember messages from last N rounds", 1, 100,
                                                             base["communication"]["history_rounds"], key=key("hist"))
        with tabs[4]:
            st.caption("The model behind every agent and what it gets to see.")
            a = c["agent"]
            a["model"] = st.text_input("Model (OpenRouter id)", base["agent"]["model"], key=key("model"),
                                       help="Open-weight reasoning models let you read the chain of thought.")
            col = st.columns(3)
            a["reasoning"] = col[0].toggle("Log reasoning", base["agent"]["reasoning"], key=key("reasoning"),
                                           help="Ask the provider for the chain of thought and save it.")
            effort = col[1].selectbox("Reasoning effort", ["default", "low", "medium", "high"],
                                      index=["default", "low", "medium", "high"].index(base["agent"]["reasoning_effort"] or "default"),
                                      key=key("effort"))
            a["reasoning_effort"] = None if effort == "default" else effort
            a["temperature"] = col[2].slider("Temperature", 0.0, 2.0, float(base["agent"]["temperature"]), 0.1, key=key("temp"))
            col = st.columns(3)
            a["prompt_prefix"] = col[0].radio("Prompt prefix", ["P1", "P2"], index=["P1", "P2"].index(base["agent"]["prompt_prefix"]),
                                              key=key("prefix"), horizontal=True,
                                              help="Fish et al.'s two task framings. P1: explore but never undermine profitability. "
                                                   "P2: explore, including risky or aggressive prices, and remember that "
                                                   "pricing below competitors sells more. P1 led to more collusion in the paper.")
            a["history_window"] = col[1].number_input("Rounds of market history shown", 1, 1000, base["agent"]["history_window"],
                                                      key=key("window"), help="Shorter windows keep prompts (and cost) small.")
            a["max_tokens"] = col[2].number_input("Max tokens per call", 500, 32000, base["agent"]["max_tokens"], step=500,
                                                  key=key("tokens"), help="Reasoning counts against this. Calls that run out "
                                                                          "are retried with double the budget.")
            col = st.columns(2)
            a["show_competitor_prices"] = col[0].toggle("Show competitors' past prices", base["agent"]["show_competitor_prices"],
                                                        key=key("comp"))
            a["max_price"] = col[1].number_input("Highest allowed price (× α)", 1.0, 1000.0, float(base["agent"]["max_price"]),
                                                 key=key("maxp"), help="Proposals above this are clipped.")
        with tabs[6]:
            st.caption("Plumbing. The defaults are fine for almost every run.")
            llm = c["llm"]
            col = st.columns(3)
            llm["concurrency"] = col[0].number_input("Parallel API calls", 1, 128, base["llm"]["concurrency"], key=key("conc"))
            llm["timeout"] = col[1].number_input("Timeout (s)", 10.0, 1200.0, float(base["llm"]["timeout"]), key=key("timeout"),
                                                 help="A stuck call holds up its whole phase until it times out and is retried. "
                                                      "Most calls finish in under a minute.")
            llm["max_retries"] = col[2].number_input("Retries", 0, 20, base["llm"]["max_retries"], key=key("retries"))
            routing = base["llm"].get("provider_routing") or {}
            only = st.text_input("Only use these OpenRouter providers", ", ".join(routing.get("only", [])), key=key("only"),
                                 placeholder="e.g. gmicloud, baidu (empty = let OpenRouter choose per call)",
                                 help="OpenRouter otherwise spreads calls over many hosts of the same model, some much "
                                      "slower than others and possibly differently quantized. Pinning a few fast ones "
                                      "speeds runs up and keeps conditions comparable. Calls fail rather than fall back.")
            names = [x.strip() for x in only.split(",") if x.strip()]
            llm["provider_routing"] = {**routing, "only": names, "allow_fallbacks": False} if names else (
                {k: v for k, v in routing.items() if k not in ("only", "allow_fallbacks")} or None)
            c["judge"]["model"] = st.text_input("Deception judge model", base["judge"]["model"], key=key("judge"),
                                                help="Used by `--judge` to score messages for deception after a run.")

    c["name"] = _slug(c["name"])
    data = {k: v for k, v in c.items()}
    if sweep:
        data["presets"], data["sweep"] = presets, sweep
    grid = [c]
    if sweep:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f, sort_keys=False)
        try:
            grid = [run for _, run in expand_sweep(f.name)]
        except Exception as e:  # noqa: BLE001
            grid = []
            errors_sweep = [f"sweep: {e}"]
        Path(f.name).unlink(missing_ok=True)
    errors, warnings = (errors_sweep, []) if not grid else ([], [])
    for run in grid:
        for out, found in zip((errors, warnings), problems(run)):
            out.extend(x for x in found if x not in out)

    with right:
        with st.container(border=True):
            st.subheader("Summary")
            if sweep:
                dims = [f"{len(v)} {k.split('.')[-1].replace('channels', 'communication conditions').replace('seed', 'seeds')}"
                        .replace("structure", "structures") for k, v in sweep.items()]
                talk = c["communication"]["subrounds"]
                st.markdown(f"A **sweep of {len(grid)} runs**: every combination of {' × '.join(dims)}. In each run "
                            f"**{c['market']['n_firms']} firms** play **{c['rounds']} rounds** with teams of {c['team']['size']} "
                            f"(1 for one-agent firms) of `{c['agent']['model']}` agents"
                            + (f", talking for up to {talk} turn{'s' if talk > 1 else ''} per round where allowed." if
                               any(run["communication"]["channels"] for run in grid) else "."))
            else:
                st.markdown(describe(c))
            for e in errors:
                st.error(e)
            for w in warnings:
                st.warning(w)
            per_run = [calls_per_round(run) for run in grid] or [(0, 0, 0)]
            n_runs = len(grid)
            calls_lo = sum(lo * run["rounds"] for (lo, _, _), run in zip(per_run, grid))
            calls_hi = sum(hi * run["rounds"] for (_, hi, _), run in zip(per_run, grid))
            sizes = sorted({run["market"]["n_firms"] * run["team"]["size"] for run in grid}) or [0]
            cost_lo, cost_hi = calls_lo * COST_PER_CALL, calls_hi * COST_PER_CALL
            money = lambda x: f"${x:,.2f}" if x < 10 else f"${x:,.0f}"
            mins = max(run["rounds"] * p for run, (_, _, p) in zip(grid, per_run)) * SECONDS_PER_PHASE / 60 if grid else 0
            col = st.columns(2)
            col[0].metric("Runs", n_runs)
            col[1].metric("Agents per run", f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}")
            col = st.columns(2)
            col[0].metric("LLM calls", f"~{short(calls_lo)}" if short(calls_lo) == short(calls_hi)
                          else f"{short(calls_lo)}–{short(calls_hi)}")
            col[1].metric("Rough cost", money(cost_lo) if money(cost_lo) == money(cost_hi)
                          else f"{money(cost_lo)}–{money(cost_hi)}")
            st.metric("Rough time per run" + (" (longest)" if sweep else ""),
                      f"{mins / 60:,.1f} h" if mins >= 120 else f"{mins:,.0f} min" if mins >= 1 else "< 1 min")
            st.caption("Estimates assume about $0.0002 per call and 15 s per call phase, as seen in DeepSeek smoke runs on pinned hosts. "
                       "Prompts grow with history, so long runs cost more. Sweeps run in parallel.")

        path = repo / "configs" / f"{c['name']}.yaml"
        header = f"# {c['name']}: made with the console's config editor (started from {tpl}).\n# " + \
                 re.sub(r"[*`]", "", describe(c)).replace(". ", ".\n# ") + "\n"
        text = header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        with st.expander("YAML preview", expanded=False):
            st.code(text, language="yaml")

        exists = path.exists()
        if exists:
            st.warning(f"configs/{path.name} already exists.")
        overwrite = st.checkbox("Overwrite it", key=key("overwrite")) if exists else True
        mock = st.toggle("Mock LLM (offline, free, for checking the setup)", key=key("mock"))
        parallel = st.number_input("Runs at once", 1, 64, min(n_runs, 8), key=key("parallel")) if sweep else 1
        col = st.columns(2)
        blocked = bool(errors) or not overwrite
        if col[0].button("Save", disabled=blocked, width="stretch"):
            path.write_text(text)
            st.success(f"Saved configs/{path.name}. It's now in the 'start a run' list.")
        if col[1].button("Save & run", type="primary", disabled=blocked, width="stretch"):
            path.write_text(text)
            log = launch(str(path.relative_to(repo)), bool(sweep), [], mock, parallel)
            st.success(f"Saved configs/{path.name} and started it (log: {log}). Switch to the dashboard to watch; "
                       "it shows up in the run list once round 1 is logged.")
