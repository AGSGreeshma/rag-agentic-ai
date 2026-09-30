"""Design tokens and global CSS for the Streamlit app."""

# Light pastel palette. "violet" = primary accent (interactive / AI states),
# "cyan" = retrieval & source context (a soft teal here). All text colours
# pass WCAG AA (>= 4.5:1) on the cream background except "faint" (metadata only).
TOKENS = {
    "bg": "#FAF7F2",        # warm cream
    "surface": "#FFFFFF",
    "elevated": "#F3EEFB",  # lavender mist (question bubbles)
    "violet": "#6A56CC",
    "cyan": "#2A7570",
    "text": "#2A2633",
    "muted": "#6B6577",
    "faint": "#847D91",
    "success": "#2B7759",
    "amber": "#96661F",
    "danger": "#B24552",
    "border": "#E9E3EE",
}

_VARS = ";".join(f"--{k}:{v}" for k, v in TOKENS.items())

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root { __VARS__;
  --violet-soft: #EFEAFD; --violet-line: #CEC3F4; --violet-fill: #B8A9F0;
  --cyan-soft: #E6F4F1; --cyan-line: #A8D8D0; --cyan-fill: #7CC4B8;
  --amber-soft: #FDF1E2; --amber-line: #EFCFA2; --amber-fill: #E8B77A;
  --danger-soft: #FCECEE; --danger-line: #EEC0C6;
  --success-soft: #E4F3EA; --success-line: #B3DDC6; --success-fill: #7CC39E;
  --pink: #F4C3D2; --peach: #F9D9C3;
  --radius: 12px; --mono: 'JetBrains Mono', ui-monospace, monospace;
  --serif: 'Instrument Serif', Georgia, 'Times New Roman', serif;
}

/* ---------- Streamlit shell ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stSidebar"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.stApp { color: var(--text); background:
  radial-gradient(900px 380px at 8% -8%, #F0E9FF 0%, transparent 60%),
  radial-gradient(760px 360px at 100% 0%, #E2F3EF 0%, transparent 55%),
  radial-gradient(700px 320px at 55% 105%, #FCE7DC 0%, transparent 60%), var(--bg);
  background-attachment: fixed; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, .stDeployButton, [data-testid="stDecoration"], footer { display: none !important; }
.block-container { max-width: 1180px; padding: 1.25rem 2rem 2.5rem; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] { padding-top: 1.25rem; }
.stMarkdown p { color: var(--text); }
[data-testid="stVerticalBlock"] { gap: 0.9rem; }
@media (max-width: 640px) { .block-container { padding: 1rem 1rem 2rem; } }

/* ---------- Header ---------- */
.ara-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 4px 0 18px; border-bottom: 1px solid var(--border); }
.ara-brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
.ara-logo { width: 42px; height: 42px; flex: none; display: grid; place-items: center;
  border: 1px solid var(--violet-line); border-radius: 12px; background: linear-gradient(135deg, var(--violet-soft), var(--cyan-soft)); }
.ara-brand-name { font: italic 400 24px/1.05 var(--serif); color: var(--text); }
.ara-brand-name span { color: var(--violet); }
.ara-brand-sub { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
.ara-status { display: inline-flex; align-items: center; gap: 8px; flex: none; padding: 7px 11px;
  border: 1px solid var(--border); border-radius: 8px; background: var(--surface);
  font: 500 11px var(--mono); letter-spacing: 0.1em; color: var(--text); }
.ara-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; display: inline-block; }
.ara-dot.ok { background: var(--success-fill); box-shadow: 0 0 0 3px var(--success-soft); }
.ara-dot.warn { background: var(--amber-fill); box-shadow: 0 0 0 3px var(--amber-soft); }
.ara-dot.off { background: #E48A96; box-shadow: 0 0 0 3px var(--danger-soft); }
.ara-dot.idle { background: var(--faint); }
.ara-banner { margin-top: 14px; padding: 10px 14px; border: 1px solid var(--amber-line); border-radius: 10px;
  background: var(--amber-soft); color: var(--amber); font-size: 13px; }

/* ---------- Labels ---------- */
.ara-label { font: 600 11px 'Inter', sans-serif; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 10px; }
.ara-label.cyan { color: var(--cyan); }
.ara-label.violet { color: var(--violet); }
.ara-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; margin: 6px 0 12px; }
.ara-section-head .ara-label { margin: 0; }
.ara-section-sub { font-size: 13px; color: var(--muted); }

/* Suggestion chips: shrink columns to content and let them wrap */
.st-key-chips [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 8px !important; }
.st-key-chips [data-testid="stColumn"], .st-key-chips [data-testid="column"] {
  flex: 0 1 auto !important; width: auto !important; min-width: 0 !important; max-width: 100% !important; }
.st-key-chips button { max-width: 100%; box-shadow: 0 1px 2px rgba(42,38,51,0.03); white-space: normal; text-align: left; height: auto !important; background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; min-height: 34px !important; padding: 5px 12px !important;
  transition: border-color .15s, background .15s; }
.st-key-chips button p { color: var(--muted) !important; font-size: 13px !important; }
.st-key-chips button:hover { border-color: var(--violet-line) !important; background: var(--violet-soft) !important; }
.st-key-chips button:hover p { color: var(--text) !important; }
.ara-chips-label { font-size: 12px; color: var(--faint); margin: 2px 0 -2px; }

/* Generic secondary buttons (sidebar etc.) */
[data-testid="stSidebar"] button { background: transparent !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; }
[data-testid="stSidebar"] button p { color: var(--muted) !important; font-size: 13px !important; }
[data-testid="stSidebar"] button:hover { border-color: var(--violet-line) !important; }

/* Loading */
.ara-stage { display: flex; align-items: center; gap: 12px; padding: 7px 0; font-size: 14px; color: var(--faint); }
.ara-stage .ara-stage-mark { width: 18px; height: 18px; flex: none; display: grid; place-items: center;
  border-radius: 50%; border: 1px solid var(--border); font-size: 11px; }
.ara-stage.done { color: var(--muted); }
.ara-stage.done .ara-stage-mark { border-color: var(--cyan-line); color: var(--cyan); }
.ara-stage.active { color: var(--text); }
.ara-stage.active .ara-stage-mark { border-color: var(--violet); }
.ara-stage.active .ara-stage-mark::after { content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--violet); animation: ara-breathe 1.6s ease-in-out infinite; }
@keyframes ara-breathe { 0%,100% { opacity: .35; } 50% { opacity: 1; } }

/* Answer text */
/* Same display face as the question bubbles; Instrument Serif has one weight, so emphasis
   is shown upright (non-italic) rather than with a synthetic bold. */
.ara-answer-body { font: italic 400 19.5px/1.55 var(--serif); color: var(--text); }
.ara-answer-body p { margin: 0 0 12px; color: var(--text); }
.ara-answer-body ul { margin: 0 0 12px; padding-left: 20px; }
.ara-answer-body li { margin: 0 0 8px; color: var(--text); }
.ara-answer-body li::marker { color: var(--violet); }
.ara-answer-body strong { font-style: normal; font-weight: 400; color: var(--violet); }
.ara-cite { display: inline-block; font: normal 500 11.5px var(--mono); color: var(--cyan); background: var(--cyan-soft);
  border: 1px solid var(--cyan-line); border-radius: 5px; padding: 0 5px; margin: 0 1px; vertical-align: 1px; }

/* Retrieved context */
.ara-src { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--cyan-fill);
  border-radius: 10px; margin-bottom: 10px; transition: border-color .15s, background .15s; }
.ara-src:hover { border-color: var(--cyan-line); }
.ara-src summary { list-style: none; cursor: pointer; padding: 14px 16px; }
.ara-src summary::-webkit-details-marker { display: none; }
.ara-src-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 8px; }
.ara-src-page { font: 500 11px var(--mono); letter-spacing: 0.1em; color: var(--cyan); }
.ara-src-match { display: flex; align-items: center; gap: 8px; font: 500 11px var(--mono); letter-spacing: 0.06em; color: var(--muted); }
.ara-src-match .bar { width: 54px; height: 4px; border-radius: 2px; background: var(--border); overflow: hidden; }
.ara-src-match .bar > i { display: block; height: 100%; background: var(--cyan-fill); }
.ara-src-preview { font-size: 13.5px; line-height: 1.6; color: var(--muted); }
.ara-src-more { display: inline-block; margin-top: 8px; font-size: 12.5px; color: var(--cyan); opacity: .85; }
.ara-src-more .less { display: none; }
.ara-src[open] .ara-src-more .more { display: none; }
.ara-src[open] .ara-src-more .less { display: inline; }
.ara-src[open] .ara-src-preview { display: none; }
.ara-src-full { padding: 0 16px 16px; margin-top: -4px; font-size: 13.5px; line-height: 1.7; color: var(--text);
  white-space: pre-wrap; }
.ara-src.unused { border-left-color: var(--border); opacity: .78; }
.ara-src.unused .ara-src-page { color: var(--muted); }
.ara-others { margin-top: 4px; }
.ara-others > summary { list-style: none; cursor: pointer; font-size: 13px; color: var(--muted); padding: 8px 2px; }
.ara-others > summary::-webkit-details-marker { display: none; }
.ara-others > summary:hover { color: var(--text); }
.ara-others > summary::before { content: "+ "; color: var(--faint); }
.ara-others[open] > summary::before { content: "- "; }

/* Pipeline */
.ara-pipe, .ara-src, .ara-oos, .ara-error { box-shadow: 0 1px 2px rgba(42,38,51,0.03); }
.ara-pipe { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 18px 14px; }
.ara-step { display: grid; grid-template-columns: 18px 1fr auto; gap: 12px; }
.ara-rail { display: flex; flex-direction: column; align-items: center; }
.ara-node { width: 12px; height: 12px; border-radius: 50%; margin-top: 3px; flex: none;
  border: 1.5px solid var(--border); background: var(--bg); transition: all .2s; }
.ara-line { width: 2px; flex: 1; min-height: 20px; margin: 4px 0 0; border-radius: 1px; background: var(--border); }
.ara-step.last .ara-line { display: none; }
.ara-step-body { padding-bottom: 14px; min-width: 0; }
.ara-step-title { font: 600 11px 'Inter', sans-serif; letter-spacing: 0.12em; text-transform: uppercase; color: var(--faint); }
.ara-step-sub { font-size: 12.5px; color: var(--faint); margin-top: 3px; line-height: 1.45; }
.ara-step-meta { font: 500 11px var(--mono); color: var(--faint); padding-top: 2px; white-space: nowrap; }
.ara-step.done .ara-node { border-color: var(--cyan-fill); background: var(--cyan-fill); }
.ara-step.done .ara-line, .ara-step.success .ara-line { background: var(--cyan-line); }
.ara-step.done .ara-step-title { color: var(--text); }
.ara-step.done .ara-step-sub, .ara-step.done .ara-step-meta { color: var(--muted); }
.ara-step.active .ara-node { border-color: var(--violet); background: var(--violet); box-shadow: 0 0 0 4px var(--violet-soft); }
.ara-step.active .ara-step-title { color: var(--violet); }
.ara-step.active .ara-step-sub { color: var(--muted); }
.ara-step.success .ara-node { border-color: var(--success-fill); background: var(--success-fill); }
.ara-step.success .ara-step-title { color: var(--success); }
.ara-step.success .ara-step-sub, .ara-step.success .ara-step-meta { color: var(--muted); }
.ara-step.warn .ara-node { border-color: var(--amber-fill); background: var(--amber-fill); }
.ara-step.warn .ara-step-title { color: var(--amber); }
.ara-step.warn .ara-step-sub, .ara-step.warn .ara-step-meta { color: var(--muted); }
.ara-step.skipped .ara-node { border-style: dashed; }
.ara-step.error .ara-node { border-color: #E48A96; background: #E48A96; }
.ara-step.error .ara-step-title { color: var(--danger); }
.ara-step.error .ara-step-sub, .ara-step.error .ara-step-meta { color: var(--muted); }
.ara-step.skipped .ara-step-title { text-decoration: line-through; text-decoration-color: var(--faint); }
.ara-pipe-foot { border-top: 1px solid var(--border); margin-top: 4px; padding-top: 12px; }
.ara-trace { font: 400 11px/1.7 var(--mono); color: var(--faint); word-break: break-word; }
.ara-trace b { color: var(--muted); font-weight: 500; }
.ara-rewrites { margin-top: 10px; }
.ara-rewrites .q { font: 400 11.5px/1.5 var(--mono); color: var(--muted); padding: 5px 8px; margin-top: 5px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--bg); }

/* Out of scope */
.ara-oos { background: var(--amber-soft); border: 1px solid var(--amber-line); border-radius: 14px; padding: 22px 24px; }
.ara-oos-head { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.ara-oos-icon { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; flex: none;
  border: 1px solid var(--amber-line); color: var(--amber); font-weight: 700; }
.ara-oos-title { font: italic 400 24px/1.1 var(--serif); color: var(--amber); }
.ara-oos p { font-size: 15px; line-height: 1.6; color: var(--text); margin: 0; }
.ara-oos p:not(.ara-oos-note) { font: italic 400 19.5px/1.5 var(--serif); }
.ara-oos-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 18px 0 14px; }
.ara-oos-grid > div { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.ara-oos-grid span { display: block; font: 600 10.5px 'Inter', sans-serif; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.ara-oos-grid b { display: block; margin-top: 4px; font-size: 15px; font-weight: 600; color: var(--amber); }
.ara-oos-note { font-size: 12.5px !important; color: var(--muted) !important; }
@media (max-width: 640px) { .ara-oos-grid { grid-template-columns: 1fr; } }

/* Error */
.ara-error { background: var(--danger-soft); border: 1px solid var(--danger-line); border-radius: 14px; padding: 18px 22px; }
.ara-error-title { font: italic 400 22px/1.2 var(--serif); color: var(--danger); margin-bottom: 4px; }
.ara-error p { font-size: 14px; color: var(--text); margin: 0; }
.ara-error code { display: inline-block; margin-top: 10px; font: 12.5px var(--mono); color: var(--muted);
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; }

/* ---------- Chat layout ---------- */
.block-container { max-width: 860px; padding-bottom: 6rem; }
.st-key-chips [data-testid="stHorizontalBlock"] { justify-content: center; }
.ara-welcome { text-align: center; padding: 34px 8px 6px; }
.ara-welcome-art { height: 118px; max-width: 420px; margin: 0 auto 26px; border: 1px solid var(--border); border-radius: 14px;
  overflow: hidden; background:
  radial-gradient(55% 90% at 18% 30%, #EDE5FF 0%, transparent 70%),
  radial-gradient(45% 80% at 88% 72%, #DCF1EC 0%, transparent 70%),
  radial-gradient(40% 70% at 62% 0%, #FCE3D6 0%, transparent 70%), var(--surface); }
.ara-welcome-art svg { width: 100%; height: 100%; display: block; }
.ara-welcome h1 { font: italic 400 64px/1 var(--serif); letter-spacing: -0.01em; color: var(--text); margin: 0 0 14px; padding: 0; }
.ara-welcome h1 em { font-style: italic; color: var(--violet); }
.ara-welcome p { font-size: 15.5px; line-height: 1.6; color: var(--muted); max-width: 500px; margin: 0 auto 20px; }
.ara-welcome-ready { display: inline-flex; align-items: center; gap: 9px; font-size: 13px; color: var(--muted); text-align: left;
  padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
.ara-chips-label { text-align: center; margin: 18px 0 2px; }

.ara-turn { margin: 4px 0; }
.ara-turn.user { display: flex; justify-content: flex-end; margin-top: 18px; }
.ara-bubble { max-width: 78%; background: var(--elevated); border: 1px solid var(--violet-line); border-radius: 16px 16px 4px 16px;
  padding: 9px 16px 10px; font: italic 400 20px/1.35 var(--serif); color: var(--text); }
.ara-turn.ai { display: grid; grid-template-columns: 30px minmax(0, 1fr); gap: 14px; margin: 16px 0 10px; }
.ara-avatar { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  border: 1px solid var(--violet-line); background: linear-gradient(135deg, var(--violet-soft), var(--cyan-soft)); }
.ara-ai { min-width: 0; }
.ara-ai-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: 10px; margin: 5px 0 10px; font-size: 13px; }
.ara-ai-head b { font: italic 400 19px/1 var(--serif); color: var(--text); }
.ara-ai-head span { color: var(--faint); font-size: 12px; }
.ara-thinking .ara-stage { padding: 4px 0; }

.ara-actions { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px; margin-top: 14px; }
.ara-chip { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; line-height: 1.3; color: var(--muted);
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; user-select: none; }
.ara-chip .ara-dot { width: 6px; height: 6px; box-shadow: none; }
.ara-dot.cyan { background: var(--cyan-fill); }
.ara-chip.conf.high { color: var(--success); background: var(--success-soft); border-color: var(--success-line); }
.ara-chip.conf.mid { color: var(--cyan); background: var(--cyan-soft); border-color: var(--cyan-line); }
.ara-chip.conf.low, .ara-chip.conf.oos { color: var(--amber); background: var(--amber-soft); border-color: var(--amber-line); }
.ara-chip-ico { font-size: 12px; line-height: 1; }
.ara-chip-ico.cyan { color: var(--cyan); } .ara-chip-ico.violet { color: var(--violet); font-weight: 700; }
.ara-drawer > summary { list-style: none; cursor: pointer; transition: border-color .15s, color .15s; }
.ara-drawer > summary::-webkit-details-marker { display: none; }
.ara-drawer > summary:hover { border-color: var(--violet-line); color: var(--text); }
.ara-drawer .caret { font-size: 10px; opacity: .7; transition: transform .15s; }
.ara-drawer[open] { flex-basis: 100%; order: 10; }
.ara-drawer[open] > summary { border-color: var(--violet-line); color: var(--text); }
.ara-drawer[open] > summary .caret { transform: rotate(180deg); }
.ara-drawer-body { margin-top: 12px; }
.ara-drawer-sub { font-size: 12.5px; color: var(--muted); margin: 0 0 10px; }
.ara-oos { padding: 18px 20px; }
.ara-json { margin: 0; padding: 14px 16px; max-height: 380px; overflow: auto; white-space: pre-wrap; word-break: break-word;
  font: 400 12px/1.6 var(--mono); color: var(--text); background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
.ara-drawer-sub code { font: 500 11.5px var(--mono); color: var(--violet); background: var(--violet-soft); padding: 1px 5px; border-radius: 4px; }

/* Chat input (pinned to the bottom by Streamlit) */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stBottom"] { background: linear-gradient(to top, var(--bg) 65%, rgba(250,247,242,0)) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 860px; padding-bottom: 20px; }
[data-testid="stChatInput"] { background: var(--surface) !important; border: 1px solid var(--violet-line) !important; border-radius: 14px !important;
  box-shadow: 0 6px 24px rgba(106,86,204,0.08); }
[data-testid="stChatInput"]:focus-within { border-color: var(--violet) !important; box-shadow: 0 0 0 3px var(--violet-soft); }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-size: 15px !important; caret-color: var(--violet); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--faint) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--violet) !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"] svg { color: #fff !important; fill: #fff !important; }
@media (max-width: 640px) { .ara-bubble { max-width: 90%; } .ara-welcome h1 { font-size: 46px; } .ara-brand-name { font-size: 20px; } }

/* ---------- Sidebar ---------- */
.ara-side-section { margin-bottom: 26px; }
.ara-kb-doc { display: flex; align-items: center; gap: 10px; font: italic 400 21px/1.2 var(--serif); color: var(--text); }
.ara-kb-file { font: 400 11px var(--mono); color: var(--faint); margin: 4px 0 10px; word-break: break-all; }
.ara-kb-state { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); margin-bottom: 12px; }
.ara-kv { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-top: 1px solid var(--border); font-size: 12.5px; }
.ara-kv span { color: var(--muted); }
.ara-kv b { color: var(--text); font-weight: 500; text-align: right; }
.ara-kv b.mono { font: 500 11.5px var(--mono); }
.ara-kv.stack { flex-direction: column; gap: 3px; }
.ara-kv.stack b { text-align: left; }
.ara-nowrap { white-space: nowrap; }
.ara-check { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text); padding: 5px 0; }
.ara-check i { width: 16px; height: 16px; flex: none; display: grid; place-items: center; border-radius: 4px;
  font-style: normal; font-size: 10px; border: 1px solid var(--border); color: var(--faint); }
.ara-check.ok i { border-color: var(--cyan-line); color: var(--cyan); background: var(--cyan-soft); }
.ara-check.pending { color: var(--faint); }
.ara-sys { display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 13px; padding: 5px 0; }
.ara-sys > span { display: inline-flex; align-items: center; gap: 9px; color: var(--text); }
.ara-sys em { font-style: normal; font-size: 12px; color: var(--muted); }

/* ---------- Footer ---------- */
.ara-footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--border); display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 12px; color: var(--faint); }
</style>
""".replace("__VARS__", _VARS)
