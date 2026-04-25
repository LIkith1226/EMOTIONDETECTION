"""
app.py — Emotion Text Detection from Text
Light, colorful, modern UI · orange brand · unique color per emotion

Created by: Likith Abhiram Jaldu and Vishwanath Reddy
"""

import os
import sys
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from model import predict_emotion, load_model, train_and_evaluate, get_top_words_for_emotion


st.set_page_config(
    page_title="Emotion Text Detection — Detect Feelings in Text",
    page_icon=None,
    layout="centered",
)


# ── Unique colour per emotion ─────────────────────────────────────────────────

EMOTION_PALETTE = {
    "sadness":    "#3B82F6",
    "joy":        "#F59E0B",
    "love":       "#EC4899",
    "anger":      "#EF4444",
    "fear":       "#9333EA",
    "surprise":   "#C026D3",
}

EMOTION_DESCRIPTIONS = {
    "sadness":    "You're experiencing sorrow or melancholy.",
    "joy":        "You're feeling happy and positive!",
    "love":       "You're expressing deep affection and care.",
    "anger":      "Something has frustrated or outraged you.",
    "fear":       "Something is frightening or threatening to you.",
    "surprise":   "Something unexpected has caught you off guard.",
}


SAMPLES = [
    ("Joy",            "#F59E0B", "I am so happy I got the job! Best day of my life!"),
    ("Anger",          "#EF4444", "I hate this so much. I am so angry right now."),
    ("Love",           "#EC4899", "I love you so much. You mean everything to me."),
    ("Fear",           "#9333EA", "I am so scared and terrified. I cannot stop shaking."),
    ("Sadness",        "#3B82F6", "I feel so sad and I miss them so much. The pain is unbearable."),
    ("Surprise",       "#C026D3", "Wow! I cannot believe this happened. I am completely shocked!"),
]


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Base reset to light ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #F4F6FF !important;
    color: #1A1F5E !important;
}
.stApp            { background: #F4F6FF !important; }
.block-container  { padding: 0 1.8rem 6rem !important; max-width: 820px !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Textarea ── */
.stTextArea > label { display: none !important; }
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 2px solid #E2E8F8 !important;
    border-radius: 16px !important;
    color: #1A1F5E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
    padding: 20px 22px !important;
    box-shadow: 0 4px 24px rgba(67,97,238,0.08) !important;
    caret-color: #FF6B35 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 4px 28px rgba(255,107,53,0.18) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #C4CADF !important; }

/* ── Primary button — orange gradient pill ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #FFFFFF !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    height: 52px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 8px 28px rgba(255,107,53,0.35) !important;
    transition: all 0.22s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 36px rgba(255,107,53,0.52) !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 6px 20px rgba(255,107,53,0.35) !important;
}

/* ── Secondary button — outlined navy ── */
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F8 !important;
    border-radius: 50px !important;
    color: #1A1F5E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    height: 44px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(26,31,94,0.06) !important;
    transition: all 0.18s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #FF6B35 !important;
    color: #FF6B35 !important;
    box-shadow: 0 4px 16px rgba(255,107,53,0.14) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander — white card ── */
details {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F8 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin: 8px 0 !important;
    box-shadow: 0 2px 12px rgba(26,31,94,0.06) !important;
}
details > summary {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #1A1F5E !important;
    padding: 16px 22px !important;
    cursor: pointer !important;
    list-style: none !important;
}
details > summary::-webkit-details-marker { display: none; }
details > summary:hover { color: #FF6B35 !important; }
details[open] > summary {
    color: #FF6B35 !important;
    border-bottom: 1.5px solid #E2E8F8 !important;
}
.streamlit-expanderContent {
    padding: 22px !important;
    color: #475569 !important;
    font-size: 0.88rem !important;
    line-height: 1.85 !important;
}

/* ── Misc ── */
hr {
    border: none !important;
    border-top: 1.5px solid #E2E8F8 !important;
    margin: 36px 0 !important;
}
.stSpinner > div { border-top-color: #FF6B35 !important; }
div[data-testid="stAlert"] {
    background: #FFF7ED !important;
    border: 1.5px solid #FED7AA !important;
    border-radius: 12px !important;
    color: #9A3412 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    try:
        return load_model()
    except FileNotFoundError:
        return train_and_evaluate()


# ── HTML builders ─────────────────────────────────────────────────────────────

def _pill(text: str, color: str = "#FF6B35") -> str:
    return (
        f'<span style="display:inline-block;padding:6px 16px;'
        f'background:{color}18;border:1px solid {color}44;'
        f'border-radius:50px;color:{color};font-size:0.7rem;'
        f'font-weight:700;letter-spacing:0.1em;'
        f'font-family:Poppins,sans-serif;text-transform:uppercase">'
        f'{text}</span>'
    )


def _stat_card(number: str, label: str, color: str) -> str:
    return (
        f'<div style="background:#FFFFFF;border-radius:14px;padding:16px 18px;'
        f'box-shadow:0 2px 16px rgba(26,31,94,0.07);text-align:center;flex:1;min-width:0">'
        f'  <div style="font-family:Poppins,sans-serif;font-size:1.5rem;'
        f'       font-weight:800;color:{color};line-height:1">{number}</div>'
        f'  <div style="font-size:0.7rem;font-weight:500;color:#94A3B8;'
        f'       margin-top:4px;letter-spacing:0.05em;text-transform:uppercase">'
        f'    {label}'
        f'  </div>'
        f'</div>'
    )


def _result_card(emotion: str, confidence: float, description: str) -> str:
    color = EMOTION_PALETTE.get(emotion, "#64748B")
    return (
        f'<div style="background:#FFFFFF;border-radius:20px;padding:28px 32px;'
        f'box-shadow:0 8px 40px rgba(26,31,94,0.10);'
        f'border-left:5px solid {color};margin:20px 0 8px">'

        f'  <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px">'
        f'    <div style="width:48px;height:48px;border-radius:12px;'
        f'         background:{color}20;display:flex;align-items:center;'
        f'         justify-content:center;flex-shrink:0">'
        f'      <div style="width:14px;height:14px;border-radius:50%;background:{color}"></div>'
        f'    </div>'
        f'    <div>'
        f'      <div style="font-family:Poppins,sans-serif;font-size:1.8rem;'
        f'           font-weight:800;color:{color};line-height:1">'
        f'        {emotion.upper()}'
        f'      </div>'
        f'      <div style="font-size:0.78rem;color:#94A3B8;margin-top:3px">'
        f'        {confidence:.1%} confidence'
        f'      </div>'
        f'    </div>'
        f'  </div>'

        f'  <div style="font-size:0.9rem;color:#475569;margin-bottom:18px;'
        f'       font-style:italic;line-height:1.5">'
        f'    {description}'
        f'  </div>'

        f'  <div style="background:#F4F6FF;border-radius:99px;height:10px;overflow:hidden">'
        f'    <div style="height:100%;width:{confidence*100:.1f}%;'
        f'         background:linear-gradient(90deg,{color},{color}99);'
        f'         border-radius:99px;transition:width 0.6s ease"></div>'
        f'  </div>'

        f'</div>'
    )


def _prob_chip(emotion: str, prob: float, is_top: bool) -> str:
    color = EMOTION_PALETTE.get(emotion, "#94A3B8")
    bg    = f"{color}18" if is_top else "#F4F6FF"
    bc    = f"{color}44" if is_top else "#E2E8F8"
    tc    = color if is_top else "#94A3B8"
    fw    = "600" if is_top else "400"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:5px 12px;background:{bg};border:1px solid {bc};'
        f'border-radius:50px;margin:3px;white-space:nowrap">'
        f'  <span style="width:7px;height:7px;border-radius:50%;'
        f'        background:{tc};flex-shrink:0"></span>'
        f'  <span style="font-size:0.76rem;font-weight:{fw};color:{tc}">'
        f'    {emotion}&nbsp;{prob*100:.0f}%'
        f'  </span>'
        f'</span>'
    )


def _word_chip(word: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:5px 14px;margin:3px;'
        f'background:{color}14;border:1px solid {color}35;color:{color};'
        f'border-radius:50px;font-size:0.76rem;font-weight:600;'
        f'white-space:nowrap">{word}</span>'
    )


def _section_heading(text: str, top: str = "32px") -> str:
    return (
        f'<div style="margin-top:{top};margin-bottom:20px">'
        f'  <span style="font-family:Poppins,sans-serif;font-size:1.1rem;'
        f'        font-weight:700;color:#1A1F5E">{text}</span>'
        f'</div>'
    )


# ── Page ──────────────────────────────────────────────────────────────────────

# Load model early (cached)
with st.spinner(""):
    pipeline = get_pipeline()

# ── Fake nav ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:20px 0 10px;border-bottom:1.5px solid #E2E8F8;margin-bottom:0">
  <span style="font-family:'Poppins',sans-serif;font-size:1.25rem;
        font-weight:800;color:#1A1F5E">
    Emotion<span style="color:#FF6B35">Text</span> Detection
  </span>
  <div style="display:flex;gap:24px;align-items:center">
    <span style="font-size:0.82rem;font-weight:500;color:#94A3B8;cursor:pointer">How it works</span>
    <span style="font-size:0.82rem;font-weight:500;color:#94A3B8;cursor:pointer">Examples</span>
    <span style="font-size:0.82rem;font-weight:500;color:#94A3B8;cursor:pointer">About</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:3.5rem 0 2rem;text-align:center">

  <div style="margin-bottom:20px">
    {_pill("AI-Powered · 6 Core Emotions · dair-ai/emotion Dataset")}
  </div>

  <h1 style="font-family:'Poppins',sans-serif;font-weight:900;
       font-size:3rem;line-height:1.15;color:#1A1F5E;margin:0 0 16px">
    Detect the&nbsp;
    <span style="color:#FF6B35;position:relative;display:inline-block">
      Emotion
      <svg style="position:absolute;bottom:-6px;left:0;width:100%;height:6px"
           viewBox="0 0 100 6" preserveAspectRatio="none">
        <path d="M0,5 Q25,0 50,5 Q75,10 100,5" stroke="#FF6B35" stroke-width="2.5"
              fill="none" stroke-linecap="round"/>
      </svg>
    </span>
    &nbsp;in&nbsp;
    <span style="color:#FF6B35;position:relative;display:inline-block">
      Text
      <svg style="position:absolute;bottom:-6px;left:0;width:100%;height:6px"
           viewBox="0 0 100 6" preserveAspectRatio="none">
        <path d="M0,5 Q25,0 50,5 Q75,10 100,5" stroke="#FF6B35" stroke-width="2.5"
              fill="none" stroke-linecap="round"/>
      </svg>
    </span>
  </h1>

  <p style="font-size:1rem;color:#94A3B8;max-width:460px;margin:0 auto 28px;line-height:1.75">
    Paste any text and our machine learning model will detect
    its emotional tone across 6 core emotions in seconds.
  </p>

</div>
""", unsafe_allow_html=True)


# ── Stats row ─────────────────────────────────────────────────────────────────
s1 = _stat_card("6",      "Emotions",        "#FF6B35")
s2 = _stat_card("16K+",   "Training Texts",  "#4361EE")
s3 = _stat_card("92%+",   "Accuracy",        "#10B981")
s4 = _stat_card("Twitter","Data Source",     "#EC4899")
st.markdown(
    f'<div style="display:flex;gap:12px;margin-bottom:36px">{s1}{s2}{s3}{s4}</div>',
    unsafe_allow_html=True,
)


# ── Input card ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#FFFFFF;border-radius:20px;padding:28px 28px 20px;
     box-shadow:0 4px 32px rgba(26,31,94,0.09);margin-bottom:0">
  <div style="font-family:'Poppins',sans-serif;font-weight:700;
       font-size:0.9rem;color:#1A1F5E;margin-bottom:12px;letter-spacing:0.01em">
    What text do you want to analyse?
  </div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "",
    placeholder="Type or paste any sentence, tweet, review or message here ...",
    height=120,
    key="main_input",
    label_visibility="collapsed",
)

c1, c2 = st.columns([4, 1])
with c1:
    go = st.button("Detect Emotion  →", type="primary")
with c2:
    if st.button("Clear"):
        st.session_state["main_input"] = ""
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ── Result ────────────────────────────────────────────────────────────────────

if go:
    raw = (st.session_state.get("main_input") or "").strip()
    if not raw:
        st.warning("Please enter some text above to detect the emotion.")
    else:
        with st.spinner("Analysing your text ..."):
            emotion, confidence, all_probs = predict_emotion(raw, pipeline)

        color       = EMOTION_PALETTE.get(emotion, "#64748B")
        description = EMOTION_DESCRIPTIONS.get(emotion, "")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(_section_heading("Result", top="0"), unsafe_allow_html=True)

        # Primary result card
        st.markdown(_result_card(emotion, confidence, description), unsafe_allow_html=True)

        # All emotion probability chips
        st.markdown(
            _section_heading("All 6 Emotions", top="24px"),
            unsafe_allow_html=True,
        )
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        chips = "".join(_prob_chip(e, p, i < 3) for i, (e, p) in enumerate(sorted_probs))
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:16px">{chips}</div>',
            unsafe_allow_html=True,
        )

        # Ranked bar breakdown (top 8)
        st.markdown(
            _section_heading("Confidence Breakdown", top="8px"),
            unsafe_allow_html=True,
        )
        for i, (e, p) in enumerate(sorted_probs[:8]):
            ec    = EMOTION_PALETTE.get(e, "#94A3B8")
            is_t  = i == 0
            fw    = "700" if is_t else "500"
            tc    = ec if is_t else "#94A3B8"
            bg    = f"{ec}10" if is_t else "transparent"
            st.markdown(
                f'<div style="display:grid;grid-template-columns:130px 1fr 52px;'
                f'     align-items:center;gap:12px;padding:10px 12px;'
                f'     border-radius:10px;margin:3px 0;background:{bg}">'
                f'  <div style="display:flex;align-items:center;gap:8px;min-width:0">'
                f'    <div style="width:8px;height:8px;border-radius:50%;'
                f'         background:{ec};flex-shrink:0"></div>'
                f'    <span style="font-size:0.78rem;font-weight:{fw};color:{tc};'
                f'          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'      {e.capitalize()}'
                f'    </span>'
                f'  </div>'
                f'  <div style="background:#E2E8F8;border-radius:99px;height:7px;overflow:hidden">'
                f'    <div style="height:100%;width:{p*100:.1f}%;'
                f'         background:linear-gradient(90deg,{ec},{ec}88);'
                f'         border-radius:99px"></div>'
                f'  </div>'
                f'  <div style="text-align:right;font-size:0.78rem;font-weight:{fw};'
                f'       color:{tc};white-space:nowrap">{p*100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Key words
        with st.expander("Key vocabulary signals for this prediction"):
            words = get_top_words_for_emotion(emotion, pipeline, n=16)
            chips_html = "".join(_word_chip(w, color) for w in words)
            st.markdown(
                f'<div style="line-height:2.5;margin-bottom:12px">{chips_html}</div>'
                f'<p style="color:#94A3B8;font-size:0.76rem;margin:0">'
                f'These are the vocabulary features with the highest model coefficient '
                f'for <strong style="color:{color}">{emotion.upper()}</strong>.</p>',
                unsafe_allow_html=True,
            )


# ── Examples ──────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(_section_heading("Try an Example", top="0"), unsafe_allow_html=True)

# 2-column grid of example buttons
pairs = [(SAMPLES[i], SAMPLES[i+1] if i+1 < len(SAMPLES) else None)
         for i in range(0, len(SAMPLES), 2)]

for row in pairs:
    cols = st.columns(2)
    for col, sample in zip(cols, row):
        if sample is None:
            break
        label, color, text = sample
        with col:
            preview = text[:52] + "..." if len(text) > 52 else text
            # Styled card via HTML + button below
            st.markdown(
                f'<div style="background:#FFFFFF;border-radius:14px;padding:14px 16px 8px;'
                f'box-shadow:0 2px 12px rgba(26,31,94,0.07);border-top:3px solid {color};'
                f'margin-bottom:2px">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    <div style="width:9px;height:9px;border-radius:50%;'
                f'         background:{color}"></div>'
                f'    <span style="font-family:Poppins,sans-serif;font-size:0.78rem;'
                f'          font-weight:700;color:{color}">{label}</span>'
                f'  </div>'
                f'  <p style="font-size:0.78rem;color:#94A3B8;margin:0;line-height:1.5;'
                f'     white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'    {preview}'
                f'  </p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Try this", key=f"ex_{label}"):
                st.session_state["main_input"] = text
                st.rerun()


# ── Info section ──────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(_section_heading("Learn More", top="0"), unsafe_allow_html=True)

with st.expander("How does it work?"):
    st.markdown("""
**TF-IDF** (Term Frequency — Inverse Document Frequency) converts text into numbers. Words that are frequent in *this* text but rare across all training texts score highest — emotionally loaded words score high, filler words score near zero.

**Negation handling** — "I do not care" becomes the token `not_care` before both training and prediction. This ensures the model doesn't confuse "care" (positive) inside a negative phrase.

**Bigrams** capture two-word context — "love you", "can't stand", "not happy" — far more informative than single words alone.

**Logistic Regression** with `class_weight='balanced'` handles the natural frequency imbalance between emotion categories in the dair-ai/emotion dataset.

**Dataset** — Trained on the dair-ai/emotion dataset from HuggingFace, which contains ~16,000 tweets labeled with 6 core emotions, providing accurate real-world emotion detection.
""")

with st.expander("What are the 6 emotions?"):
    emotion_list = ", ".join(
        f'<span style="color:{EMOTION_PALETTE.get(e, "#94A3B8")};font-weight:600">{e}</span>'
        for e in sorted(EMOTION_PALETTE.keys())
    )
    st.markdown(
        f'<p style="line-height:2.2;font-size:0.88rem">{emotion_list}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
The dair-ai/emotion dataset contains tweets annotated with 6 core emotions: **sadness, joy, love, anger, fear, and surprise**. 
This focused emotion set provides more accurate and reliable predictions across these fundamental human emotions.
""")

with st.expander("Where can I use this?"):
    st.markdown("""
**Customer Support** — Automatically classify tickets by emotional tone. Route distressed or angry customers to a priority queue.

**Social Media** — Monitor brand perception in real time. Detect emotional spikes during product launches or incidents.

**Feedback Analysis** — Segment reviews by emotion. Identify which product areas generate the most disappointment or delight.

**Mental Health Tech** — Track emotional patterns in journal entries or chat conversations over time.
""")

with st.expander("Known limitations"):
    st.markdown("""
— **Sarcasm** is not handled. "Oh great, another Monday" may classify as approval.

— **Ambiguous short text** — single words give low-confidence results.

— **Mixed emotions** — only one label is returned even when the text carries several.

— **English only** — trained on English Reddit comments.
""")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 0;border-top:1.5px solid #E2E8F8;
     margin-top:40px">
  <span style="font-family:'Poppins',sans-serif;font-size:1rem;
        font-weight:800;color:#1A1F5E">
    Emotion<span style="color:#FF6B35">Text</span> Detection
  </span>
  <p style="font-size:0.72rem;color:#C4CADF;margin:8px 0 0;letter-spacing:0.05em">
    Created by Likith Abhiram Jaldu and Vishwanath Reddy<br>
    Scikit-learn · Streamlit · dair-ai/emotion Dataset · TF-IDF + Logistic Regression
  </p>
</div>
""", unsafe_allow_html=True)
