
%%writefile app.py
import streamlit as st
import feedparser
import re
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# --- Custom CSS for the Chat Bubbles ---
CHAT_STYLE = """
<style>
.chat-bubble { padding: 10px 15px; border-radius: 20px; max-width: 70%; margin-bottom: 10px; display: inline-block; clear: both; }
.chat-bubble.left { background-color: #333333; float: left; border-bottom-left-radius: 5px; }
.chat-bubble.right { background-color: #005C4B; float: right; border-bottom-right-radius: 5px; }
.speaker-name { font-weight: bold; margin-bottom: 5px; font-size: 0.9em; }
.message-text { font-size: 1em; line-height: 1.4; white-space: pre-wrap; color: #E0E0E0; }
.speaker-1 { color: #FFB3BA; } .speaker-2 { color: #BAFFC9; } .speaker-3 { color: #BAE1FF; } .speaker-4 { color: #FFFFBA; }
.chat-container { width: 100%; overflow: auto; }
</style>
"""

# --- Part 1: Prompts ---
NEWS_FILTER_TEMPLATE = """
You are a sharp political news editor. Your job is to determine if a news summary is interesting enough for a political satire show.
A "satire-worthy" story involves politics, controversy, major policy decisions, or hypocrisy.
Stories about sports, movies, general business, celebrity news or feel-good events are "not interesting."

News Summary: "{news_summary}"

Is this summary satire-worthy? Answer with only "YES" or "NO".
"""
news_filter_prompt = PromptTemplate.from_template(NEWS_FILTER_TEMPLATE)

# --- Part 1: Prompts (Final Version with Strict Formatting) ---
WHATSAPP_TEMPLATE = """
You are an award-winning scriptwriter for a top-tier Indian political satire show. You are a master impersonator of political figures and an expert in the nuances of Indian Hinglish WhatsApp chats.

**Your Task:** Write a fake, "leaked" WhatsApp chat between senior members of a nationalist, right-leaning Indian political party as they react to a news event.

**Character Personas (CRITICAL):**
- **Samit Shah:** The 'Chanakya'. Ruthless, intimidating, and secretive. Obsessed with electoral math, not ideology. Views everything as a political battle to be won at any cost. Always blames the '70 years of the past' for everything.
- **Jogi Raditya:** The Firebrand. Extremely ambitious and authoritarian. Sees 'bulldozer' vigilantism as the *only* solution to every problem, from crime to bad press. Communicates purely in divisive, communal 'us vs. them' rhetoric. Rash and impulsive.
- **S. Rajshankar:** The Technocrat. Arrogant, combative, and condescending. Believes he is the smartest person in any room. Dismisses all criticism as an 'international plot' or 'Western hypocrisy.' His solution is always a 'sharp' (i.e., rude) tweet.
- **Nirmal Siyaraman:** The Defender. Extremely defensive and aggressive, especially with the media. Will attack any journalist who asks a critical question. Master of whataboutism and deflection ("What about...?!"). Can seem completely out of touch with on-the-ground reality (e.g., "I don't eat onions").
- **Mahendra Godi:** The Messianic Leader. Intolerant of any dissent. Speaks only in grand, messianic terms and slogans, viewing himself as the singular savior. Avoids all unscripted questions. His contribution to the chat is usually a vague, emotionally-charged platitude.

**Narrative Arc (Follow this structure):**
1.  **The News Breaks (Initial Reaction):** The chat starts with concern or anger.
2.  **Brainstorming the Spin:** The characters debate how to frame the issue to their advantage.
3.  **The Absurd "Action Plan":** They agree on a ridiculous, over-the-top plan to counter the "narrative."

**Formatting Rule (Follow PRECISELY):**
- Start each message on a new line.
- The format MUST be exactly `Character Name: Message Text`.
- DO NOT add the character name anywhere else in the message.
- DO NOT use markdown bolding (`**`) in the output.

**Example of Correct Output Format:**
Samit Shah: Arre, yeh news dekhi?
Jogi Raditya: Haan sir, trending hai!
S. Rajshankar: Iske peeche international saazish hai.
Nirmal Siyaraman: Let's not jump to conclusions. First, we blame the Opposition.
Mahendra Godi: Mitron, humari sarkar ko log pyar karte hai.

**News Summary to Analyze:** "{news_summary}"

**Chat Parameters to Weave In:**
- **Primary Target of Blame:** {blame}
- **Promise Tone:** {promise_tone}/10 (How wild should the final "action plan" promise be?)
- **Nationalism Level:** {nationalism_level}/10 (How much nationalistic jargon should they use?)
- **Development Focus:** {development_focus}/10 (How much should they talk about grand, unrelated projects?)

**Leaked WhatsApp Chat:**
"""
whatsapp_prompt = PromptTemplate.from_template(WHATSAPP_TEMPLATE)


# --- Part 2: News Engine ---
class NewsEngine:
    def __init__(self):
        self.feeds = {
            "Mainstream Mix (Google)": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
            "The Hindu (National)": "https://www.thehindu.com/news/national/?service=rss",
            "Indian Express (Explained)": "https://indianexpress.com/section/explained/feed/",
            "NDTV (India)": "http://feeds.feedburner.com/NDTV-IndiaNews",
        }

    def get_topics_from_feed(self, url):
        try:
            feed = feedparser.parse(url)
            topics = {}
            for entry in feed.entries[:30]:
                if hasattr(entry, 'summary'):
                    summary_html = BeautifulSoup(entry.summary, 'html.parser')
                    summary_text = summary_html.get_text()
                    if len(summary_text) > 50:
                        topics[entry.title] = summary_text
            return topics
        except Exception:
            return {}

# --- Part 3: Chains and Model Manager ---
class ContentGenerator:
    def __init__(self, api_key: str, model_provider: str):
        self.api_key = api_key
        self.model_provider = model_provider
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if self.model_provider == "groq":
            return ChatGroq(api_key=self.api_key, model="llama-3.1-8b-instant", temperature=0.9)
        elif self.model_provider == "openai":
            return ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo", temperature=0.9)
        return None

    def filter_topics(self, topics_dict):
        worthy_topics = {}
        for title, summary in topics_dict.items():
            chain = LLMChain(llm=self.llm, prompt=news_filter_prompt)
            result = chain.invoke({"news_summary": summary})
            if "YES" in result.get('text', 'NO'):
                worthy_topics[title] = summary
        return worthy_topics

    def generate_whatsapp_chat(self, news_summary, blame, promise_tone, nationalism_level, development_focus):
        chain = LLMChain(llm=self.llm, prompt=whatsapp_prompt)
        result = chain.invoke({"news_summary": news_summary, "blame": blame, "promise_tone": promise_tone, "nationalism_level": nationalism_level, "development_focus": development_focus})
        return result.get('text', 'Failed to generate chat.')

class ModelManager:
    @staticmethod
    def get_available_providers(): return {"Groq (Llama-3.1-8B)": "groq", "OpenAI (GPT-3.5-Turbo)": "openai"}
    @staticmethod
    def get_api_key_name(provider): return {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    @staticmethod
    def validate_api_key(api_key, provider):
        if not api_key: return False
        if provider == "groq": return api_key.startswith("gsk_")
        elif provider == "openai": return api_key.startswith("sk-")
        return True

# --- Part 4: Main Streamlit App ---
st.set_page_config(page_title="Satirical News Engine", layout="wide")
from bs4 import BeautifulSoup

def parse_chat(chat_text):
    messages = []
    pattern = re.compile(r"^\s*(?:\[.*?\]\s*)?([^:]+):\s*(.*)", re.MULTILINE)
    matches = pattern.findall(chat_text)
    for match in matches:
        messages.append({"speaker": match[0].strip(), "message": match[1].strip()})
    return messages

if 'engine' not in st.session_state:
    st.session_state.engine = NewsEngine()

st.title("🎯 Jumla: The Satirical News Engine")

with st.sidebar:
    st.header("🛠️ Configuration")
    providers = ModelManager.get_available_providers()
    selected_provider_name = st.selectbox("LLM Provider", options=list(providers.keys()))
    selected_provider = providers[selected_provider_name]
    api_key = st.text_input(f"Enter {ModelManager.get_api_key_name(selected_provider)}", type="password")

    st.header("📰 News Selection")
    selected_feed_name = st.selectbox("Step 1: Select News Source", options=list(st.session_state.engine.feeds.keys()))

    if st.button("🔍 Find Satirical Topics"):
        if ModelManager.validate_api_key(api_key, selected_provider):
            with st.spinner("AI Editor is scanning the news..."):
                generator = ContentGenerator(api_key, selected_provider)
                feed_url = st.session_state.engine.feeds[selected_feed_name]
                all_topics = st.session_state.engine.get_topics_from_feed(feed_url)
                st.session_state.filtered_topics = generator.filter_topics(all_topics)
                st.session_state.source_for_topics = selected_feed_name # Track which source was filtered
        else:
            st.error("Please enter a valid API key first.")

    if 'filtered_topics' in st.session_state and st.session_state.source_for_topics == selected_feed_name:
        if not st.session_state.filtered_topics:
            st.warning("No satire-worthy news found. Try another source or button again.")
            selected_topic_title = None
        else:
            selected_topic_title = st.selectbox("Step 2: Select a Satire-Worthy Topic", options=list(st.session_state.filtered_topics.keys()))
    else:
        st.info("Click 'Find Satirical Topics' to load headlines.")
        selected_topic_title = None

    st.header("🎭 Satire Parameters")
    blame = st.selectbox("Blame Target", ["Opposition", "Previous Government", "Media", "Foreign Forces"])
    promise_tone = st.slider("Promise Tone (Realistic ↔️ Wild Freebies)", 1, 10, 8)
    nationalism_level = st.slider("Nationalism Level (Subtle ↔️ Overt)", 1, 10, 7)
    development_focus = st.slider("Development Focus (Vague ↔️ Specific Projects)", 1, 10, 5)

if st.button("🚀 Generate Satirical Chat", type="primary", use_container_width=True):
    if ModelManager.validate_api_key(api_key, selected_provider) and selected_topic_title:
        with st.spinner("Briefing the scriptwriters..."):
            news_summary = st.session_state.filtered_topics[selected_topic_title]
            generator = ContentGenerator(api_key, selected_provider)
            whatsapp_chat = generator.generate_whatsapp_chat(news_summary=news_summary, blame=blame, promise_tone=promise_tone, nationalism_level=nationalism_level, development_focus=development_focus)
            st.session_state.summary = news_summary
            st.session_state.title = selected_topic_title
            st.session_state.chat = whatsapp_chat
            st.success("Satire generated!")
    else:
        st.error("Please find and select a topic before generating.")

if 'chat' in st.session_state:
    st.divider()
    st.subheader(f"📰 On: {st.session_state.title}")
    st.write(st.session_state.summary)

    st.subheader("📱 'Leaked' WhatsApp Chat")

    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    parsed_messages = parse_chat(st.session_state.chat)
    speaker_map = {}

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in parsed_messages:
        speaker = msg["speaker"]
        message_text = msg["message"]

        # --- NEW ROBUST CLEANING LOGIC ---
        # 1. Remove speaker name from start/end (case-insensitive)
        name_pattern = r'^\s*\**' + re.escape(speaker) + r'\**\s*:?\s*|' + r'\s*\**' + re.escape(speaker) + r'\**\s*$'
        cleaned_message = re.sub(name_pattern, '', message_text, flags=re.IGNORECASE).strip()
        # 2. Remove any other stray bolding markers
        cleaned_message = cleaned_message.replace('**', '')

        if speaker not in speaker_map:
            speaker_map[speaker] = { "color_class": f"speaker-{(len(speaker_map) % 4) + 1}", "align_class": "right" if len(speaker_map) % 2 != 0 else "left" }

        color, align = speaker_map[speaker]["color_class"], speaker_map[speaker]["align_class"]

        message_html = f"""<div class="chat-bubble {align}"><div class="speaker-name {color}">{speaker}</div><div class="message-text">{cleaned_message}</div></div>"""
        st.markdown(message_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)