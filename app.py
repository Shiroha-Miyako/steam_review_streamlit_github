import time
from typing import List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# =========================
# Configuration
# =========================
SENTIMENT_MODEL_ID = "ShirohaNaruse/CustomModel_steam_sentiment"
ISSUE_MODEL_ID = "ShirohaNaruse/CustomModel_steam_issue"

APP_TITLE = "Steam Review Intelligence Assistant"
FETCH_MAX_REVIEWS = 1000

ISSUE_LABEL_MAP = {
    "LABEL_0": "Bug / Crash",
    "LABEL_1": "Account / Access",
    "LABEL_2": "Multiplayer / Server",
    "LABEL_3": "Performance",
    "LABEL_4": "Gameplay",
    "LABEL_5": "Content",
    "LABEL_6": "Price / Value",
    "LABEL_7": "Emotional Expression",
    0: "Bug / Crash",
    1: "Account / Access",
    2: "Multiplayer / Server",
    3: "Performance",
    4: "Gameplay",
    5: "Content",
    6: "Price / Value",
    7: "Emotional Expression",
    "0": "Bug / Crash",
    "1": "Account / Access",
    "2": "Multiplayer / Server",
    "3": "Performance",
    "4": "Gameplay",
    "5": "Content",
    "6": "Price / Value",
    "7": "Emotional Expression",
    "Bug / Crash": "Bug / Crash",
    "Account / Access": "Account / Access",
    "Multiplayer / Server": "Multiplayer / Server",
    "Performance": "Performance",
    "Gameplay": "Gameplay",
    "Content": "Content",
    "Price / Value": "Price / Value",
    "Emotional Expression": "Emotional Expression",
}

SENTIMENT_LABEL_MAP = {
    "LABEL_0": "Negative / Not Recommended",
    "LABEL_1": "Positive / Recommended",
    0: "Negative / Not Recommended",
    1: "Positive / Recommended",
    "0": "Negative / Not Recommended",
    "1": "Positive / Recommended",
    "NEGATIVE": "Negative / Not Recommended",
    "POSITIVE": "Positive / Recommended",
    "Negative": "Negative / Not Recommended",
    "Positive": "Positive / Recommended",
    "Negative / Not Recommended": "Negative / Not Recommended",
    "Positive / Recommended": "Positive / Recommended",
}

ACTION_MAP = {
    "Bug / Crash": "Prioritize bug fixing and investigate crash logs or broken gameplay flows.",
    "Account / Access": "Check login, account binding, phone verification, suspension, ban, and access-related complaints.",
    "Multiplayer / Server": "Improve server stability, matchmaking, connection quality, and online experience.",
    "Performance": "Optimize FPS, loading time, memory usage, and hardware compatibility.",
    "Gameplay": "Review core mechanics, controls, difficulty balance, character balance, and progression design.",
    "Content": "Review content updates, maps, heroes, skins, story content, replay value, and content roadmap communication.",
    "Price / Value": "Review pricing, monetization, refund reasons, battle pass value, and perceived content value.",
    "Emotional Expression": "Review as general player reaction. Positive expressions may support marketing, while negative emotions may indicate dissatisfaction without a specific actionable issue.",
    "General": "Review manually if the comment receives many votes or appears in negative feedback.",
}

ISSUE_ORDER = [
    "Bug / Crash",
    "Account / Access",
    "Multiplayer / Server",
    "Performance",
    "Gameplay",
    "Content",
    "Price / Value",
    "Emotional Expression",
]

PRIORITY_ORDER = ["High", "Medium", "Marketing Insight", "Low"]
PRIORITY_RANK = {
    "High": 0,
    "Medium": 1,
    "Marketing Insight": 2,
    "Low": 3,
}

DISPLAY_COLUMNS = [
    "review_text",
    "sentiment",
    "sentiment_confidence",
    "issue_category",
    "issue_confidence",
    "priority",
]


# =========================
# Model loading
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
    sentiment_tokenizer.model_input_names = ["input_ids", "attention_mask"]

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_ID
    )

    issue_tokenizer = AutoTokenizer.from_pretrained(ISSUE_MODEL_ID)
    issue_tokenizer.model_input_names = ["input_ids", "attention_mask"]

    issue_model = AutoModelForSequenceClassification.from_pretrained(
        ISSUE_MODEL_ID
    )

    sentiment_pipe = pipeline(
        "text-classification",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        truncation=True,
        max_length=256,
    )

    issue_pipe = pipeline(
        "text-classification",
        model=issue_model,
        tokenizer=issue_tokenizer,
        truncation=True,
        max_length=256,
    )

    return sentiment_pipe, issue_pipe


# =========================
# Helper functions
# =========================
def normalize_sentiment_label(raw_label):
    return SENTIMENT_LABEL_MAP.get(
        raw_label,
        SENTIMENT_LABEL_MAP.get(str(raw_label), str(raw_label)),
    )


def normalize_issue_label(raw_label):
    return ISSUE_LABEL_MAP.get(
        raw_label,
        ISSUE_LABEL_MAP.get(str(raw_label), str(raw_label)),
    )


def is_negative(sentiment: str) -> bool:
    return str(sentiment).lower().startswith("negative")


def get_review_status(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"

    negative_share = df["sentiment"].apply(is_negative).mean() * 100
    positive_share = 100 - negative_share

    if positive_share >= 80:
        return "Very Positive"
    elif positive_share >= 70:
        return "Mostly Positive"
    elif positive_share >= 40:
        return "Mixed"
    elif positive_share >= 20:
        return "Mostly Negative"
    else:
        return "Very Negative"


def assign_priority(sentiment: str, issue_category: str) -> str:
    if is_negative(sentiment) and issue_category in [
        "Bug / Crash",
        "Account / Access",
        "Multiplayer / Server",
        "Performance",
    ]:
        return "High"

    if is_negative(sentiment) and issue_category in [
        "Gameplay",
        "Content",
        "Price / Value",
    ]:
        return "Medium"

    if (not is_negative(sentiment)) and issue_category == "Emotional Expression":
        return "Marketing Insight"

    return "Low"


def get_suggested_action(issue_category: str) -> str:
    return ACTION_MAP.get(issue_category, ACTION_MAP["General"])


def sort_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sorted_df = df.copy()
    sorted_df["_priority_rank"] = sorted_df["priority"].map(PRIORITY_RANK).fillna(99)
    sorted_df = sorted_df.sort_values(
        by=["_priority_rank", "issue_category", "issue_confidence"],
        ascending=[True, True, False],
    ).drop(columns=["_priority_rank"])

    return sorted_df.reset_index(drop=True)


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    available_columns = [col for col in DISPLAY_COLUMNS if col in df.columns]
    display_df = df[available_columns].copy()
    display_df = sort_by_priority(display_df)
    return display_df


def analyze_texts(
    texts: List[str],
    sentiment_pipe,
    issue_pipe,
    batch_size: int = 16,
) -> pd.DataFrame:
    clean_texts = [str(x) if pd.notna(x) else "" for x in texts]
    clean_texts = [x.strip() for x in clean_texts]

    sentiment_outputs = sentiment_pipe(clean_texts, batch_size=batch_size)
    issue_outputs = issue_pipe(clean_texts, batch_size=batch_size)

    rows = []

    for text, sentiment_out, issue_out in zip(
        clean_texts,
        sentiment_outputs,
        issue_outputs,
    ):
        sentiment = normalize_sentiment_label(sentiment_out.get("label"))
        issue_category = normalize_issue_label(issue_out.get("label"))
        priority = assign_priority(sentiment, issue_category)

        rows.append(
            {
                "review_text": text,
                "sentiment": sentiment,
                "sentiment_confidence": round(float(sentiment_out.get("score", 0)), 4),
                "issue_category": issue_category,
                "issue_confidence": round(float(issue_out.get("score", 0)), 4),
                "priority": priority,
                "suggested_action": get_suggested_action(issue_category),
            }
        )

    return pd.DataFrame(rows)


def get_top_issue_summary(df: pd.DataFrame, sentiment_type: str) -> str:
    if df.empty:
        return "No reviews in this group."

    issue_counts = df["issue_category"].value_counts()

    if issue_counts.empty:
        return "No issue category detected."

    top_issue = issue_counts.index[0]
    top_count = int(issue_counts.iloc[0])
    share = top_count / len(df) * 100

    return (
        f"{sentiment_type} reviews are mainly associated with **{top_issue}** "
        f"({top_count:,} reviews, {share:.1f}% of {sentiment_type.lower()} reviews)."
    )


def generate_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No reviews were analyzed."

    total = len(df)
    negative_df = df[df["sentiment"].apply(is_negative)]
    positive_df = df[~df["sentiment"].apply(is_negative)]

    positive_count = len(positive_df)
    negative_count = len(negative_df)

    positive_share = positive_count / total * 100
    negative_share = negative_count / total * 100

    review_status = get_review_status(df)
    high_priority_count = int((df["priority"] == "High").sum())

    positive_issue_sentence = get_top_issue_summary(positive_df, "Positive")
    negative_issue_sentence = get_top_issue_summary(negative_df, "Negative")

    if high_priority_count > 0:
        high_df = df[df["priority"] == "High"]
        high_issue_counts = high_df["issue_category"].value_counts()
        top_high_issue = high_issue_counts.index[0]
        high_sentence = (
            f"There are **{high_priority_count:,} high-priority reviews**, "
            f"mainly concentrated in **{top_high_issue}**. "
            "These are the reviews that should be checked first because they are negative and linked to technical, access, server, or performance issues."
        )
    else:
        high_sentence = (
            "No high-priority review cluster was detected in this batch. "
            "The current batch is more useful for understanding general sentiment and lower-risk feedback themes."
        )

    return (
        f"**Review status:** {review_status}. "
        f"This batch contains **{total:,} reviews**: **{positive_count:,} positive** "
        f"({positive_share:.1f}%) and **{negative_count:,} negative** ({negative_share:.1f}%). "
        f"{positive_issue_sentence} {negative_issue_sentence} {high_sentence}"
    )


# =========================
# Dashboard rendering
# =========================
def render_kpi_cards(df: pd.DataFrame):
    total = len(df)

    if total:
        negative_count = int(df["sentiment"].apply(is_negative).sum())
        positive_count = total - negative_count
    else:
        negative_count = 0
        positive_count = 0

    high_priority = int((df["priority"] == "High").sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Positive Reviews", f"{positive_count:,}")
    c3.metric("Negative Reviews", f"{negative_count:,}")
    c4.metric("High Priority", f"{high_priority:,}")


def render_interactive_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    st.subheader("Interactive Review Explorer")
    st.write(
        "Use the filters below to focus on one issue category or one priority level. "
        "The dashboard and detailed table update automatically without fetching reviews again."
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        issue_options = ["All"] + [x for x in ISSUE_ORDER if x in set(df["issue_category"])]
        selected_issue = st.selectbox(
            "Issue Category",
            issue_options,
            key=f"{key_prefix}_issue_filter",
        )

    with filter_col2:
        priority_options = ["All"] + [x for x in PRIORITY_ORDER if x in set(df["priority"])]
        selected_priority = st.selectbox(
            "Priority",
            priority_options,
            key=f"{key_prefix}_priority_filter",
        )

    filtered_df = df.copy()

    if selected_issue != "All":
        filtered_df = filtered_df[filtered_df["issue_category"] == selected_issue]

    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

    filtered_df = sort_by_priority(filtered_df)

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} analyzed reviews after filtering."
    )

    return filtered_df


def build_issue_statistics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Issue Category",
                "Total",
                "Positive",
                "Negative",
                "High Priority",
            ]
        )

    rows = []

    for issue in ISSUE_ORDER:
        issue_df = df[df["issue_category"] == issue]

        if issue_df.empty:
            continue

        negative_count = int(issue_df["sentiment"].apply(is_negative).sum())
        positive_count = len(issue_df) - negative_count
        high_priority_count = int((issue_df["priority"] == "High").sum())

        rows.append(
            {
                "Issue Category": issue,
                "Total": len(issue_df),
                "Positive": positive_count,
                "Negative": negative_count,
                "High Priority": high_priority_count,
            }
        )

    stats_df = pd.DataFrame(rows)

    if not stats_df.empty:
        stats_df = stats_df.sort_values(by="Total", ascending=False).reset_index(drop=True)

    return stats_df


def render_issue_overview(df: pd.DataFrame):
    if df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("Issue Overview")

    left_col, right_col = st.columns([2, 1])

    with left_col:
        heatmap_data = (
            df.groupby(["issue_category", "sentiment"])
            .size()
            .reset_index(name="count")
        )

        if not heatmap_data.empty:
            pivot = (
                heatmap_data
                .pivot(index="issue_category", columns="sentiment", values="count")
                .fillna(0)
            )
            pivot = pivot.reindex([x for x in ISSUE_ORDER if x in pivot.index])

            fig = px.imshow(
                pivot,
                text_auto=True,
                aspect="auto",
                title="Issue Category × Sentiment",
            )
            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Issue Category",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right_col:
        stats_df = build_issue_statistics(df)
        st.markdown("#### Issue Totals")
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            height=520,
        )


def render_table_field_explanation():
    st.markdown(
        """
**Field explanation**

| Field | Meaning |
|---|---|
| `review_text` | Original player review text. |
| `sentiment` | Predicted review sentiment: positive/recommended or negative/not recommended. |
| `sentiment_confidence` | Model confidence for the sentiment prediction. |
| `issue_category` | Predicted developer-oriented issue category. |
| `issue_confidence` | Model confidence for the issue-category prediction. |
| `priority` | Rule-based priority generated from sentiment and issue category. |
        """
    )


def render_analysis_results(df: pd.DataFrame, key_prefix: str):
    st.subheader("Executive Summary")
    render_kpi_cards(df)
    st.write(generate_summary(df))

    filtered_df = render_interactive_filter(df, key_prefix=key_prefix)

    if filtered_df.empty:
        st.warning("No reviews match the selected filters.")
        return

    render_issue_overview(filtered_df)

    st.subheader("Filtered Review Table")
    render_table_field_explanation()

    display_df = prepare_display_table(filtered_df)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=460,
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download filtered table as CSV",
        data=csv_bytes,
        file_name="filtered_review_table.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_filtered",
    )


# =========================
# Steam review fetching
# =========================
def fetch_steam_reviews_by_app_id(
    app_id: str,
    max_reviews: int = 100,
    language: str = "english",
) -> pd.DataFrame:
    app_id = str(app_id).strip()

    if not app_id.isdigit():
        raise ValueError("Steam App ID must be a numeric ID, such as 1086940.")

    max_reviews = int(max_reviews)
    max_reviews = max(1, min(max_reviews, FETCH_MAX_REVIEWS))

    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    reviews = []
    cursor = "*"

    while len(reviews) < max_reviews:
        params = {
            "json": 1,
            "filter": "recent",
            "language": language,
            "num_per_page": min(100, max_reviews - len(reviews)),
            "cursor": cursor,
            "purchase_type": "all",
        }

        response = requests.get(
            base_url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Steam request failed with status code {response.status_code}."
            )

        data = response.json()

        if not data.get("success"):
            raise RuntimeError("Steam review endpoint returned an unsuccessful response.")

        batch = data.get("reviews", [])

        if not batch:
            break

        for item in batch:
            review_text = item.get("review", "")

            if review_text and str(review_text).strip():
                reviews.append(
                    {
                        "review_text": review_text,
                        "voted_up": item.get("voted_up", None),
                        "votes_up": item.get("votes_up", 0),
                        "weighted_vote_score": item.get("weighted_vote_score", 0),
                        "timestamp_created": item.get("timestamp_created", None),
                    }
                )

        new_cursor = data.get("cursor")

        if not new_cursor or new_cursor == cursor:
            break

        cursor = new_cursor
        time.sleep(0.25)

    return pd.DataFrame(reviews).head(max_reviews)


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Steam Review Intelligence Assistant")
st.caption(
    "A developer-oriented review triage tool using two fine-tuned Hugging Face DistilBERT models: "
    "sentiment classification and issue category classification."
)

try:
    sentiment_pipe, issue_pipe = load_models()
except Exception as exc:
    st.error(
        "Model loading failed. Please check the Hugging Face model IDs and requirements.txt."
    )
    st.exception(exc)
    st.stop()


if "steam_result_df" not in st.session_state:
    st.session_state["steam_result_df"] = None

if "steam_app_id" not in st.session_state:
    st.session_state["steam_app_id"] = None

if "csv_result_df" not in st.session_state:
    st.session_state["csv_result_df"] = None


tab_steam, tab_csv, tab_about = st.tabs(
    [
        "Fetch by Steam App ID",
        "Batch CSV Dashboard",
        "About the Model",
    ]
)


with tab_steam:
    st.header("Fetch Steam Reviews by App ID")

    st.write(
        "Enter a numeric Steam App ID. You can find it in the Steam store URL. "
        "For example, in `https://store.steampowered.com/app/1086940/...`, "
        "the App ID is `1086940`."
    )

    st.info(
        "This feature uses Steam's public review endpoint. For live demos, "
        "fetching 100–200 reviews is usually more stable than very large batches."
    )

    app_id = st.text_input("Steam App ID", value="1086940")

    review_count = st.selectbox(
        "Number of recent English reviews to fetch",
        [20, 50, 100, 200, 500, 1000],
        index=2,
    )

    if review_count >= 500:
        st.warning(
            "Large fetches may take longer on Streamlit Cloud. "
            "For classroom demos, 100–200 reviews is safer."
        )

    fetch_clicked = st.button("Fetch and Analyze Steam Reviews", type="primary")

    if fetch_clicked:
        try:
            with st.spinner("Fetching recent Steam reviews..."):
                fetched_df = fetch_steam_reviews_by_app_id(
                    app_id,
                    max_reviews=review_count,
                )

            if fetched_df.empty:
                st.warning("No reviews found. Please check the App ID or try CSV upload.")
                st.session_state["steam_result_df"] = None
                st.session_state["steam_app_id"] = None
            else:
                st.success(
                    f"Fetched {len(fetched_df):,} reviews from Steam App ID {app_id}."
                )

                with st.spinner("Analyzing fetched reviews..."):
                    result_df = analyze_texts(
                        fetched_df["review_text"].tolist(),
                        sentiment_pipe,
                        issue_pipe,
                    )

                meta_cols = [c for c in fetched_df.columns if c != "review_text"]

                if meta_cols:
                    result_df = pd.concat(
                        [result_df, fetched_df[meta_cols].reset_index(drop=True)],
                        axis=1,
                    )

                result_df = sort_by_priority(result_df)

                st.session_state["steam_result_df"] = result_df
                st.session_state["steam_app_id"] = app_id

        except Exception as exc:
            st.error(
                "Failed to fetch or analyze Steam reviews. Please check the App ID, "
                "try a smaller number of reviews, or use CSV upload."
            )
            st.exception(exc)

    if st.session_state["steam_result_df"] is not None:
        st.divider()
        st.caption(
            f"Showing stored analysis results for Steam App ID "
            f"{st.session_state['steam_app_id']}. Changing filters below will not fetch reviews again."
        )
        render_analysis_results(
            st.session_state["steam_result_df"],
            key_prefix="steam",
        )


with tab_csv:
    st.header("Batch CSV Dashboard")

    st.write(
        "Upload a CSV file with one required column named `review_text`. "
        "Each row should contain one review."
    )

    st.markdown(
        """
**Required CSV format**

| review_text |
|---|
| The game keeps crashing after the latest update. |
| I cannot log in because phone verification does not work. |
| Peak game, very fun with friends. |

Optional columns such as `app_name`, `review_score`, or `review_votes` can be included, but only `review_text` is required.
        """
    )

    sample_csv = pd.DataFrame(
        {
            "review_text": [
                "The game keeps crashing after the latest update.",
                "I cannot log in because phone verification does not work.",
                "Peak game, very fun with friends.",
            ]
        }
    ).to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download sample CSV template",
        data=sample_csv,
        file_name="sample_review_upload.csv",
        mime="text/csv",
        key="download_sample_csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    max_rows = st.slider(
        "Maximum rows to analyze",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            if "review_text" not in input_df.columns:
                st.error("The uploaded CSV must contain a column named `review_text`.")
            else:
                input_df = input_df.dropna(subset=["review_text"]).head(max_rows)

                st.write(f"Loaded {len(input_df):,} reviews for analysis.")

                analyze_csv_clicked = st.button(
                    "Analyze Uploaded Reviews",
                    type="primary",
                )

                if analyze_csv_clicked:
                    with st.spinner("Running sentiment and issue category models..."):
                        result_df = analyze_texts(
                            input_df["review_text"].tolist(),
                            sentiment_pipe,
                            issue_pipe,
                        )

                    result_df = sort_by_priority(result_df)

                    st.session_state["csv_result_df"] = result_df

        except Exception as exc:
            st.error("Failed to read or analyze the uploaded CSV.")
            st.exception(exc)

    if st.session_state["csv_result_df"] is not None:
        st.divider()
        st.caption(
            "Showing stored CSV analysis results. Changing filters below will not re-run the model."
        )
        render_analysis_results(
            st.session_state["csv_result_df"],
            key_prefix="csv",
        )


with tab_about:
    st.header("About the Model")

    st.markdown(
        """
### Project purpose
This app helps game developers analyze Steam player reviews by turning unstructured review text into structured product feedback.

### Final model pipeline
The final system uses two fine-tuned Hugging Face DistilBERT models:

1. **Sentiment Classification**  
   Model: `ShirohaNaruse/CustomModel_steam_sentiment`  
   It predicts whether a review is positive/recommended or negative/not recommended.

2. **Issue Category Classification**  
   Model: `ShirohaNaruse/CustomModel_steam_issue`  
   It predicts the developer-oriented issue category:
   Bug / Crash, Account / Access, Multiplayer / Server, Performance, Gameplay, Content, Price / Value, or Emotional Expression.

### Latest experimental results
The final experiment used a 100k-style working sample after duplicate removal.  
The sentiment model and issue category model were both fine-tuned from `distilbert-base-uncased`.

| Pipeline | Final Model | Training Samples | Testing Samples |
|---|---:|---:|---:|
| Sentiment Classification | Fine-tuned DistilBERT | 72,744 | 9,095 |
| Issue Category Classification | Fine-tuned DistilBERT | Updated issue training set | Natural held-out issue test set |

### Important limitation
The issue category labels were generated through keyword-based weak labeling and training-set balancing. Therefore, the issue model should be interpreted as a developer-oriented triage tool rather than a perfect human-labeled classifier.

### App functions
The app supports Steam App ID fetching, CSV upload, review dashboard, issue-priority filtering, detailed review inspection, and downloadable filtered results. The app stores analysis results, so changing filters does not fetch reviews again.
        """
    )
