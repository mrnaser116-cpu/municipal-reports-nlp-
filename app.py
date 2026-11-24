import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Municipal Reports",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ بلاغات البلدية")
st.subheader("Municipal Reports - NLP Capstone Project")
st.caption("SDAIA Academy | SDA-AIE-211")

st.sidebar.title("أقسام المشروع")

page = st.sidebar.radio(
    "اختر القسم",
    [
        "الصفحة الرئيسية",
        "Lab 1 - معالجة النصوص",
        "Lab 2 - Attention",
        "Lab 3A - تصنيف البلاغات",
        "Lab 3B - NER",
        "Lab 4 - اللهجات العربية",
        "Lab 5 - البحث الدلالي",
        "Lab 6 - التقييم"
    ]
)

def normalize_report_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def mask_pii(text):
    text = str(text)
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b05\d{8}\b', '[PHONE]', text)
    text = re.sub(r'\b(?:REP|REPORT)-\d+\b', '[REPORT]', text, flags=re.IGNORECASE)
    return text

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text

@st.cache_resource
def load_tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)

@st.cache_resource
def load_search_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

if page == "الصفحة الرئيسية":

    st.header("عن المشروع")
    st.write(
        "مشروع تعليمي لتحليل البلاغات البلدية "
        "باستخدام معالجة اللغة الطبيعية والمحولات."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Labs", 6)
    c2.metric("Reports - Lab 3A", 12)
    c3.metric("Topics", 5)

    st.info(
        "اختر أحد الأقسام من القائمة الجانبية "
        "لتجربة وظائف المشروع."
    )

    st.write("### مجالات البلاغات")
    st.bar_chart(
        pd.DataFrame(
            {"Reports": [3, 3, 2, 2, 2]},
            index=[
                "Water",
                "Roads",
                "Waste",
                "Lighting",
                "Digital Services"
            ]
        )
    )

    st.caption("Educational prototype - Labs 1–6")

elif page == "Lab 1 - معالجة النصوص":

    st.header("Text Preprocessing & Privacy")

    text = st.text_area(
        "أدخل بلاغًا تجريبيًا",
        "    انقطاع     الماء    اليوم   \n"
    )

    st.caption("استخدم بيانات تجريبية فقط، وليس معلومات شخصية حقيقية.")

    if st.button("معالجة البلاغ"):
        clean = normalize_report_text(text)
        masked = mask_pii(clean)

        st.write("Original Text")
        st.code(text, language="text")

        st.write("Clean Text")
        st.code(clean, language="text")

        st.write("Masked Text")
        st.code(masked, language="text")

        st.metric("Word Count", len(masked.split()))

    if st.button("Tokenisation"):
        try:
            with st.spinner("Loading tokenizer..."):
                tokenizer = load_tokenizer(
                    "distilbert-base-multilingual-cased"
                )
                masked = mask_pii(
                    normalize_report_text(text)
                )
                tokens = tokenizer.tokenize(masked)

            st.write(tokens)
            st.metric("Token Count", len(tokens))

            words = len(masked.split())
            if words:
                st.metric(
                    "Fertility",
                    round(len(tokens) / words, 3)
                )

            if tokens:
                st.metric(
                    "UNK Rate",
                    tokens.count(tokenizer.unk_token) / len(tokens)
                )
        except Exception as e:
            st.error(str(e))

    st.caption("Regex masking is limited to the patterns in Lab 1.")

elif page == "Lab 2 - Attention":

    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    st.header("Scaled Dot-Product Attention")

    texts = [
        "انقطاع المياه",
        "تعطل إشارة المرور",
        "تراكم النفايات بالشارع",
        "تلف إنارة الممر",
        "تأخر صيانة الرصيف"
    ]

    text = st.selectbox("اختر البلاغ", texts)

    if st.button("عرض Attention Map"):

        torch.manual_seed(42)

        words = text.split()

        q = torch.randn(1, 1, len(words), 8)
        k = torch.randn(1, 1, len(words), 8)
        v = torch.randn(1, 1, len(words), 8)

        scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v

        expected = F.scaled_dot_product_attention(q, k, v)

        torch.testing.assert_close(output, expected)

        st.success("Attention matches PyTorch")

        fig, ax = plt.subplots()
        image = ax.imshow(
            weights[0, 0].detach().numpy(),
            cmap="Blues"
        )
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        fig.colorbar(image)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            pd.DataFrame({
                "Word": words,
                "Position": range(len(words))
            })
        )

    st.info("Synthetic Q/K/V educational experiment.")

elif page == "Lab 3A - تصنيف البلاغات":

    st.header("Topic Classification")

    data = {
        "citizen_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "text": [
            "انقطاع المياه عن الحي منذ الصباح الباكر",
            "تسرب مياه تحت الرصيف يحتاج صيانة عاجلة",
            "تعطل الإشارة المرورية عند التقاطع الرئيسي",
            "وجود حفرة عميقة في منتصف الطريق",
            "تراكم النفايات قرب الحاويات في الشارع العام",
            "تأخر رفع المخلفات المنزلية في الحي",
            "تلف إنارة الممر والمشاة ليلاً",
            "أعمدة الإنارة مطفأة في الشارع الفرعي",
            "لا أستطيع تقديم البلاغ عبر الخدمة الإلكترونية",
            "التطبيق البلدي يغلق عند إرفاق الصور",
            "تضرر الأسفلت بعد هطول الأمطار",
            "مشكلة في ضغط شبكة المياه بالمنطقة"
        ],
        "topic": [
            "water", "water",
            "roads", "roads",
            "waste", "waste",
            "lighting", "lighting",
            "digital_services", "digital_services",
            "roads", "water"
        ]
    }

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)
    st.bar_chart(df["topic"].value_counts())

    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=101
    )

    train_idx, test_idx = next(
        gss.split(df, groups=df["citizen_id"])
    )

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    shared = set(train_df["citizen_id"]) & set(
        test_df["citizen_id"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", len(train_df))
    c2.metric("Test", len(test_df))
    c3.metric("Shared Citizen IDs", len(shared))

    st.write("Model: CAMeLBERT-mix")
    st.write("Epochs: 5")
    st.write("Batch Size: 4")
    st.write("Learning Rate: 3e-5")

    st.warning(
        "Live classification requires the saved "
        "fine-tuned model. Dataset labels are not predictions."
    )

elif page == "Lab 3B - NER":

    st.header("Named Entity Recognition")

    samples = [
        {
            "words": ["تسرب", "مياه", "في", "حي", "السعادة"],
            "labels": ["B-ISSUE", "I-ISSUE", "O", "B-LOC", "I-LOC"]
        },
        {
            "words": ["تعطل", "إشارة", "المرور", "عند", "تقاطع", "الملك", "فهد"],
            "labels": ["B-ISSUE", "I-ISSUE", "I-ISSUE", "O", "B-LOC", "I-LOC", "I-LOC"]
        },
        {
            "words": ["تراكم", "نفايات", "قرب", "الرصيف", "في", "الفيحاء"],
            "labels": ["B-ISSUE", "I-ISSUE", "O", "B-LOC", "O", "B-LOC"]
        }
    ]

    choice = st.selectbox(
        "اختر المثال",
        range(len(samples)),
        format_func=lambda i: " ".join(samples[i]["words"])
    )

    sample = samples[choice]

    st.dataframe(pd.DataFrame({
        "Word": sample["words"],
        "BIO Label": sample["labels"]
    }))

    if st.button("Token Alignment"):
        try:
            tokenizer = load_tokenizer(
                "CAMeL-Lab/bert-base-arabic-camelbert-mix"
            )

            encoded = tokenizer(
                sample["words"],
                is_split_into_words=True
            )

            tokens = tokenizer.convert_ids_to_tokens(
                encoded["input_ids"]
            )

            st.dataframe(pd.DataFrame({
                "Token": tokens,
                "Word ID": [
                    str(i) for i in encoded.word_ids()
                ]
            }))
        except Exception as e:
            st.error(str(e))

    st.info("Manual BIO labels - not trained NER predictions.")

elif page == "Lab 4 - اللهجات العربية":

    st.header("Arabic NLP & Dialect Adaptation")

    texts = [
        "وش وضع أنقطاع المياة",
        "الموية مقطوعة من أمس في الحي",
        "وش وضع بلاغي عن المخالفة ",
        "فيه نفايات متراكمة بالحاوية"
    ]

    example = st.selectbox("اختر المثال", texts)
    text = st.text_area("النص", example)

    if st.button("Normalize Arabic"):
        st.write("Before:")
        st.code(text, language="text")

        st.write("After:")
        st.code(normalize_arabic(text), language="text")

    st.write("### Clitic Segmentation")
    st.write("بالحاوية")
    st.write("ب + ال + حاوية")

    st.write("### Illustrative Macro-F1")

    models = {
        "CAMeLBERT-mix": 0.92,
        "CAMeLBERT-DA": 0.99
    }

    st.bar_chart(
        pd.DataFrame.from_dict(
            models,
            orient="index",
            columns=["Macro-F1"]
        )
    )

    st.warning("Hard-coded example values, not measured model results.")

elif page == "Lab 5 - البحث الدلالي":

    st.header("Semantic Search")

    texts = [
        "انقطعت الموية عن الحي",
        "مشكلة في فاتورة المياة",
        "يوجد خلل في المياه"
    ]

    query = st.text_input(
        "أدخل البلاغ للبحث",
        "الموية لا تعمل"
    )

    if st.button("بحث عن البلاغات المتشابهة"):
        if not query.strip():
            st.warning("أدخل نصًا للبحث.")
        else:
            try:
                import faiss

                with st.spinner("Loading MiniLM and FAISS..."):

                    model = load_search_model()
                    embeddings = model.encode(texts)

                    index = faiss.IndexFlatL2(
                        embeddings.shape[1]
                    )

                    index.add(embeddings)

                    query_embedding = model.encode([query])

                    distances, indices = index.search(
                        query_embedding, 2
                    )

                st.dataframe(pd.DataFrame({
                    "Report": [
                        texts[i] for i in indices[0]
                    ],
                    "L2 Distance": distances[0]
                }))

                st.caption("Lower L2 distance means closer embeddings.")

            except Exception as e:
                st.error(str(e))

    if st.button("Cross-Encoder Reranking"):
        if not query.strip():
            st.warning("أدخل نصًا للبحث.")
        else:
            try:
                results = [
                    "مشكلة في فاتورة المياة",
                    "انقطعت المياة عن الحي",
                    "يوجد خلل في المياه"
                ]

                with st.spinner("Loading Cross-Encoder..."):
                    reranker = load_reranker()

                    pairs = [
                        [query, text] for text in results
                    ]

                    scores = reranker.predict(pairs)

                st.dataframe(pd.DataFrame({
                    "Report": results,
                    "Score": [
                        round(float(s), 2) for s in scores
                    ]
                }))

                st.caption(
                    "Scores shown in original Lab 5 order. "
                    "Higher scores indicate greater relevance."
                )

            except Exception as e:
                st.error(str(e))

    st.write("### Original Lab 5 Metric Exercise")

    results = [
        "مشكلة في فاتورة المياة",
        "انقطعت الموية عن الحي",
        "يوجد خلل في المياه"
    ]

    correct_answer = "انقطعت الموية عن الحي"
    rank = results.index(correct_answer) + 3
    recall_at_10 = 1 if rank <= 10 else 0
    mrr_at_10 = 1 / rank

    c1, c2 = st.columns(2)
    c1.metric("Original Recall@10", recall_at_10)
    c2.metric("Original MRR@10", mrr_at_10)

    st.warning(
        "Original code uses index + 3. "
        "The standard rank uses index + 1, "
        "giving reciprocal rank 0.50 as documented "
        "in the presentation."
    )

elif page == "Lab 6 - التقييم":

    st.header("Evaluation & Error Analysis")

    st.info("All labels and predictions in this section are manual examples.")

    reports = [
        {"language": "Arabic", "dialect": "Saudi", "length": "Short", "correct": 1},
        {"language": "Arabic", "dialect": "Saudi", "length": "Long", "correct": 1},
        {"language": "Arabic", "dialect": "MSA", "length": "Short", "correct": 1},
        {"language": "Arabic", "dialect": "Egyptian", "length": "Short", "correct": 0},
        {"language": "English", "dialect": "Standard", "length": "Long", "correct": 0}
    ]

    df = pd.DataFrame(reports)

    st.write("### Sliced Evaluation")
    st.dataframe(df)

    group = st.selectbox(
        "Evaluation Slice",
        ["language", "dialect", "length"]
    )

    st.bar_chart(df.groupby(group)["correct"].mean())

    st.write("### Bootstrap Confidence Interval")

    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])

    st.metric(
        "Example Accuracy",
        f"{accuracy_score(y_true, y_pred):.0%}"
    )

    if st.button("Calculate 95% CI"):

        scores = []

        for _ in range(1000):
            i = np.random.choice(
                len(y_true),
                len(y_true),
                replace=True
            )
            scores.append(
                accuracy_score(y_true[i], y_pred[i])
            )

        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)

        st.write("95% CI:", lower, "-", upper)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(scores, bins=11)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    st.write("### Error Analysis")

    errors = [
        ("الموية مقطوعة عن البيت", "انقطاع المياه", "تراكم النفايات"),
        ("الموية تقطر", "تسرب المياه", "تسرب المياه"),
        ("الزبالة متكدسة عند الحي", "تراكم النفايات", "ازدحام مروري"),
        ("كيف أحجز تذكرة ؟", "خارج النطاق", "تسرب المياه")
    ]

    error_df = pd.DataFrame(
        errors,
        columns=[
            "البلاغ",
            "التصنيف الصحيح",
            "توقع النموذج"
        ]
    )

    error_df["التقييم"] = [
        "صحيح" if actual == predicted else "خطأ"
        for text, actual, predicted in errors
    ]

    st.dataframe(error_df, use_container_width=True)
    st.bar_chart(error_df["التقييم"].value_counts())

    st.write("### Evaluation Report")

    report = {
        "Domain": "Municipal Services (Water, Traffic, Waste)",
        "Sliced Evaluation": "Language, Dialect (Saudi/MSA/Egyptian), Length",
        "Confidence Interval": "95%",
        "Error Analysis": "Completed (Water, Waste, Traffic, Out-of-Scope)",
        "Error Taxonomy": "Completed"
    }

    st.table(
        pd.DataFrame(
            list(report.items()),
            columns=["Item", "Description"]
        )
    )

st.divider()
st.caption("Municipal Reports | SDAIA Academy | Educational Prototype")
