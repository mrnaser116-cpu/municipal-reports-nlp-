import re
import numpy as np
import pandas as pd
import streamlit as st
import torch
import faiss

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="Balagh AI",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f7f6;
    color: #183b32;
}
.stButton > button {
    background-color: #176b50;
    color: white;
    border-radius: 10px;
    width: 100%;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce8e2;
}
</style>
""", unsafe_allow_html=True)

# Educational synthetic dataset

DATA = {
    "المياه": [
        "انقطاع المياه عن الحي منذ الصباح",
        "تسرب مياه تحت الرصيف",
        "الموية مقطوعة من أمس",
        "ضعف ضغط المياه في المنزل",
        "انفجار أنبوب المياه",
        "مشكلة في شبكة المياه",
        "الماء لا يصل إلى البيت",
        "تسرب الموية في الشارع",
        "Water supply is not working",
        "Water leaking from a pipe"
    ],
    "الطرق": [
        "تعطل الإشارة المرورية",
        "وجود حفرة عميقة في الطريق",
        "تضرر الأسفلت بعد الأمطار",
        "الرصيف مكسور",
        "الشارع يحتاج إعادة سفلتة",
        "تأخر صيانة الرصيف",
        "الطريق مليان حفر",
        "تلف الطريق عند التقاطع",
        "Traffic signal is broken",
        "Large pothole in the road"
    ],
    "النفايات": [
        "تراكم النفايات قرب الحاويات",
        "تأخر رفع المخلفات المنزلية",
        "الحاويات ممتلئة",
        "وجود مخلفات بناء",
        "الزبالة متراكمة في الشارع",
        "انتشار النفايات في الحي",
        "لم يتم جمع المخلفات",
        "تحتاج الحاوية إلى تنظيف",
        "Garbage is accumulating",
        "Waste containers are full"
    ],
    "الإنارة": [
        "تلف إنارة الممر",
        "أعمدة الإنارة مطفأة",
        "ضعف إنارة الشارع",
        "اللمبات ما تشتغل",
        "الشارع مظلم في الليل",
        "تعطل مصابيح الطريق",
        "صيانة أعمدة الإنارة",
        "انطفاء إنارة الحديقة",
        "Street lights are broken",
        "The street is dark"
    ],
    "الخدمات الرقمية": [
        "لا أستطيع تقديم البلاغ إلكترونيا",
        "التطبيق يغلق عند إرفاق الصور",
        "خطأ في البوابة الإلكترونية",
        "مشكلة في تسجيل الدخول",
        "الموقع ما يفتح",
        "تعطل الخدمة الرقمية",
        "لا يمكن رفع المرفقات",
        "مشكلة في تطبيق البلدية",
        "Application is not working",
        "Unable to submit online report"
    ]
}

# Lab 1 and Lab 4: preprocessing

def prepare_text(text):
    text = text.translate(str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    ))

    text = re.sub(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "[EMAIL]", text
    )

    text = re.sub(
        r"(?:\+?966|00966|0)5(?:[\s-]*\d){8}\b",
        "[PHONE]", text
    )

    text = re.sub(
        r"\b(?:REP|REPORT|BLG|REF)[-\s#]*\d+\b",
        "[REFERENCE]", text, flags=re.I
    )

    text = re.sub(
        r"[\u064B-\u065F\u0670\u0640]",
        "", text
    )

    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي")

    return re.sub(r"\s+", " ", text).strip()


texts = []
labels = []

for category, reports in DATA.items():
    for report in reports:
        texts.append(prepare_text(report))
        labels.append(category)


# Lab 2: Transformer attention

@st.cache_resource
def load_transformer():
    name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

    tokenizer = AutoTokenizer.from_pretrained(name)

    model = AutoModel.from_pretrained(
        name,
        attn_implementation="eager"
    )

    model.eval()

    return tokenizer, model


# Lab 3A and Lab 5: classification and retrieval

@st.cache_resource
def load_pipeline():

    encoder = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    indices = np.arange(len(texts))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    y = np.array(labels)

    evaluation_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    evaluation_model.fit(
        embeddings[train_idx],
        y[train_idx]
    )

    predictions = evaluation_model.predict(
        embeddings[test_idx]
    )

    metrics = classification_report(
        y[test_idx],
        predictions,
        output_dict=True,
        zero_division=0
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    classifier.fit(embeddings, y)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return (
        encoder,
        classifier,
        index,
        metrics,
        y[test_idx],
        predictions
    )


@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/"
        "mmarco-mMiniLMv2-L12-H384-v1"
    )


# Lab 3B: rule-based entity extraction

def extract_entities(text, category):

    location = re.search(
        r"(?:حي|شارع|طريق|تقاطع)\s+"
        r"[\w]+(?:\s+[\w]+)?",
        text
    )

    patterns = {
        "المياه": (
            r"(?:انقطاع|تسرب|ضعف ضغط|انفجار)"
            r"\s+[\w]+"
        ),
        "الطرق": (
            r"(?:تعطل|حفرة|تضرر|تلف|صيانة)"
            r"(?:\s+[\w]+){0,2}"
        ),
        "النفايات": (
            r"(?:تراكم|انتشار|مخلفات|الحاويات)"
            r"(?:\s+[\w]+){0,2}"
        ),
        "الإنارة": (
            r"(?:تلف|ضعف|انطفاء|تعطل)"
            r"(?:\s+[\w]+){0,2}"
        ),
        "الخدمات الرقمية": (
            r"(?:خطا|تعطل|مشكلة|التطبيق)"
            r"(?:\s+[\w]+){0,2}"
        )
    }

    issue = re.search(
        patterns[category],
        text
    )

    return {
        "ISSUE": (
            issue.group(0)
            if issue else "غير مستخرج"
        ),
        "LOC": (
            location.group(0)
            if location else "غير محدد"
        )
    }


# User interface

st.title("🏙️ Balagh AI | بلاغات البلدية")

st.markdown(
    "### نظام موحد لتحليل البلاغات البلدية"
)

st.caption(
    "Educational NLP Capstone | Labs 1–6"
)

st.info(
    "نموذج تعليمي مستقل يستخدم بيانات اصطناعية، "
    "وليس منصة رسمية لتقديم البلاغات."
)

with st.sidebar:
    st.header("Balagh AI")

    st.write("معالجة اللغة الطبيعية")
    st.write("تصنيف البلاغات")
    st.write("استخراج الكيانات")
    st.write("البحث الدلالي")
    st.write("تقييم النماذج")

    st.divider()

    st.metric("عدد التصنيفات", len(DATA))
    st.metric("البلاغات التعليمية", len(texts))


analysis_tab, evaluation_tab, about_tab = st.tabs([
    "تحليل البلاغ",
    "تقييم الأداء",
    "عن المشروع"
])


with analysis_tab:

    st.subheader("إدخال بلاغ جديد")

    report = st.text_area(
        "نص البلاغ",
        placeholder="اكتب البلاغ هنا...",
        height=140,
        max_chars=2000
    )

    show_attention = st.checkbox(
        "عرض Transformer Attention",
        value=False
    )

    use_reranker = st.checkbox(
        "إعادة ترتيب النتائج باستخدام Cross-Encoder",
        value=True
    )

    if st.button("تحليل البلاغ", type="primary"):

        if not report.strip():

            st.warning("الرجاء كتابة نص البلاغ.")

        else:

            clean = prepare_text(report)

            st.subheader("1. معالجة النص والخصوصية")

            st.code(clean, language=None)

            st.metric(
                "عدد الكلمات",
                len(clean.split())
            )

            try:

                with st.spinner(
                    "جاري تحميل النماذج وتحليل البلاغ..."
                ):

                    (
                        encoder,
                        classifier,
                        index,
                        metrics,
                        actual,
                        predicted
                    ) = load_pipeline()

                    query = encoder.encode(
                        [clean],
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    ).astype("float32")

                    category = classifier.predict(
                        query
                    )[0]

                    probability = classifier.predict_proba(
                        query
                    ).max()

                    entities = extract_entities(
                        clean,
                        category
                    )

                    scores, ids = index.search(
                        query, 5
                    )

                st.subheader("2. التصنيف واستخراج الكيانات")

                c1, c2, c3 = st.columns(3)

                c1.metric("التصنيف", category)

                c2.metric(
                    "احتمال النموذج",
                    f"{probability:.1%}"
                )

                c3.metric(
                    "الموقع",
                    entities["LOC"]
                )

                st.write(
                    "المشكلة المستخرجة:",
                    entities["ISSUE"]
                )

                st.caption(
                    "التصنيف باستخدام MiniLM embeddings "
                    "وLogistic Regression. "
                    "استخراج الكيانات بالقواعد النصية."
                )

                probabilities = pd.DataFrame({
                    "Category": classifier.classes_,
                    "Probability": (
                        classifier.predict_proba(query)[0]
                    )
                })

                st.bar_chart(
                    probabilities.set_index("Category")
                )

                st.subheader("3. البحث عن البلاغات المشابهة")

                results = pd.DataFrame({
                    "Report": [
                        texts[i] for i in ids[0]
                    ],
                    "Category": [
                        labels[i] for i in ids[0]
                    ],
                    "Cosine similarity": scores[0]
                })

                if use_reranker:

                    try:

                        with st.spinner(
                            "جاري إعادة ترتيب النتائج..."
                        ):

                            reranker = load_reranker()

                            pairs = [
                                [clean, text]
                                for text in results["Report"]
                            ]

                            results["Cross-Encoder score"] = (
                                reranker.predict(pairs)
                            )

                            results = results.sort_values(
                                "Cross-Encoder score",
                                ascending=False
                            )

                    except Exception as error:

                        st.warning(
                            "تعذر تشغيل Cross-Encoder. "
                            "نعرض نتائج FAISS فقط."
                        )

                        st.caption(str(error))

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "درجات التشابه ليست احتمالات "
                    "مؤكدة لتكرار البلاغ."
                )

                if show_attention:

                    st.subheader(
                        "4. Transformer Tokenization "
                        "and Attention"
                    )

                    try:

                        with st.spinner(
                            "جاري تحميل CAMeLBERT..."
                        ):

                            tokenizer, transformer = (
                                load_transformer()
                            )

                            inputs = tokenizer(
                                clean,
                                return_tensors="pt",
                                truncation=True,
                                max_length=128
                            )

                            with torch.no_grad():

                                output = transformer(
                                    **inputs,
                                    output_attentions=True
                                )

                            tokens = (
                                tokenizer.convert_ids_to_tokens(
                                    inputs["input_ids"][0]
                                )
                            )

                            attention = (
                                output.attentions[-1][0, 0]
                                .cpu()
                                .numpy()
                            )

                        st.write("Tokens:", tokens)

                        st.metric(
                            "Token count",
                            len(tokens)
                        )

                        st.dataframe(
                            pd.DataFrame(
                                attention,
                                index=tokens,
                                columns=tokens
                            ).style.background_gradient(
                                cmap="Blues"
                            ),
                            use_container_width=True
                        )

                        st.caption(
                            "Actual CAMeLBERT attention: "
                            "last layer, first head."
                        )

                    except Exception as error:

                        st.warning(
                            "تعذر تشغيل CAMeLBERT."
                        )

                        st.caption(str(error))

            except Exception as error:

                st.error(
                    "تعذر تشغيل نموذج التحليل."
                )

                st.caption(str(error))


# Lab 6: evaluation

with evaluation_tab:

    st.header("Model Evaluation")

    st.write(
        "تقييم فعلي على مجموعة اختبار منفصلة "
        "من البيانات التعليمية."
    )

    if st.button("تشغيل التقييم"):

        try:

            with st.spinner("جاري حساب النتائج..."):

                (
                    encoder,
                    classifier,
                    index,
                    metrics,
                    actual,
                    predicted
                ) = load_pipeline()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Accuracy",
                f"{accuracy_score(actual, predicted):.1%}"
            )

            c2.metric(
                "Precision",
                f"{metrics['macro avg']['precision']:.1%}"
            )

            c3.metric(
                "Recall",
                f"{metrics['macro avg']['recall']:.1%}"
            )

            c4.metric(
                "Macro-F1",
                f"{metrics['macro avg']['f1-score']:.1%}"
            )

            st.write(
                "Training samples:", 35,
                "| Test samples:", len(actual)
            )

            st.subheader("Classification Report")

            report_df = pd.DataFrame(metrics).T

            st.dataframe(
                report_df,
                use_container_width=True
            )

            st.subheader("Performance by Category")

            st.bar_chart(
                report_df.loc[
                    list(DATA.keys()),
                    ["precision", "recall", "f1-score"]
                ]
            )

            st.subheader("Confusion Matrix")

            matrix = confusion_matrix(
                actual,
                predicted,
                labels=list(DATA.keys())
            )

            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=list(DATA.keys()),
                    columns=list(DATA.keys())
                ),
                use_container_width=True
            )

            st.subheader("Actual vs Predicted")

            st.dataframe(pd.DataFrame({
                "Actual": actual,
                "Predicted": predicted,
                "Correct": actual == predicted
            }))

            st.warning(
                "هذه النتائج تخص بيانات تعليمية صغيرة، "
                "ولا تمثل أداء النظام على بلاغات حقيقية."
            )

        except Exception as error:

            st.error(str(error))


with about_tab:

    st.header("Project Architecture")

    st.table(pd.DataFrame({
        "Lab": [
            "Lab 1",
            "Lab 2",
            "Lab 3A",
            "Lab 3B",
            "Lab 4",
            "Lab 5",
            "Lab 6"
        ],
        "Implementation": [
            "Text cleaning and PII masking",
            "CAMeLBERT tokenizer and attention",
            "MiniLM + Logistic Regression",
            "Rule-based entity extraction",
            "Arabic normalization",
            "FAISS + Cross-Encoder",
            "Held-out classification evaluation"
        ]
    }))

    st.write(
        "Input → Preprocessing → Classification "
        "→ Entity Extraction → Retrieval → Evaluation"
    )

    st.caption(
        "Educational prototype inspired by "
        "Municipal Reports NLP Labs."
    )

st.divider()
st.caption("Balagh AI | Municipal Reports NLP")
