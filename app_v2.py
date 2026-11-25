import re
import uuid
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="Balagh AI",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f3f7f5;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    color: #17634c;
}
.stButton > button {
    background: #17634c;
    color: white;
    border-radius: 10px;
    width: 100%;
    min-height: 45px;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce8e2;
    border-radius: 14px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

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

def clean_text(text):

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

for category, samples in DATA.items():
    for sample in samples:
        texts.append(clean_text(sample))
        labels.append(category)


def database():

    conn = sqlite3.connect(
        "balagh_v2.db",
        timeout=30
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            text TEXT,
            category TEXT,
            issue TEXT,
            location TEXT,
            confidence REAL,
            status TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    return conn


def get_reports():

    with database() as conn:

        return pd.read_sql_query(
            "SELECT * FROM reports "
            "ORDER BY created_at DESC",
            conn
        )


def save_report(result):

    with database() as conn:

        conn.execute("""
            INSERT INTO reports
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["id"],
            result["text"],
            result["category"],
            result["issue"],
            result["location"],
            result["confidence"],
            "جديد",
            datetime.now().isoformat(
                timespec="seconds"
            )
        ))

        conn.commit()


@st.cache_resource
def load_models():

    encoder = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    classifier.fit(embeddings, labels)

    return encoder, classifier, embeddings


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/"
        "mmarco-mMiniLMv2-L12-H384-v1"
    )


@st.cache_resource
def load_transformer():

    name = (
        "CAMeL-Lab/"
        "bert-base-arabic-camelbert-mix"
    )

    tokenizer = AutoTokenizer.from_pretrained(name)

    model = AutoModel.from_pretrained(
        name,
        attn_implementation="eager"
    )

    model.eval()

    return tokenizer, model


def extract_entities(text):

    location = re.search(
        r"(?:حي|شارع|طريق|تقاطع)\s+[\w]+",
        text
    )

    issue = re.split(
        r"\s+(?:في|عند|قرب|منذ|للتواصل)\s+",
        text,
        maxsplit=1
    )[0]

    return (
        issue,
        location.group(0)
        if location else "غير محدد"
    )


def analyze(text):

    safe = clean_text(text)

    encoder, classifier, embeddings = load_models()

    query = encoder.encode(
        [safe],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    category = classifier.predict(query)[0]

    confidence = float(
        classifier.predict_proba(query).max()
    )

    issue, location = extract_entities(safe)

    saved = get_reports()

    candidates = texts + saved["text"].tolist()

    candidate_labels = (
        labels + saved["category"].tolist()
    )

    candidate_ids = [
        "DEMO-" + str(i + 1)
        for i in range(len(texts))
    ] + saved["id"].tolist()

    vectors = embeddings

    if not saved.empty:

        saved_vectors = encoder.encode(
            saved["text"].tolist(),
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        vectors = np.vstack([
            embeddings,
            saved_vectors
        ])

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    scores, ids = index.search(query, 5)

    similar = pd.DataFrame({
        "رقم البلاغ": [
            candidate_ids[i] for i in ids[0]
        ],
        "البلاغ": [
            candidates[i] for i in ids[0]
        ],
        "التصنيف": [
            candidate_labels[i] for i in ids[0]
        ],
        "التشابه": scores[0]
    })

    warnings = []

    try:

        raise RuntimeError("Cross-Encoder disabled to save memory")
        pairs = [
            [safe, item]
            for item in similar["البلاغ"]
        ]

        similar["درجة إعادة الترتيب"] = (
            reranker.predict(pairs)
        )

        similar = similar.sort_values(
            "درجة إعادة الترتيب",
            ascending=False
        )

    except Exception:

        warnings.append(
            "تعذرت إعادة ترتيب النتائج، "
            "تم استخدام البحث الدلالي فقط."
        )

    token_count = None
    attention = None
    tokens = []

    try:

        raise RuntimeError("CAMeLBERT disabled to save memory")
        inputs = tokenizer(
            safe,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():

            output = model(
                **inputs,
                output_attentions=True
            )

        tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        token_count = len(tokens)

        attention = (
            output.attentions[-1][0, 0]
            .cpu()
            .numpy()
        )

    except Exception:

        warnings.append(
            "تعذر تشغيل تحليل Transformer."
        )

    return {
        "id": "BLG-" + uuid.uuid4().hex[:8].upper(),
        "text": safe,
        "category": category,
        "confidence": confidence,
        "issue": issue,
        "location": location,
        "similar": similar,
        "token_count": token_count,
        "tokens": tokens,
        "attention": attention,
        "warnings": warnings
    }


st.title("🏙️ Balagh AI | بلاغ")

st.caption(
    "منصة ذكية لإدارة وتحليل البلاغات البلدية"
)

st.sidebar.title("Balagh AI")

page = st.sidebar.radio(
    "القائمة الرئيسية",
    [
        "الرئيسية",
        "تقديم بلاغ",
        "إدارة البلاغات",
        "تقييم النظام"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "مشروع تعليمي للذكاء الاصطناعي "
    "ومعالجة اللغة الطبيعية"
)

reports = get_reports()


if page == "الرئيسية":

    st.header("لوحة التحكم")

    st.write(
        "مرحبًا بك في منصة بلاغ، "
        "لتحليل ومتابعة البلاغات البلدية."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "إجمالي البلاغات",
        len(reports)
    )

    c2.metric(
        "بلاغات جديدة",
        int((reports["status"] == "جديد").sum())
    )

    c3.metric(
        "قيد المعالجة",
        int(
            (reports["status"] == "قيد المعالجة").sum()
        )
    )

    c4.metric(
        "تمت المعالجة",
        int(
            (reports["status"] == "تمت المعالجة").sum()
        )
    )

    if reports.empty:

        st.info(
            "لا توجد بلاغات مسجلة حتى الآن. "
            "يمكنك تقديم بلاغ من القائمة الجانبية."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("البلاغات حسب التصنيف")

            st.bar_chart(
                reports["category"].value_counts()
            )

        with c2:

            st.subheader("حالات البلاغات")

            st.bar_chart(
                reports["status"].value_counts()
            )

        st.subheader("أحدث البلاغات")

        st.dataframe(
            reports[
                [
                    "id",
                    "category",
                    "location",
                    "status",
                    "created_at"
                ]
            ].head(10),
            hide_index=True,
            use_container_width=True
        )


elif page == "تقديم بلاغ":

    st.header("تقديم بلاغ جديد")

    st.write(
        "اكتب تفاصيل المشكلة وسيقوم النظام "
        "بتحليل البلاغ وتصنيفه تلقائيًا."
    )

    st.caption(
        "يرجى عدم إدخال بيانات شخصية حقيقية."
    )

    with st.form("report_form"):

        text = st.text_area(
            "تفاصيل البلاغ",
            placeholder=(
                "مثال: تسرب مياه في حي السعادة "
                "منذ الصباح"
            ),
            height=170,
            max_chars=2000
        )

        submitted = st.form_submit_button(
            "إرسال البلاغ"
        )

    if submitted:

        if not text.strip():

            st.warning("يرجى كتابة تفاصيل البلاغ.")

        else:

            try:

                with st.spinner(
                    "جاري تحليل البلاغ وتسجيله..."
                ):

                    result = analyze(text)

                    save_report(result)

                    st.session_state[
                        "last_result"
                    ] = result

            except Exception as error:

                st.error(
                    "تعذر تسجيل البلاغ. "
                    "يرجى المحاولة مرة أخرى."
                )

                st.caption(str(error))

    if "last_result" in st.session_state:

        result = st.session_state["last_result"]

        st.success("تم تسجيل البلاغ بنجاح")

        st.subheader("رقم المتابعة")

        st.code(result["id"])

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "التصنيف",
            result["category"]
        )

        c2.metric(
            "الموقع",
            result["location"]
        )

        c3.metric(
            "الحالة",
            "جديد"
        )

        st.write(
            "**المشكلة:**",
            result["issue"]
        )

        st.write(
            "**نص البلاغ المسجل:**",
            result["text"]
        )

        st.subheader("البلاغات المشابهة")

        st.dataframe(
            result["similar"],
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "تتضمن النتائج بلاغات تعليمية "
            "وبلاغات مسجلة في التطبيق."
        )

        for warning in result["warnings"]:
            st.warning(warning)


elif page == "إدارة البلاغات":

    st.header("إدارة ومتابعة البلاغات")

    if reports.empty:

        st.info("لا توجد بلاغات مسجلة.")

    else:

        search = st.text_input(
            "البحث برقم البلاغ أو النص"
        )

        status_filter = st.selectbox(
            "تصفية حسب الحالة",
            [
                "الكل",
                "جديد",
                "قيد المعالجة",
                "تمت المعالجة"
            ]
        )

        filtered = reports.copy()

        if search:

            filtered = filtered[
                filtered["id"].str.contains(
                    search,
                    case=False,
                    regex=False
                ) |
                filtered["text"].str.contains(
                    search,
                    case=False,
                    regex=False
                )
            ]

        if status_filter != "الكل":

            filtered = filtered[
                filtered["status"] == status_filter
            ]

        st.dataframe(
            filtered[
                [
                    "id",
                    "text",
                    "category",
                    "location",
                    "status",
                    "created_at"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

        st.subheader("تحديث حالة البلاغ")

        selected = st.selectbox(
            "رقم البلاغ",
            reports["id"].tolist()
        )

        new_status = st.selectbox(
            "الحالة الجديدة",
            [
                "جديد",
                "قيد المعالجة",
                "تمت المعالجة"
            ]
        )

        if st.button("حفظ الحالة"):

            with database() as conn:

                conn.execute(
                    "UPDATE reports SET status = ? "
                    "WHERE id = ?",
                    (new_status, selected)
                )

                conn.commit()

            st.rerun()

        st.download_button(
            "تصدير البلاغات CSV",
            filtered.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="balagh_reports.csv",
            mime="text/csv"
        )


elif page == "تقييم النظام":

    st.header("مؤشرات أداء النظام")

    st.caption(
        "تقييم فعلي باستخدام بيانات تعليمية اصطناعية."
    )

    if st.button("حساب مؤشرات الأداء"):

        try:

            with st.spinner(
                "جاري تقييم النموذج..."
            ):

                encoder, classifier, embeddings = (
                    load_models()
                )

                y = np.array(labels)

                train, test = train_test_split(
                    np.arange(len(y)),
                    test_size=0.3,
                    random_state=42,
                    stratify=y
                )

                evaluation_model = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )

                evaluation_model.fit(
                    embeddings[train],
                    y[train]
                )

                predictions = evaluation_model.predict(
                    embeddings[test]
                )

                metrics = classification_report(
                    y[test],
                    predictions,
                    output_dict=True,
                    zero_division=0
                )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Accuracy",
                f"{metrics['accuracy']:.1%}"
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

            st.subheader("الأداء حسب التصنيف")

            df = pd.DataFrame(metrics).T

            st.bar_chart(
                df.loc[
                    list(DATA.keys()),
                    ["precision", "recall", "f1-score"]
                ]
            )

            st.subheader("Confusion Matrix")

            matrix = confusion_matrix(
                y[test],
                predictions,
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

            st.subheader("نتائج الاختبار")

            st.dataframe(pd.DataFrame({
                "Report": [texts[i] for i in test],
                "Actual": y[test],
                "Predicted": predictions,
                "Correct": y[test] == predictions
            }))

            st.caption(
                "35 training samples | 15 test samples"
            )

        except Exception as error:

            st.error(str(error))

    if "last_result" in st.session_state:

        result = st.session_state["last_result"]

        if result["attention"] is not None:

            st.subheader("Transformer Attention")

            st.metric(
                "Tokens",
                result["token_count"]
            )

            st.dataframe(
                pd.DataFrame(
                    result["attention"],
                    index=result["tokens"],
                    columns=result["tokens"]
                ).style.background_gradient(
                    cmap="Blues"
                ),
                use_container_width=True
            )


st.divider()

st.caption(
    "Balagh AI © | Educational prototype "
    "| ليست منصة بلدية رسمية"
)
