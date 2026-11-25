import re
import uuid
import sqlite3
from datetime import datetime
from contextlib import closing

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Balagh AI",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.stButton > button {
    background: #17634c;
    color: white;
    border-radius: 10px;
    width: 100%;
}
[data-testid="stMetric"] {
    border: 1px solid #a9c9bc;
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
        "Water leaking from a pipe",
        "المواسير تسرب مياه",
        "انقطاع الماء اليوم",
        "فيه موية تتسرب من الانبوب",
        "No water in the neighborhood",
        "Low water pressure"
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
        "Large pothole in the road",
        "حفرة خطيرة في الشارع",
        "تشققات في الاسفلت",
        "الاشارة ما تشتغل",
        "Damaged road surface",
        "Broken sidewalk"
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
        "Waste containers are full",
        "تكدس القمامة بجانب المنازل",
        "الزبالة ما انشالت",
        "نفايات منتشرة قرب الحديقة",
        "Garbage collection delayed",
        "Waste on the sidewalk"
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
        "The street is dark",
        "لمبات الشارع طافية",
        "الممر بدون إنارة",
        "عمود النور خربان",
        "Broken street lamp",
        "Public lighting failure"
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
        "Unable to submit online report",
        "التطبيق يعلق عند ارسال البلاغ",
        "ما اقدر ادخل حسابي",
        "الخدمة الالكترونية متوقفة",
        "Login error on the website",
        "Cannot upload attachments"
    ]
}

def clean_text(text):
    text = text.translate(str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    ))

    text = re.sub(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "[EMAIL]",
        text
    )

    text = re.sub(
        r"(?<!\d)(?:\+966|00966|966|0)5"
        r"(?:[\s-]*\d){8}(?!\d)",
        "[PHONE]",
        text
    )

    text = re.sub(
        r"\b(?:REP|REPORT|BLG|REF)[-\s#]*\d+\b",
        "[REFERENCE]",
        text,
        flags=re.I
    )

    text = re.sub(
        r"[\u064B-\u065F\u0670\u0640]",
        "",
        text
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


def create_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                max_features=15000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=5
            )
        )
    ])


@st.cache_resource
def train_model():

    x_train, x_test, y_train, y_test = (
        train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels
        )
    )

    evaluation_model = create_model()

    evaluation_model.fit(
        x_train,
        y_train
    )

    predictions = evaluation_model.predict(
        x_test
    )

    metrics = classification_report(
        y_test,
        predictions,
        labels=list(DATA.keys()),
        output_dict=True,
        zero_division=0
    )

    model = create_model()

    model.fit(texts, labels)

    return {
        "model": model,
        "metrics": metrics,
        "actual": y_test,
        "predicted": predictions,
        "test_texts": x_test,
        "train_count": len(x_train),
        "test_count": len(x_test)
    }


def database():

    conn = sqlite3.connect(
        "balagh_lite.db",
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

    with closing(database()) as conn:

        return pd.read_sql_query(
            "SELECT * FROM reports "
            "ORDER BY created_at DESC",
            conn
        )


def save_report(result):

    with closing(database()) as conn:

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


def find_similar(text, model):

    saved = get_reports()

    candidates = (
        texts + saved["text"].tolist()
    )

    categories = (
        labels + saved["category"].tolist()
    )

    ids = [
        "DEMO-" + str(i + 1)
        for i in range(len(texts))
    ] + saved["id"].tolist()

    vectorizer = model.named_steps["tfidf"]

    query_vector = vectorizer.transform(
        [text]
    )

    candidate_vectors = vectorizer.transform(
        candidates
    )

    scores = cosine_similarity(
        query_vector,
        candidate_vectors
    )[0]

    top = np.argsort(scores)[::-1][:5]

    results = []

    for i in top:

        if scores[i] <= 0:
            continue

        results.append({
            "رقم البلاغ": ids[i],
            "البلاغ": candidates[i],
            "التصنيف": categories[i],
            "التشابه": round(
                float(scores[i]) * 100, 1
            )
        })

    return pd.DataFrame(
        results,
        columns=[
            "رقم البلاغ",
            "البلاغ",
            "التصنيف",
            "التشابه"
        ]
    )


def analyze(text):

    safe = clean_text(text)

    trained = train_model()

    model = trained["model"]

    category = model.predict([safe])[0]

    confidence = float(
        model.predict_proba([safe]).max()
    )

    issue, location = extract_entities(safe)

    similar = find_similar(
        safe,
        model
    )

    return {
        "id": (
            "BLG-" +
            uuid.uuid4().hex[:8].upper()
        ),
        "text": safe,
        "category": category,
        "issue": issue,
        "location": location,
        "confidence": confidence,
        "similar": similar
    }


st.title("🏙️ Balagh AI | بلاغ")

st.caption(
    "منصة ذكية لتحليل وإدارة البلاغات البلدية"
)

st.sidebar.title("القائمة الرئيسية")

page = st.sidebar.radio(
    "انتقل إلى",
    [
        "الرئيسية",
        "تقديم بلاغ",
        "إدارة البلاغات",
        "تقييم النظام",
        "عن المشروع"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "نموذج تعليمي مستقل"
)

reports = get_reports()


if page == "الرئيسية":

    st.header("لوحة التحكم")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "إجمالي البلاغات",
        len(reports)
    )

    c2.metric(
        "جديد",
        int(
            (reports["status"] == "جديد").sum()
        )
    )

    c3.metric(
        "قيد المعالجة",
        int(
            (
                reports["status"] ==
                "قيد المعالجة"
            ).sum()
        )
    )

    c4.metric(
        "تمت المعالجة",
        int(
            (
                reports["status"] ==
                "تمت المعالجة"
            ).sum()
        )
    )

    if reports.empty:

        st.info(
            "لا توجد بلاغات مسجلة حاليًا. "
            "يمكنك تقديم بلاغ جديد."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("تصنيفات البلاغات")

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
            reports.head(10),
            hide_index=True,
            use_container_width=True
        )


elif page == "تقديم بلاغ":

    st.header("تقديم بلاغ جديد")

    st.caption(
        "يرجى عدم إدخال معلومات شخصية حقيقية."
    )

    with st.form(
        "report_form",
        clear_on_submit=True
    ):

        text = st.text_area(
            "تفاصيل البلاغ",
            placeholder=(
                "مثال: تسرب مياه في حي السعادة"
            ),
            height=160,
            max_chars=2000
        )

        submitted = st.form_submit_button(
            "إرسال البلاغ"
        )

    if submitted:

        st.session_state.pop(
            "last_result",
            None
        )

        if not text.strip():

            st.warning(
                "يرجى كتابة تفاصيل البلاغ."
            )

        else:

            try:

                with st.spinner(
                    "جاري تحليل وتسجيل البلاغ..."
                ):

                    result = analyze(text)

                    save_report(result)

                    st.session_state[
                        "last_result"
                    ] = result

            except Exception as error:

                st.error(
                    "تعذر تسجيل البلاغ."
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
            "احتمال النموذج",
            f"{result['confidence']:.1%}"
        )

        st.write(
            "**المشكلة:**",
            result["issue"]
        )

        st.write(
            "**البلاغ بعد حماية الخصوصية:**",
            result["text"]
        )

        if result["confidence"] < 0.60:

            st.warning(
                "التصنيف يحتاج مراجعة بشرية."
            )

        st.subheader("البلاغات المشابهة")

        if result["similar"].empty:

            st.info(
                "لم يتم العثور على نتائج مشابهة."
            )

        else:

            st.dataframe(
                result["similar"],
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "نتائج التشابه تعتمد على النص، "
            "ولا تؤكد تكرار البلاغ."
        )


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
                filtered["status"] ==
                status_filter
            ]

        st.dataframe(
            filtered,
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

            with closing(database()) as conn:

                conn.execute(
                    "UPDATE reports "
                    "SET status = ? WHERE id = ?",
                    (new_status, selected)
                )

                conn.commit()

            st.rerun()

        st.download_button(
            "تصدير البلاغات",
            filtered.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="balagh_reports.csv",
            mime="text/csv"
        )


elif page == "تقييم النظام":

    st.header("تقييم أداء النموذج")

    trained = train_model()

    metrics = trained["metrics"]

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

    st.write(
        "Training samples:",
        trained["train_count"],
        "| Test samples:",
        trained["test_count"]
    )

    st.subheader("الأداء حسب التصنيف")

    df = pd.DataFrame(metrics).T

    st.bar_chart(
        df.loc[
            list(DATA.keys()),
            ["precision", "recall", "f1-score"]
        ]
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        trained["actual"],
        trained["predicted"],
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

    st.subheader("نتائج الاختبار الفعلية")

    st.dataframe(
        pd.DataFrame({
            "Report": trained["test_texts"],
            "Actual": trained["actual"],
            "Predicted": trained["predicted"],
            "Correct": (
                np.array(trained["actual"]) ==
                trained["predicted"]
            )
        }),
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "تم التقييم على بيانات تعليمية اصطناعية "
        "وليست بلاغات بلدية حقيقية."
    )


elif page == "عن المشروع":

    st.header("Balagh AI")

    st.write(
        "منصة تعليمية لتحليل البلاغات البلدية "
        "باستخدام معالجة اللغة الطبيعية "
        "والتعلم الآلي."
    )

    st.table(pd.DataFrame({
        "Component": [
            "Preprocessing",
            "Privacy",
            "Feature Extraction",
            "Classification",
            "Entity Extraction",
            "Retrieval",
            "Evaluation",
            "Database"
        ],
        "Technology": [
            "Arabic Normalization",
            "Regex PII Masking",
            "TF-IDF",
            "Logistic Regression",
            "Rule-based Extraction",
            "Cosine Similarity",
            "Held-out Test Set",
            "SQLite"
        ]
    }))

    st.metric(
        "البلاغات التعليمية",
        len(texts)
    )

    st.metric(
        "عدد التصنيفات",
        len(DATA)
    )

    st.info(
        "هذه النسخة محسنة لتعمل باستهلاك "
        "منخفض للذاكرة."
    )


st.divider()

st.caption(
    "Balagh AI | Educational NLP Project "
    "| ليست منصة بلدية رسمية"
)
