import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from openai import OpenAI

# =========================
# 🔑 Укажи свой OpenAI ключ
# =========================
client = OpenAI(api_key="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")

# =========================
# ⚙️ Обучение модели (один раз)
# =========================
@st.cache_data
def train_model():
    df = pd.read_csv("TOI_2025.10.05_05.42.17.csv")

    data = df[df['tfopwg_disp'].isin(['PC', 'CP', 'FP'])].copy()
    data['target'] = data['tfopwg_disp'].map({'PC': 1, 'CP': 1, 'FP': 0})

    drop_cols = ['toi', 'tid', 'tfopwg_disp', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    X = data.drop(columns=drop_cols + ['target']).select_dtypes(include=['number']).dropna(axis=1, how='all')
    y = data['target']

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_imputed, y)

    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top5_features = feature_importances.nlargest(5).index.tolist()

    X_top5 = X[top5_features]
    X_top5_imputed = imputer.fit_transform(X_top5)

    X_train, X_test, y_train, y_test = train_test_split(
        X_top5_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mlp_top5 = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
    mlp_top5.fit(X_train_scaled, y_train)

    return mlp_top5, imputer, scaler, top5_features

mlp_top5, imputer, scaler, top5_features = train_model()

# =========================
# 🔮 Функция предсказания
# =========================
def predict_exoplanet(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input_imputed = imputer.transform(df_input)
    df_input_scaled = scaler.transform(df_input_imputed)
    pred = mlp_top5.predict(df_input_scaled)[0]
    return "Exoplanet" if pred == 1 else "Not Exoplanet"

# =========================
# 💬 Интерфейс чата Streamlit
# =========================
st.set_page_config(page_title="Exoplanet Predictor Chat", page_icon="🪐")
st.title("🪐 Exoplanet Chat — AI Exoplanet Predictor")

st.markdown("Введите параметры планеты, и AI расскажет, экзопланета это или нет, а также объяснит свой ответ.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Показываем историю чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поля ввода параметров
with st.expander("🔧 Ввести данные планеты"):
    user_input = {}
    for feature in ['pl_eqt', 'pl_insol', 'pl_tranmid', 'st_tmag', 'pl_rade']:
        val = st.text_input(f"{feature}:", placeholder="Введите значение...")
        if val:
            try:
                user_input[feature] = float(val)
            except ValueError:
                st.warning(f"{feature} должно быть числом!")

# Чат
if prompt := st.chat_input("Введите команду или вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Если пользователь ввёл данные планеты — делаем предсказание
    if len(user_input) == 5:
        prediction = predict_exoplanet(user_input)

        # Объяснение от GPT
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Ты объясняешь результаты предсказания экзопланет простыми словами."},
                {"role": "user", "content": f"На основе данных {user_input} модель предсказала: {prediction}. Объясни результат понятно для пользователя."}
            ]
        )
        answer = response.choices[0].message.content

        final_response = f"**Результат:** {prediction}\n\n**Пояснение:** {answer}"
    else:
        # Если нет данных, просто говорим с GPT
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Ты помощник, который говорит об экзопланетах и астрономии простым языком."},
                *st.session_state.messages
            ]
        )
        final_response = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.markdown(final_response)
