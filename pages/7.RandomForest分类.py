import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 初始化会话状态
if 'preprocess_params' not in st.session_state:
    st.session_state.preprocess_params = {
        'scale_numeric': True,
        'encode_categorical': True,
        'missing_strategy': '删除含缺失行'
    }

st.set_page_config(page_title="随机森林分类分析", layout="wide")
st.title("📊 随机森林分类分析工具")


# ==================== 数据预处理模块 ====================
def data_preprocessing(df):
    with st.expander("🔧 数据预处理设置", expanded=True):
        col1, col2 = st.columns(2)

        # 缺失值处理
        with col1:
            st.markdown("### 🚫 缺失值处理")
            st.session_state.preprocess_params['missing_strategy'] = st.selectbox(
                "处理方式",
                ["删除含缺失行", "数值列填充均值", "分类列填充众数"],
                key="missing_strategy"
            )

        # 特征工程
        with col2:
            st.markdown("### 🛠 特征工程")
            st.session_state.preprocess_params['scale_numeric'] = st.checkbox(
                "标准化数值特征",
                value=True,
                key="scale_numeric"
            )
            st.session_state.preprocess_params['encode_categorical'] = st.checkbox(
                "编码分类特征",
                value=True,
                key="encode_categorical"
            )

    # 应用缺失值处理策略
    if st.session_state.preprocess_params['missing_strategy'] == "删除含缺失行":
        return df.dropna()
    return df


# ==================== 主程序流程 ====================
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed_df = data_preprocessing(df.copy())

    # ==================== 特征选择界面 ====================
    with st.sidebar:
        st.markdown("## 🎯 目标变量设置")
        target_col = st.selectbox("选择目标列", processed_df.columns)

        st.markdown("## 📊 特征选择")
        feature_cols = st.multiselect(
            "选择预测特征",
            [col for col in processed_df.columns if col != target_col]
        )

        st.markdown("## ⚙ 算法选择与参数设置")
        model_type = st.selectbox("选择分类算法",
                                  ["随机森林"])

        param_setting = {}
        with st.expander("算法参数设置"):
            if model_type == "随机森林":
                param_setting = {
                    'n_estimators': st.slider("树的数量", 10, 1000, 100),
                    'max_depth': st.slider("最大深度", 1, 50, 5)
                }


    if feature_cols:
        # ==================== 数据预处理管道 ====================
        # 编码目标变量
        le = LabelEncoder()
        y = le.fit_transform(processed_df[target_col])

        # 特征类型识别
        numeric_features = processed_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        categorical_features = list(set(feature_cols) - set(numeric_features))

        # 数值型特征处理
        numeric_steps = [('imputer', SimpleImputer(strategy='mean'))]
        if st.session_state.preprocess_params['scale_numeric']:
            numeric_steps.append(('scaler', StandardScaler()))

        numeric_transformer = Pipeline(numeric_steps)

        # 分类型特征处理
        categorical_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
        if st.session_state.preprocess_params['encode_categorical']:
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))

        categorical_transformer = Pipeline(categorical_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # ==================== 模型训练 ====================
        X = processed_df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 初始化模型
        if model_type == "随机森林":
            model = RandomForestClassifier(**param_setting)


        # 构建完整流程
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        full_pipeline.fit(X_train, y_train)

        # ==================== 模型评估 ====================
        st.subheader("📈 模型性能评估")
        y_pred = full_pipeline.predict(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("准确率", f"{accuracy_score(y_test, y_pred):.2%}")

            st.markdown("&zwnj;**分类报告**&zwnj;")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        with col2:
            st.markdown("&zwnj;**混淆矩阵**&zwnj;")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax)
            st.pyplot(fig)

        # 显示特征重要性（仅适用于树模型）
        if model_type in ["决策树", "随机森林"]:
            st.markdown("&zwnj;**特征重要性**&zwnj;")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # 获取处理后的特征名称
                feature_names = []
                if len(numeric_features) > 0:
                    feature_names += numeric_features
                if len(categorical_features) > 0:
                    ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
                    cat_names = ohe.get_feature_names_out(categorical_features)
                    feature_names += list(cat_names)

                importance_df = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': importances
                }).sort_values('重要性', ascending=False)

                st.dataframe(importance_df, height=300)


