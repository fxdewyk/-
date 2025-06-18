import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 初始化会话状态
if 'preprocess_params' not in st.session_state:
    st.session_state.preprocess_params = {
        'scale_numeric': True,
        'encode_categorical': True,
        'missing_strategy': '删除含缺失行'
    }

st.set_page_config(page_title="🌲 高级随机森林回归", layout="wide")
st.title("🌲 高级随机森林回归分析")

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

        st.markdown("## ⚙ 高级参数设置")
        with st.expander("树参数设置"):
            params = {
                'n_estimators': st.slider("树的数量", 10, 1000, 100),
                'max_depth': st.slider("最大深度", 1, 50, 5),
                'min_samples_split': st.slider("最小分割样本", 2, 20, 2)
            }

        with st.expander("交叉验证设置"):
            cv_settings = {
                'cv_folds': st.slider("交叉验证折数", 2, 10, 5),
                'test_size': st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            }

    if feature_cols:
        # ==================== 数据预处理管道 ====================
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
        y = processed_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cv_settings['test_size'],
            random_state=42
        )

        # 超参数网格搜索
        model = GridSearchCV(
            RandomForestRegressor(),
            param_grid={k: [v] for k, v in params.items()},
            cv=cv_settings['cv_folds'],
            scoring='neg_mean_squared_error'
        )
        model.fit(preprocessor.fit_transform(X_train), y_train)

        # ==================== 模型评估 ====================
        st.subheader("📈 模型性能评估")
        y_pred = model.predict(preprocessor.transform(X_test))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        with col2:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("最佳参数", str(model.best_params_))

        # ==================== 可视化模块 ====================
        tabs = st.tabs(["📈 预测效果", "⭐ 特征重要性", "🔍 SHAP解释", "📊 残差分析"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("实际值")
            ax.set_ylabel("预测值")
            st.pyplot(fig)

        with tabs[1]:
            importance = model.best_estimator_.feature_importances_
            features = preprocessor.get_feature_names_out()

            fig, ax = plt.subplots(figsize=(10, 6))
            pd.Series(importance, index=features).nlargest(10).plot.barh(ax=ax)
            ax.set_title("Top 10 重要特征")
            st.pyplot(fig)

        with tabs[2]:
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(preprocessor.transform(X_test))

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, preprocessor.transform(X_test),
                            feature_names=features, plot_type="bar")
            st.pyplot(fig)

        with tabs[3]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            residuals = y_test - y_pred
            ax1.hist(residuals, bins=30)
            ax1.set_title("残差分布")

            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_title("残差 vs 预测值")

            st.pyplot(fig)

        # ==================== 模型保存模块 ====================
        st.sidebar.markdown("## 💾 模型管理")
        if st.sidebar.button("保存当前模型"):
            joblib.dump({
                'model': model.best_estimator_,
                'preprocessor': preprocessor,
                'feature_cols': feature_cols
            }, 'random_forest_model.pkl')
            st.sidebar.success("模型已保存为 random_forest_model.pkl")

else:
    st.info("📥 请上传CSV格式的数据文件")
