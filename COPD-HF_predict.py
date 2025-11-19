import warnings

warnings.filterwarnings("ignore")
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, bootstrap
from tableone import TableOne
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from missforest import MissForest
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score, classification_report, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 设置工作路径
os.chdir("C:/Users/Zhefy/Desktop/smote（4）")

# 创建必要的目录
os.makedirs("数据", exist_ok=True)
os.makedirs("生成的表格", exist_ok=True)
os.makedirs("绘制的图片", exist_ok=True)
os.makedirs("训练好的模型", exist_ok=True)
os.makedirs("预测效果评价文件", exist_ok=True)


####################### 配置类 #######################
class Config:
    """配置参数"""
    # 路径配置
    DATA_PATH = "./COPD_HF_6m.csv"
    SHAP_SAVE_PATH = "shap图像分析"  # 新增：SHAP图像保存路径

    # 数据分割参数
    TEST_SIZE = 0.15
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    # 特征选择参数
    N_FEATURES = 30  # 减少特征数量防止过拟合

    # SMOTE参数
    SMOTE_K_NEIGHBORS = 5  # 减少邻居数

    # 模型参数
    CV_FOLDS = 5
    CI_N_BOOTSTRAPS = 1000  # bootstrap抽样次数，用于计算置信区间

    # SHAP分析参数
    SHAP_N_INTERPRET = 50
    SHAP_OBS_INDEX = 5


####################### 数据导入和预处理 #######################
def load_and_preprocess_data():
    """数据加载和预处理"""
    data = pd.read_csv("./COPD_HF_6m.csv", encoding="GBK")
    print("数据基本信息:")
    print(data.info())
    print("\n目标变量分布:")
    print(data['readm_6m'].value_counts())
    print(data['readm_6m'].value_counts(normalize=True))
    return data


####################### 数据填充 #######################
def impute_missing_values(data):
    """按readm_6m分组进行缺失值填充"""

    def impute_by_group(group_data):
        """对单个组进行缺失值填充"""
        try:
            # 检查 missforest 版本，适配不同版本的参数
            try:
                # 新版本可能使用 random_state
                imputer = MissForest(random_state=42)
            except TypeError:
                # 旧版本可能使用 random_seed 或其他参数
                try:
                    imputer = MissForest(random_seed=42)
                except TypeError:
                    # 如果都不支持，使用默认参数
                    imputer = MissForest()

            # 执行填充
            imputed_data = imputer.fit_transform(group_data)
            return imputed_data
        except Exception as e:
            print(f"填充过程中出错: {e}")
            # 如果 MissForest 失败，使用简单填充方法
            return group_data.fillna(group_data.median())

    # 确保数据备份
    data_backup = data.copy()

    # 分离特征和目标变量
    target_col = 'readm_6m'
    feature_cols = [col for col in data.columns if col != target_col]

    print("开始分组缺失值填充...")
    print(f"目标变量: {target_col}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"数据形状: {data.shape}")

    # 存储填充后的各组数据
    imputed_groups = []

    # 按目标变量分组进行填充
    for group_name, group_data in data.groupby(target_col):
        print(f"处理组: {target_col} = {group_name}, 样本数: {len(group_data)}")
        # 提取特征数据
        group_features = group_data[feature_cols].copy()
        # 检查缺失值
        missing_before = group_features.isnull().sum().sum()
        print(f"  填充前缺失值数量: {missing_before}")
        if missing_before > 0:
            # 进行缺失值填充
            imputed_features = impute_by_group(group_features)
            # 确保返回的是DataFrame
            if isinstance(imputed_features, np.ndarray):
                imputed_features = pd.DataFrame(imputed_features,
                                                columns=feature_cols,
                                                index=group_features.index)
            # 恢复目标变量
            imputed_group = imputed_features.copy()
            imputed_group[target_col] = group_name
        else:
            # 没有缺失值，直接使用原数据
            imputed_group = group_data.copy()
        imputed_groups.append(imputed_group)
    # 合并所有组
    if imputed_groups:
        data_imputed = pd.concat(imputed_groups, ignore_index=True)
    else:
        data_imputed = data_backup.copy()

    # 对整数变量进行四舍五入
    integer_columns = ['platelet', 'sodium', 'hco3', 'bun', 'alt', 'ast', 'sao2', 'pco2', 'ntprobnp']
    for col in integer_columns:
        if col in data_imputed.columns:
            data_imputed[col] = data_imputed[col].round().astype(int)

    print("插补后缺失值数量:")
    missing_after = data_imputed.isnull().sum()
    print(missing_after[missing_after > 0])

    if missing_after.sum() > 0:
        print("仍有缺失值，使用中位数填充剩余缺失值...")
        data_imputed = data_imputed.fillna(data_imputed.median())

    return data_imputed


def prepare_datasets(data_imputed):
    """准备训练集和测试集"""
    # 数据集划分
    train_data, test_data = train_test_split(
        data_imputed, test_size=Config.TEST_SIZE, stratify=data_imputed["readm_6m"], random_state=Config.RANDOM_STATE
    )

    # 保存原始数据集
    train_data.to_csv("数据/train_data_notscaled.csv", index=False)
    test_data.to_csv("数据/test_data_notscaled.csv", index=False)

    return train_data, test_data


def apply_smote(X_train, y_train):
    """应用SMOTE过采样"""
    print("SMOTE处理前的训练集类别分布:")
    print(Counter(y_train))

    smote = SMOTE(random_state=Config.RANDOM_STATE, k_neighbors=Config.SMOTE_K_NEIGHBORS)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("SMOTE处理后的训练集类别分布:")
    print(Counter(y_train_resampled))

    return X_train_resampled, y_train_resampled


def apply_undersample(X_train, y_train, sampling_strategy=0.8):
    """应用随机欠采样"""
    print("欠采样处理前的训练集类别分布:")
    print(Counter(y_train))

    # 初始化随机欠采样器
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=getattr(Config, 'RANDOM_STATE', 42)  # 若Config未定义则使用默认值
    )

    # 执行欠采样
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    print("欠采样处理后的训练集类别分布:")
    print(Counter(y_train_resampled))

    return X_train_resampled, y_train_resampled


####################### 特征选择 #######################
def select_top_features(X_train, y_train, n_features=30):
    """使用随机森林选择最重要的特征"""
    print("使用原始数据（非SMOTE）进行特征选择...")

    rf_selector = RandomForestClassifier(
        n_estimators=100,
        random_state=Config.RANDOM_STATE,
        max_depth=6  # 限制深度防止过拟合
    )
    rf_selector.fit(X_train, y_train)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)

    # 选择最重要的特征
    top_features = feature_importance.head(n_features)['feature'].tolist()

    print(f"最重要的 {n_features} 个特征:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")

    # 保存特征重要性结果
    feature_importance.to_csv("生成的表格/feature_importance.csv", index=False, encoding="utf-8-sig")

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(30), x='importance', y='feature')
    plt.title('Top 20 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig("绘制的图片/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    return top_features


####################### 数据标准化函数 #######################
def scale_selected_features(X_train, X_test, top_features):
    """对选定的特征进行标准化"""
    print("对选定的特征进行标准化...")

    # 只选择top_features进行标准化
    X_train_selected = X_train[top_features].copy()
    X_test_selected = X_test[top_features].copy()

    # 初始化标准化器
    scaler = MinMaxScaler()   # 使用MinMaxScaler标准化

    # 对训练集进行拟合和转换
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=top_features, index=X_train_selected.index)

    # 对测试集进行转换
    X_test_scaled = scaler.transform(X_test_selected)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=top_features, index=X_test_selected.index)

    print(f"标准化后的训练集形状: {X_train_scaled.shape}")
    print(f"标准化后的测试集形状: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


####################### 稳健的模型训练函数 #######################
def train_model_with_cv(model, param_grid, X_train, y_train):
    """使用交叉验证训练模型，返回最佳模型和最佳参数"""
    grid_search = GridSearchCV(
        model, param_grid, cv=Config.CV_FOLDS,
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """训练逻辑回归模型"""
    print("训练逻辑回归模型...")

    # 更保守的参数网格
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],  # 更强的正则化
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    model = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    # 验证集性能
    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_decision_tree(X_train, y_train, X_val, y_val):
    """训练决策树模型"""
    print("训练决策树模型...")

    # 限制复杂度的参数
    param_grid = {
        'max_depth': [3, 4, 5],  # 限制深度
        'min_samples_split': [20, 50],  # 增加最小分割样本
        'min_samples_leaf': [10, 20],  # 增加叶节点最小样本
        'max_features': ['sqrt', 0.5]  # 限制特征数量
    }

    model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_random_forest(X_train, y_train, X_val, y_val):
    """训练随机森林模型"""
    print("训练随机森林模型...")

    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7, 9],  # 限制深度
        'min_samples_split': [20, 35, 50, 75],  # 增加最小分割样本
        'min_samples_leaf': [10, 20, 30, 40],  # 增加叶节点最小样本
        'max_features': ['sqrt', 0.4]  # 限制特征数量
    }

    model = RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=-1)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_xgboost(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    print("训练XGBoost模型...")

    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15],  # 更小的学习率
        'max_depth': [3, 4, 5, 6, 7],  # 限制深度
        'n_estimators': [100, 150, 200, 300],
        'subsample': [0.75],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1, 1],  # L1正则化
        'reg_lambda': [1, 10]  # L2正则化
    }

    model = XGBClassifier(random_state=Config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_lightgbm(X_train, y_train, X_val, y_val):
    """训练LightGBM模型"""
    print("训练LightGBM模型...")

    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'num_leaves': [15, 31, 50],  # 限制叶节点数
        'n_estimators': [100, 200, 300],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1, 1],
        'reg_lambda': [1, 10],
        'min_child_samples': [10, 25, 50]  # 增加最小子样本数
    }

    model = lgb.LGBMClassifier(random_state=Config.RANDOM_STATE, verbose=-1)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_svm(X_train, y_train, X_val, y_val):
    """训练SVM模型"""
    print("训练SVM模型...")

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    model = SVC(probability=True, random_state=Config.RANDOM_STATE)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


def train_ann(X_train, y_train, X_val, y_val):
    """训练ANN模型"""
    print("训练ANN模型...")

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # 简化网络结构
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01, 0.1]  # 添加正则化
    }

    model = MLPClassifier(random_state=Config.RANDOM_STATE, max_iter=1000, early_stopping=True)
    best_model, best_params = train_model_with_cv(model, param_grid, X_train, y_train)

    y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"验证集AUC: {val_auc:.4f}")

    return best_model, best_params


####################### 模型评估与置信区间计算 #######################
def calculate_ci_bootstrap(y_true, y_score, metric_func, n_bootstraps=1000, random_state=42):
    """
    使用bootstrap方法计算指标的95%置信区间
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # 有放回抽样
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            continue  # 确保抽样后仍有两个类别

        score = metric_func(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)

    # 计算95%置信区间
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper


def calculate_metrics(y_true, y_pred, y_pred_prob):
    """计算各种评估指标，包括AUC和AUPRC的95%置信区间"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)  # 特异度
    auc_score = roc_auc_score(y_true, y_pred_prob)

    # 计算AUC的95%置信区间
    auc_ci = calculate_ci_bootstrap(
        y_true, y_pred_prob,
        lambda yt, ys: roc_auc_score(yt, ys),
        n_bootstraps=Config.CI_N_BOOTSTRAPS,
        random_state=Config.RANDOM_STATE
    )

    # 计算PR曲线的AUC及其置信区间
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall_curve, precision_curve)

    auprc_ci = calculate_ci_bootstrap(
        y_true, y_pred_prob,
        lambda yt, ys: auc(*precision_recall_curve(yt, ys)[1:]),
        n_bootstraps=Config.CI_N_BOOTSTRAPS,
        random_state=Config.RANDOM_STATE
    )

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'AUC': auc_score,
        'AUC_95CI_lower': auc_ci[0],
        'AUC_95CI_upper': auc_ci[1],
        'AUPRC': auprc,
        'AUPRC_95CI_lower': auprc_ci[0],
        'AUPRC_95CI_upper': auprc_ci[1]
    }


def comprehensive_evaluation(y_true, y_pred, y_pred_prob, model_name):
    """更全面的模型评估"""
    # 基础指标
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob)

    # 分类报告
    print(f"\n{classification_report(y_true, y_pred)}")

    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.plot(recall, precision,
             label=f'PR Curve (AUPRC = {metrics["AUPRC"]:.2f} [{metrics["AUPRC_95CI_lower"]:.2f}-{metrics["AUPRC_95CI_upper"]:.2f}])')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"绘制的图片/pr_curve_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制单模型ROC曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr,
             label=f'ROC Curve (AUC = {metrics["AUC"]:.2f} [{metrics["AUC_95CI_lower"]:.2f}-{metrics["AUC_95CI_upper"]:.2f}])')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"绘制的图片/roc_curve_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return metrics


####################### 可视化函数 #######################
def plot_confusion_matrix(y_true, y_pred, model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"绘制的图片/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(model, X_train, y_train, model_name):
    """绘制学习曲线"""
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc', random_state=Config.RANDOM_STATE
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('AUC Score')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"绘制的图片/learning_curve_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


####################### 可视化函数（支持多模型对比） #######################
def plot_multi_roc_curve(model_results, save_path):
    """绘制多模型ROC曲线（单图对比）"""
    plt.figure(figsize=(10, 8))
    # 定义不同模型的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0

    for model_name, (y_true, y_pred_prob, metrics) in model_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = metrics['AUC']
        ci_lower = metrics['AUC_95CI_lower']
        ci_upper = metrics['AUC_95CI_upper']
        # 循环使用颜色，避免索引超出
        color = colors[color_idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])')
        color_idx += 1

    # 绘制对角线（随机猜测线）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve Comparison of All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_dca_curve(model_results, save_path):
    """绘制多模型DCA曲线（单图对比）"""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0
    thresholds = np.linspace(0, 1, 100)  # 统一阈值范围，确保对比公平

    for model_name, (y_true, y_pred_prob, _) in model_results.items():
        net_benefits = []
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            n = len(y_true)
            if threshold == 1:  # 避免分母为0
                net_benefit = 0
            else:
                net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)

        color = colors[color_idx % len(colors)]
        plt.plot(thresholds, net_benefits, color=color, lw=2, label=model_name)
        color_idx += 1

    # 绘制参考线（Treat None 和 Treat All）
    plt.plot(thresholds, np.zeros_like(thresholds), 'k--', label='Treat None')
    y_true_mean = np.mean(list(model_results.values())[0][0])  # 取第一个模型的y_true计算均值
    treat_all_nb = y_true_mean - (1 - y_true_mean) * thresholds / (1 - thresholds)
    plt.plot(thresholds, treat_all_nb, 'r--', label='Treat All')

    plt.xlim([0, 1])
    plt.ylim([-0.1, 0.5])  # 固定y轴范围，便于对比
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('DCA Curve Comparison of All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_pr_curve(model_results, save_path):
    """绘制多模型PR曲线（单图对比）"""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0

    for model_name, (y_true, y_pred_prob, metrics) in model_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        auprc = metrics['AUPRC']
        ci_lower = metrics['AUPRC_95CI_lower']
        ci_upper = metrics['AUPRC_95CI_upper']
        color = colors[color_idx % len(colors)]
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{model_name} (AUPRC = {auprc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])')
        color_idx += 1

    # 绘制随机猜测线（正例比例）
    pos_ratio = np.mean(list(model_results.values())[0][0])
    plt.plot([0, 1], [pos_ratio, pos_ratio], 'k--',
             label=f'Random Guess (Pos Ratio = {pos_ratio:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR Curve Comparison of All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


####################### 过拟合诊断 #######################
def diagnose_overfitting(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """诊断过拟合情况"""
    print("\n=== 过拟合诊断 ===")

    diagnosis_results = {}

    for model_name, model in models.items():
        # 训练集表现
        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred_prob)

        # 验证集表现
        y_val_pred_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_prob)

        # 测试集表现
        y_test_pred_prob = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred_prob)

        diagnosis_results[model_name] = {
            'Train_AUC': train_auc,
            'Val_AUC': val_auc,
            'Test_AUC': test_auc,
            'Overfitting_Degree': train_auc - test_auc,
            'Generalization_Gap': val_auc - test_auc
        }

        print(f"\n{model_name}:")
        print(f"  训练集 AUC: {train_auc:.4f}")
        print(f"  验证集 AUC: {val_auc:.4f}")
        print(f"  测试集 AUC: {test_auc:.4f}")
        print(f"  过拟合程度: {train_auc - test_auc:.4f}")
        print(f"  泛化差距: {val_auc - test_auc:.4f}")

    # 保存诊断结果
    diagnosis_df = pd.DataFrame(diagnosis_results).T
    diagnosis_df.to_csv("生成的表格/overfitting_diagnosis.csv", encoding="utf-8-sig")

    return diagnosis_results


####################### SHAP可解释性分析函数 #######################
def shap_explain_best_model(best_model, best_model_name, X_train, X_test, save_path, n_interpret=50, obs_index=5):
    """对最佳模型执行SHAP可解释性分析"""
    # 创建SHAP图像保存目录
    shap_save_dir = os.path.join(save_path, "shap图像分析")
    os.makedirs(shap_save_dir, exist_ok=True)
    print(f"SHAP图像将保存至: {shap_save_dir}")

    # 处理数据格式
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(X_train.shape[1])])
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

    # 根据模型类型选择SHAP解释逻辑
    if "Logistic Regression" in best_model_name:
        def model_predict(X):
            return best_model.predict(X)

        explainer = shap.KernelExplainer(model_predict, X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])
    else:
        def model_predict_proba(X):
            return best_model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(model_predict_proba, X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])

    # 生成并保存SHAP图像
    plt.figure(figsize=(15, 12))
    shap.plots.bar(shap_values, max_display=16, show=False)
    plt.title(f"SHAP Feature Importance (Bar) - {best_model_name}", fontsize=14)
    plt.savefig(os.path.join(shap_save_dir, f"bar_{best_model_name.replace(' ', '_').lower()}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 12))
    shap.plots.beeswarm(shap_values, max_display=16, show=False)
    plt.title(f"SHAP Feature Importance (Beeswarm) - {best_model_name}", fontsize=14)
    plt.savefig(os.path.join(shap_save_dir, f"beeswarm_{best_model_name.replace(' ', '_').lower()}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    # 单个样本解释
    plt.figure(figsize=(15, 12))
    shap.plots.waterfall(shap_values[obs_index], max_display=16, show=False)
    plt.title(f"SHAP Waterfall Plot (Obs Index: {obs_index}) - {best_model_name}", fontsize=14)
    plt.savefig(os.path.join(shap_save_dir, f"waterfall_{obs_index}_{best_model_name.replace(' ', '_').lower()}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    # 力图（单个样本）
    force_plot_sig = shap.plots.force(shap_values[obs_index], feature_names=X_train.columns, show=False)
    shap.save_html(
        os.path.join(shap_save_dir, f"force_plot_sig_{obs_index}_{best_model_name.replace(' ', '_').lower()}.html"),
        force_plot_sig)

    print(f"模型 {best_model_name} 的SHAP分析完成，图像已保存至 {shap_save_dir}")


####################### 主函数 #######################
def main():
    """主函数"""
    config = Config()

    # 1. 数据加载和预处理
    print("=== 数据加载和预处理 ===")
    data = load_and_preprocess_data()
    data_imputed = impute_missing_values(data)
    train_data, test_data = prepare_datasets(data_imputed)

    # 2. 准备特征和目标变量
    X_train_original = train_data.drop('readm_6m', axis=1)
    y_train_original = train_data['readm_6m']
    X_test_original = test_data.drop('readm_6m', axis=1)
    y_test = test_data['readm_6m']

    # 3. 特征选择
    print("\n=== 特征选择 ===")
    top_features = select_top_features(X_train_original, y_train_original, n_features=config.N_FEATURES)

    # 4. 数据标准化
    print("\n=== 数据标准化 ===")
    X_train_scaled, X_test_scaled, scaler = scale_selected_features(X_train_original, X_test_original, top_features)

    # 5. 应用采样
    print("\n=== 应用采样 ===")
    X_train_resampled, y_train_resampled = apply_undersample(X_train_scaled, y_train_original)

    # 6. 数据集分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_resampled,
        y_train_resampled,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_train_resampled
    )

    # 7. 模型训练和调优
    print("\n=== 模型训练和调优 ===")
    models = {}
    best_params = {}  # 存储每个模型的最优参数

    lr_model, lr_params = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = lr_model
    best_params['Logistic Regression'] = lr_params

    dt_model, dt_params = train_decision_tree(X_train, y_train, X_val, y_val)
    models['Decision Tree'] = dt_model
    best_params['Decision Tree'] = dt_params

    rf_model, rf_params = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    best_params['Random Forest'] = rf_params

    xgb_model, xgb_params = train_xgboost(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model
    best_params['XGBoost'] = xgb_params

    lgb_model, lgb_params = train_lightgbm(X_train, y_train, X_val, y_val)
    models['LightGBM'] = lgb_model
    best_params['LightGBM'] = lgb_params

    svm_model, svm_params = train_svm(X_train, y_train, X_val, y_val)
    models['SVM'] = svm_model
    best_params['SVM'] = svm_params

    ann_model, ann_params = train_ann(X_train, y_train, X_val, y_val)
    models['ANN'] = ann_model
    best_params['ANN'] = ann_params

    # 输出并保存每个模型的最优参数
    print("\n=== 各模型最优参数 ===")
    for model_name, params in best_params.items():
        print(f"\n{model_name} 最优参数:")
        for param, value in params.items():
            print(f"  {param}: {value}")

    # 保存最优参数到文件
    with open("生成的表格/model_best_params.pkl", 'wb') as f:
        pickle.dump(best_params, f)
    # 同时保存为CSV格式（便于阅读）
    params_df = pd.DataFrame([str(params) for params in best_params.values()],
                             index=best_params.keys(),
                             columns=['Best Parameters'])
    params_df.to_csv("生成的表格/model_best_params.csv", encoding="utf-8-sig")

    # 8. 过拟合诊断
    diagnosis_results = diagnose_overfitting(models, X_train, y_train, X_val, y_val, X_test_scaled, y_test)

    # 9. 模型评估
    print("\n=== 模型评估 ===")
    results = {}
    model_results = {}  # 存储(y_true, y_pred_prob, metrics)
    for model_name, model in models.items():
        print(f"\n评估 {model_name}:")
        y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = model.predict(X_test_scaled)
        metrics = comprehensive_evaluation(y_test, y_test_pred, y_test_pred_prob, model_name)
        results[model_name] = metrics
        model_results[model_name] = (y_test, y_test_pred_prob, metrics)
        plot_confusion_matrix(y_test, y_test_pred, model_name)
        try:
            plot_learning_curve(model, X_train, y_train, model_name)
        except Exception as e:
            print(f"绘制学习曲线失败: {e}")
        print(f"{model_name} 测试集性能:")
        print(f"  AUC: {metrics['AUC']:.4f} (95% CI: {metrics['AUC_95CI_lower']:.4f}-{metrics['AUC_95CI_upper']:.4f})")
        print(
            f"  AUPRC: {metrics['AUPRC']:.4f} (95% CI: {metrics['AUPRC_95CI_lower']:.4f}-{metrics['AUPRC_95CI_upper']:.4f})")

    # 10. 统一绘制多模型对比图
    print("\n=== 绘制多模型对比曲线 ===")
    plot_multi_roc_curve(model_results, "绘制的图片/multi_model_roc_curve.png")
    plot_multi_dca_curve(model_results, "绘制的图片/multi_model_dca_curve.png")
    plot_multi_pr_curve(model_results, "绘制的图片/multi_model_pr_curve.png")
    print("多模型对比曲线已保存至：绘制的图片/ 目录下")

    # 11. SHAP分析
    print("\n=== SHAP可解释性分析 ===")
    best_model_name = max(results, key=lambda x: results[x]['AUC'])
    best_model = models[best_model_name]
    print(f"对最佳模型 {best_model_name} 执行SHAP分析")

    shap_explain_best_model(
        best_model=best_model,
        best_model_name=best_model_name,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        save_path="绘制的图片",
        n_interpret=Config.SHAP_N_INTERPRET,
        obs_index=Config.SHAP_OBS_INDEX
    )

    # 对LightGBM模型进行SHAP分析
    lgb_model_name = "LightGBM"
    lgb_model = models[lgb_model_name]
    print(f"\n对LightGBM模型 {lgb_model_name} 执行SHAP分析")
    shap_explain_best_model(
        best_model=lgb_model,
        best_model_name=lgb_model_name,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        save_path="绘制的图片",
        n_interpret=Config.SHAP_N_INTERPRET,
        obs_index=Config.SHAP_OBS_INDEX
    )

    # 12. 保存结果
    print("\n=== 保存结果 ===")
    for model_name, model in models.items():
        with open(f"训练好的模型/{model_name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    results_df = pd.DataFrame(results).T
    results_df.to_csv("预测效果评价文件/model_performance_comparison.csv", index=True, encoding="utf-8-sig")
    print("模型性能比较已保存到: 预测效果评价文件/model_performance_comparison.csv")

    # 13. 性能对比
    print("\n=== 最终模型性能对比 ===")
    print(
        results_df[['AUC', 'AUC_95CI_lower', 'AUC_95CI_upper', 'AUPRC', 'AUPRC_95CI_lower', 'AUPRC_95CI_upper']].round(
            4))
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
    results_df[metrics_to_plot].plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("绘制的图片/model_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 14. 输出总结
    print("\n=== 训练总结 ===")
    print(f"最佳模型: {best_model_name}")
    print(
        f"最佳测试集AUC: {results[best_model_name]['AUC']:.4f} (95% CI: {results[best_model_name]['AUC_95CI_lower']:.4f}-{results[best_model_name]['AUC_95CI_upper']:.4f})")
    print(f"使用的特征数量: {len(top_features)}")
    print(f"SHAP分析结果已保存至: {config.SHAP_SAVE_PATH}")
    print("训练完成！")


if __name__ == "__main__":
    main()