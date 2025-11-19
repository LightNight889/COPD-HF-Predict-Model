import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入SHAP，如果失败则提供替代方案
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP库未安装，将使用特征重要性替代SHAP分析")

class COPDReadmissionPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("COPD患者6个月再入院风险预测系统")
        self.root.geometry("1000x700")
        
        # 首先检查当前目录
        self.check_current_directory()
        
        # 加载模型
        self.model = self.load_model()
        if self.model is None:
            messagebox.showerror("错误", "无法加载模型文件，请确保random_forest_model.pkl与程序在同一目录")
            return
            
        # 初始化SHAP解释器
        self.explainer = None
        if SHAP_AVAILABLE:
            self.init_shap_explainer()
        
        # 定义特征（使用更易理解的命名）
        self.numeric_features = [
            'ntprobnp', 'ph', 'neutrophil', 'bun', 'troponin-t', 'platelet', 'PaCO2', 
            'eosinophil', 'sodium', 'HCO₃⁻', 'creatinine', 'alt', 'albumin', 'hemoglobin', 
            'wbc', 'SaO2', 'ast', 'potassium', 'calcium', 'age', 'total bilirubin'
        ]
        
        self.binary_features = [
            'diabetes', 'LABA-LAMA', 'osteoporosis', 'arrhythmia', 'SABA/SAMA', 
            'LABA/LAMA', 'MRA', 'Beta 1-blockers', 'mono-ICS'
        ]
        
        # 原始特征名到显示名的映射
        self.feature_mapping = {
            'troponin-t': 'troponin_t',
            'PaCO2': 'pco2',
            'HCO₃⁻': 'hco3',
            'SaO2': 'sao2',
            'age': 'age_int',
            'total bilirubin': 'total_bilirubin',
            'LABA-LAMA': 'laba_lama_1',
            'SABA/SAMA': 'saba_sama_1',
            'LABA/LAMA': 'long_bron_only',
            'MRA': 'mra',
            'Beta 1-blockers': 'beta1_sel',
            'mono-ICS': 'mono_ics_1'
        }
        
        # 创建输入框架
        self.create_input_frame()
        
        # 创建结果框架
        self.create_result_frame()
        
        # 创建SHAP图框架
        self.create_shap_frame()
    
    def check_current_directory(self):
        """检查当前目录和文件"""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("=" * 50)
        print("目录检查:")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本所在目录: {current_dir}")
        print(f"模型文件路径: {os.path.join(current_dir, 'random_forest_model.pkl')}")
        print(f"模型文件是否存在: {os.path.exists(os.path.join(current_dir, 'random_forest_model.pkl'))}")
        
        print("\n当前目录文件列表:")
        for file in os.listdir(current_dir):
            file_path = os.path.join(current_dir, file)
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "文件夹"
            print(f"  - {file} ({file_size})")
        print("=" * 50)
    
# 修改 load_model 方法
    def load_model(self):
        """加载预训练的随机森林模型"""
        import os
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "random_forest_model.pkl")
            train_data_path = os.path.join(current_dir, "train_data_notscaled.csv")
            
            # 检查并加载训练数据以拟合scaler
            if os.path.exists(train_data_path):
                print(f"正在加载训练数据: {train_data_path}")
                # 加载训练数据
                train_data = pd.read_csv(train_data_path)
                
                # 分离特征列（假设最后一列是目标变量）
                feature_columns = train_data.columns[:-1]
                X_train = train_data[feature_columns]
                
                # 初始化并拟合MinMaxScaler
                self.scaler = MinMaxScaler()
                self.scaler.fit(X_train)
                print("MinMaxScaler已根据训练数据拟合完成")
            else:
                print("未找到训练数据文件，将使用默认scaler")
                self.scaler = MinMaxScaler()
            
            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在于 {model_path}")
                # 尝试在工作目录查找
                work_dir_model = "random_forest_model.pkl"
                if os.path.exists(work_dir_model):
                    print(f"在工作目录找到模型文件: {work_dir_model}")
                    model_path = work_dir_model
                else:
                    return None
            
            print(f"正在加载模型: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # 如果保存的是包含模型和scaler的字典
            if isinstance(model_data, dict):
                self.model = model_data['model']
                # 如果字典中有scaler，则使用它更新我们已经拟合的scaler
                if 'scaler' in model_data:
                    print("使用模型文件中的scaler参数")
                    self.scaler = model_data['scaler']
            else:
                # 如果只保存了模型
                self.model = model_data
                
            print("模型加载成功")
            return self.model
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def init_shap_explainer(self):
        """初始化SHAP解释器"""
        try:
            # 创建SHAP解释器 - 简化版本
            self.explainer = shap.TreeExplainer(self.model)
            print("SHAP解释器初始化成功")
        except Exception as e:
            print(f"SHAP解释器初始化失败: {str(e)}")
            self.explainer = None
    
    def create_input_frame(self):
        """创建输入框架"""
        input_frame = ttk.LabelFrame(self.root, text="患者特征输入", padding=10)
        input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 创建滚动框架
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 存储输入变量
        self.input_vars = {}
        
        # 创建数值型特征输入
        ttk.Label(scrollable_frame, text="实验室检查和生命体征", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        
        row = 1
        for i, feature in enumerate(self.numeric_features):
            ttk.Label(scrollable_frame, text=feature).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            var = tk.DoubleVar(value=0.0)
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.input_vars[feature] = var
            
            # 添加单位标签
            unit = self.get_feature_unit(feature)
            if unit:
                ttk.Label(scrollable_frame, text=unit).grid(row=row, column=2, sticky="w", padx=5)
            
            row += 1
        
        # 创建二分类特征输入
        ttk.Label(scrollable_frame, text="合并症和用药情况 (是/否)", font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10, 10)
        )
        
        row += 1
        for feature in self.binary_features:
            ttk.Label(scrollable_frame, text=feature).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value="否")
            combobox = ttk.Combobox(scrollable_frame, textvariable=var, values=["是", "否"], state="readonly", width=13)
            combobox.grid(row=row, column=1, padx=5, pady=2)
            self.input_vars[feature] = var
            row += 1
        
        # 添加按钮
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="开始预测", command=self.predict).pack(side="left", padx=5)
        ttk.Button(button_frame, text="特征重要性分析", command=self.show_shap_plot).pack(side="left", padx=5)
        ttk.Button(button_frame, text="清空输入", command=self.clear_inputs).pack(side="left", padx=5)
        
        # 打包canvas和scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_result_frame(self):
        """创建结果显示框架"""
        self.result_frame = ttk.LabelFrame(self.root, text="预测结果", padding=10)
        self.result_frame.pack(fill="x", padx=10, pady=5)
        
        self.result_label = ttk.Label(
            self.result_frame, 
            text="请输入患者特征后点击'开始预测'", 
            font=("Arial", 12),
            wraplength=800
        )
        self.result_label.pack(fill="x", pady=10)
    
    def create_shap_frame(self):
        """创建SHAP图显示框架"""
        self.shap_frame = ttk.LabelFrame(self.root, text="特征重要性分析", padding=10)
        self.shap_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.shap_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 初始显示提示
        self.ax.text(0.5, 0.5, "点击'特征重要性分析'按钮显示特征重要性", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def get_feature_unit(self, feature):
        """获取特征的医学单位"""
        units = {
            'ntprobnp': 'pg/mL',
            'ph': '',
            'neutrophil': '×10^9/L',
            'bun': 'mg/dL',
            'troponin-t': 'ng/mL',
            'platelet': '×10^9/L',
            'PaCO2': 'mmHg',
            'eosinophil': '×10^9/L',
            'sodium': 'mmol/L',
            'HCO₃⁻': 'mmol/L',
            'creatinine': 'mg/dL',
            'alt': 'U/L',
            'albumin': 'g/dL',
            'hemoglobin': 'g/dL',
            'wbc': '×10^9/L',
            'SaO2': '%',
            'ast': 'U/L',
            'potassium': 'mmol/L',
            'calcium': 'mg/dL',
            'age': '岁',
            'total bilirubin': 'mg/dL'
        }
        return units.get(feature, '')
    
    def convert_to_model_features(self, data):
        """将显示特征名转换为模型特征名"""
        model_data = {}
        for display_name, value in data.items():
            # 如果特征名在映射中，使用映射后的名称，否则使用原名称
            model_name = self.feature_mapping.get(display_name, display_name)
            model_data[model_name] = value
        return model_data
    
    def get_input_data(self):
        """获取用户输入的数据"""
        try:
            data = {}
            
            # 处理数值型特征
            for feature in self.numeric_features:
                value = self.input_vars[feature].get()
                if value == "":
                    messagebox.showwarning("输入错误", f"请填写{feature}的值")
                    return None
                data[feature] = float(value)
            
            # 处理二分类特征
            for feature in self.binary_features:
                value = self.input_vars[feature].get()
                data[feature] = 1 if value == "是" else 0
            
            # 转换为模型使用的特征名
            model_data = self.convert_to_model_features(data)
            return model_data
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数值")
            return None
    
    def predict(self):
        """进行预测"""
        data = self.get_input_data()
        if data is None:
            return
        
        # 转换为DataFrame（按正确的特征顺序）
        feature_order = [
            'ntprobnp', 'ph', 'neutrophil', 'bun', 'troponin_t', 'platelet', 'pco2', 
            'eosinophil', 'sodium', 'hco3', 'creatinine', 'alt', 'albumin', 'hemoglobin', 
            'wbc', 'sao2', 'ast', 'potassium', 'calcium', 'age_int', 'total_bilirubin',
            'diabetes', 'laba_lama_1', 'osteoporosis', 'arrhythmia', 'saba_sama_1', 
            'long_bron_only', 'mra', 'beta1_sel', 'mono_ics_1'
        ]
        
        # 确保数据按正确顺序排列
        ordered_data = {feature: data[feature] for feature in feature_order}
        df = pd.DataFrame([ordered_data])
        
        # 使用MinMaxScaler标准化
        X_scaled = self.scaler.transform(df)
        
        # 预测概率
        probability = self.model.predict_proba(X_scaled)[0, 1]
        probability_percent = round(probability * 100, 2)
        
        # 显示结果
        result_text = f"此COPD合并病人6个月再入院的概率为{probability_percent}%"
        self.result_label.config(text=result_text)
        
        # 存储当前样本用于分析
        self.current_sample = X_scaled[0]
        self.current_sample_features = feature_order  # 存储特征顺序
        self.current_sample_df = df  # 存储原始数据框
    
    def show_shap_plot(self):
        """显示特征重要性图"""
        if not hasattr(self, 'current_sample'):
            messagebox.showwarning("警告", "请先进行预测")
            return
        
        try:
            # 清空图形
            self.ax.clear()
            
            # 创建显示用的特征名（将模型特征名映射回显示名）
            display_feature_names = []
            for model_name in self.current_sample_features:
                # 反向查找显示名
                display_name = model_name
                for disp, mod in self.feature_mapping.items():
                    if mod == model_name:
                        display_name = disp
                        break
                display_feature_names.append(display_name)
            
            # 尝试使用SHAP分析
            if SHAP_AVAILABLE and self.explainer is not None:
                try:
                    # 正确地准备数据供SHAP使用
                    # 使用原始DataFrame而不是缩放后的数组
                    sample_for_shap = self.current_sample_df.values
                    
                    # 计算SHAP值
                    shap_values = self.explainer.shap_values(sample_for_shap)
                    
                    # 处理SHAP值（随机森林可能返回列表）
                    if isinstance(shap_values, list):
                        # 对于分类问题，取第二个类的SHAP值（正类）
                        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        shap_vals = shap_values[0]
                    
                    # 创建特征重要性条形图
                    feature_importance = pd.DataFrame({
                        'feature': display_feature_names,
                        'importance': np.abs(shap_vals)
                    })
                    feature_importance = feature_importance.sort_values('importance', ascending=True)
                    
                    # 绘制水平条形图
                    y_pos = np.arange(len(feature_importance))
                    self.ax.barh(y_pos, feature_importance['importance'])
                    self.ax.set_yticks(y_pos)
                    self.ax.set_yticklabels(feature_importance['feature'])
                    self.ax.set_xlabel('SHAP值 (特征重要性)')
                    self.ax.set_title('基于SHAP的特征重要性分析')
                    
                except Exception as e:
                    print(f"SHAP分析失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 如果SHAP失败，回退到使用模型自带的特征重要性
                    self.fallback_feature_importance(display_feature_names)
            else:
                # 使用模型自带的特征重要性
                self.fallback_feature_importance(display_feature_names)
            
            # 调整图形
            self.fig.tight_layout()
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"生成特征重要性图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def fallback_feature_importance(self, display_feature_names):
        """回退方法：使用模型自带的特征重要性"""
        try:
            # 获取模型的特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # 创建特征重要性DataFrame
                feature_importance = pd.DataFrame({
                    'feature': display_feature_names,
                    'importance': importances
                })
                feature_importance = feature_importance.sort_values('importance', ascending=True)
                
                # 绘制水平条形图
                y_pos = np.arange(len(feature_importance))
                self.ax.barh(y_pos, feature_importance['importance'])
                self.ax.set_yticks(y_pos)
                self.ax.set_yticklabels(feature_importance['feature'])
                self.ax.set_xlabel('特征重要性')
                self.ax.set_title('基于随机森林的特征重要性分析')
            else:
                # 如果模型没有feature_importances_属性，显示错误信息
                self.ax.text(0.5, 0.5, "无法获取特征重要性数据", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=self.ax.transAxes, fontsize=14)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                
        except Exception as e:
            print(f"回退方法也失败: {str(e)}")
            self.ax.text(0.5, 0.5, "特征重要性分析不可用", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=self.ax.transAxes, fontsize=14)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
    
    def clear_inputs(self):
        """清空所有输入"""
        for var in self.input_vars.values():
            if isinstance(var, tk.DoubleVar):
                var.set(0.0)
            else:
                var.set("否")
        
        self.result_label.config(text="请输入患者特征后点击'开始预测'")
        
        # 清空SHAP图
        self.ax.clear()
        self.ax.text(0.5, 0.5, "点击'特征重要性分析'按钮显示特征重要性", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = COPDReadmissionPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()