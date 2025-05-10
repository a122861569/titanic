# titanic
https://github.com/a122861569/titanic/blob/4026d3e8ebf61888f4ee4a5eec3eb7a55ee31edd/%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7%E5%AD%98%E6%B4%BB%E7%8E%87.ipynb
🧾 1. 明确问题（Problem Definition）
目标：预测乘客是否在 Titanic 沉船中幸存（即二分类问题）。

标签（Target）：Survived（0=未幸存，1=幸存）

📥 2. 数据获取与初步理解（Data Collection & EDA）
加载训练集和测试集（例如 train.csv, test.csv）。

使用 .head(), .info(), .describe() 初步查看数据。

利用可视化（如 sns.countplot）探索变量与目标的关系。

🧼 3. 数据清洗与整合（💥最重要的部分之一）
数据清洗对模型的预测能力至关重要。你已经做了以下几项：

✅ 缺失值处理（Missing Values）
Age：按 Sex 和 Pclass 分组后用中位数填补：

python
复制
编辑
train_data["Age"] = train_data.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
Embarked：用众数填充。

Fare（测试集）：用中位数填充。

Cabin：缺失处理为二元变量 Has_Cabin。

✅ 特征工程（Feature Engineering）
性别转数字：Sex 由 male/female → 0/1。

登船港口转哑变量（独热编码）：Embarked → Emb_C, Emb_Q, Emb_S。

家庭人数合并：FamilySize = SibSp + Parch + 1。

提取头衔 Title：从 Name 字段中用正则提取称呼，并做映射归类。

标准化 Age 和 Fare：使用 StandardScaler() 保证这些特征数值尺度统一，避免模型偏向数值大的变量。

🧱 4. 特征选择与构建训练集（Feature Selection）
选择有意义的字段组成 X_train，如：

python
复制
编辑
features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "Title"]
X_train = train_data[features]
y_train = train_data["Survived"]
🔧 5. 建立模型（Model Building）
使用 RandomForestClassifier 创建初始模型，并用 .fit() 拟合训练数据。

🔍 6. 模型评估（Model Evaluation）
用 cross_val_score 进行 交叉验证，获取模型泛化能力的平均准确率和标准差。

🔎 7. 模型调参（Hyperparameter Tuning）
使用 GridSearchCV 对 n_estimators 和 max_depth 等参数进行网格搜索，选出效果最优的组合。

📈 8. 模型预测（Prediction）
用 .predict() 预测测试集的结果，并生成提交文件 submission.csv。

📤 9. 结果提交（Submission）
创建提交用的 DataFrame，并保存为 .csv 文件上传到竞赛平台（如 Kaggle）。
