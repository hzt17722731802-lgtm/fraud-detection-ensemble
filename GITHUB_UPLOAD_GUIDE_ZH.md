# 上传到 GitHub

## 推荐填写

- Repository name：`transaction-fraud-detection-ensemble`
- README 标题：`Transaction Fraud Detection with Majority Voting`
- Description：`Compare five classical ML classifiers and a majority-voting ensemble for transaction fraud detection, with fold-local preprocessing, cross-validation, and visual evaluation.`
- Topics：`machine-learning`、`fraud-detection`、`ensemble-learning`、`scikit-learn`、`classification`、`python`

## 上传步骤

1. 解压 `fraud-detection-ensemble.zip`。
2. 在 GitHub 新建仓库，填写上面的 name 和 description；如需公开展示，选择 Public。
3. 本地已经准备好 README 和 .gitignore，不需要再选模板生成重复文件。
4. 在空仓库页面选择上传已有文件，将解压目录内的文件及 data、results 文件夹上传到仓库根目录。
5. 不要只上传 ZIP；直接上传展开后的文件，GitHub 才能展示 README、Notebook 和图片。
6. 提交信息可填写：`Add fraud detection baselines and majority-voting ensemble`。
7. 上传后检查 README 图片能否显示、Notebook 是否能打开，然后在个人主页的仓库置顶区域选中这个仓库。

请上传打包好的项目文件夹，不要上传工作目录中的临时依赖、虚拟环境或课程 PDF。
本包不含原始 CSV；Notebook 在 Colab 中会提示上传 tranrecords.csv。
README 已说明课程来源、AI 辅助扩展、方法修正和结果局限。

## 这份导出包含什么

完整清洗与 EDA、金额对数变换、类别编码、相关性热力图、标准化、五种分类器、
五折交叉验证、三票通过的多数投票、指标表、模型对比图、混淆矩阵。
Notebook 附带运行结果，Python 文件包含同样的完整代码。

本地已运行：2500 行原始数据，清理后 2456 行，训练集 1964 行，测试集 492 行。
多数投票测试 F1 为 0.9048，随机森林为 0.9431。不要把这个项目描述成“集成必然优于单模型”。

## 方法上的调整

这是整理并修正后的项目版本，不是原教学 Notebook 的逐字复制。
主要修正：提前保留测试集；每折内独立填补和标准化；去掉 SGD 重复标准化；修正计时；固定随机种子。
因此它的数字可能与你原来的 Colab 输出不同。
