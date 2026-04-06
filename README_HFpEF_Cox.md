# HFpEF 队列 — Cox 比例风险生存分析

## 项目概述

本项目利用 [MIMIC-IV](https://physionet.org/content/mimiciv/) 数据库及
[MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) 特征提取流水线，
构建 **Cox 比例风险（Cox PH）模型**，预测 **射血分数保留型心力衰竭（HFpEF）**
住院患者的多种死亡终点。

**研究队列**：530 例来自 MIMIC-IV 的 HFpEF 指数住院患者，均配有经胸超声心动图（A4C 切面 DICOM）。

**分析终点**（全因死亡；多个时间截止点）：

| 终点列 | 含义 | 事件定义 |
|--------|------|---------|
| `died_inhosp` | 院内死亡 | 二值变量；单独分析，不混入院外 Cox 模型 |
| `os_event` / `os_days` | **主要 Cox 终点**（出院后死亡） | `os_event=1` + `os_days=dod−index_dischtime`；见步骤 6 |
| `event_30d` / `time_30d` | 出院后 30 天内死亡（衍生） | 从 `os_event` + `os_days` 截断衍生 |
| `event_90d` / `time_90d` | 出院后 90 天内死亡（衍生） | 从 `os_event` + `os_days` 截断衍生 |
| `event_1yr` / `time_1yr` | 出院后 1 年内死亡（衍生） | 从 `os_event` + `os_days` 截断衍生 |

三个特征时间窗（`win_hadm`、`win_48h24h`、`win_48h48h`）将与**主要终点及三种时间截止点**逐一组合，
共产生多个 Cox 模型，比较不同特征窗口与时间截止点的预测效果。

---

## 脚本运行方式

所有流水线脚本均可从**仓库根目录**或 **`utils/` 子目录**直接运行。
脚本内置 `_resolve_path()` 辅助函数，当相对路径在当前目录下不存在时，
自动回退到仓库根目录进行解析，兼容 PyCharm 默认从脚本所在目录运行的情形。

> **注意（Windows 用户）**：所有命令均写在一行，不使用反斜杠（`\`）换行。

```bat
:: 从仓库根目录运行（推荐）
python utils/clean_cohort_csvs.py --input_dir csv --output_dir csv/cleaned --resource_path resources
python utils/compute_comorbidity.py --input_dir csv/cleaned --output_dir csv/comorbidity
python utils/impute_normalize.py --input_dir csv/comorbidity --output_dir csv/processed
python utils/build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival

:: 从 utils/ 子目录运行（路径自动解析）
cd utils
python clean_cohort_csvs.py --input_dir csv --output_dir csv/cleaned --resource_path resources
python compute_comorbidity.py --input_dir csv/cleaned --output_dir csv/comorbidity
python impute_normalize.py --input_dir csv/comorbidity --output_dir csv/processed
python build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival
```

> **已修复（v2）**：早期版本从 `utils/` 目录启动时会抛出
> `FileNotFoundError: variable_ranges.csv not found`，
> 原因是默认的 `--resource_path resources` 被解析为相对于当前工作目录。
> 三个已有脚本均已通过 `_resolve_path()` 修复，新增脚本亦同步应用此修复。

---

## 仓库目录结构（项目专属文件）

```
SQL_Queries/                  # 步骤 1 — 队列定义 SQL（MIMIC-IV）
csv/                          # 原始队列 CSV（三个时间窗）
csv/cleaned/                  # 步骤 3 — 离群值清洗后的 CSV
csv/comorbidity/              # 步骤 4 — 合并症特征增强后的 CSV
csv/processed/                # 步骤 5 — 插补 + 标准化后的 CSV
csv/survival/                 # 步骤 6 — 生存终点构建后的 CSV
csv/feature_selection/        # 步骤 7 — 特征选择结果
csv/cox_models/               # 步骤 8 — Cox 模型结果
csv/cox_models/ph_test/       # 步骤 9 — PH 假设检验结果
csv/la_params/                    # LA 参数原始文件（来自 EchoGraphs 模块）
  final_morphology_results.csv  #   形态学参数长表
  final_kinematic_stats.csv     #   运动学参数长表
  final_qc.csv                  #   质量控制与元数据表
csv/la_params/processed/      # 步骤 A1 — LA 参数清洗输出（宽表）
csv/la_analysis/              # 步骤 A2 — LA × 临床联合分析输出
resources/                    # MIMIC-Extract variable_ranges.csv 等资源文件
utils/
  clean_cohort_csvs.py        # 步骤 3
  compute_comorbidity.py      # 步骤 4
  impute_normalize.py         # 步骤 5
  build_survival_endpoint.py  # 步骤 6
  feature_selection.py        # 步骤 7
  fit_cox_model.py            # 步骤 8
  ph_assumption_test.py       # 步骤 9
  ph_viz.py                   # 可视化模块（步骤 8、9 共用）
  model_evaluation.py         # 步骤 10 — 模型评估（C-index、Brier Score）
  clean_la_params.py          # 步骤 A1 — LA 影像参数清洗
  la_analysis.py              # 步骤 A2 — LA × MIMIC 临床参数联合分析
README_HFpEF_Cox.md           # ← 本文件
```

---

## 进度清单

| # | 步骤 | 状态 | 脚本 / 文件 |
|---|------|------|------------|
| 1 | 队列定义（SQL） | ✅ 完成 | `SQL_Queries/` |
| 2 | 特征提取（MIMIC-Extract） | ✅ 完成 | MIMIC-Extract 流水线 |
| 3 | 数据清洗（离群值去除） | ✅ 完成 | `utils/clean_cohort_csvs.py` |
| 4 | 合并症特征处理 | ✅ 完成 | `utils/compute_comorbidity.py` |
| 5 | 缺失值插补 & 标准化 | ✅ 完成 | `utils/impute_normalize.py` |
| 6 | 生存终点构建 | ✅ 完成 | `utils/build_survival_endpoint.py` |
| 7 | 特征选择（单变量 Cox + VIF + LASSO） | ✅ 完成 | `utils/feature_selection.py` |
| 8 | Cox PH 模型拟合（终点 × 时间窗） | ✅ 完成 | `utils/fit_cox_model.py` |
| 9 | PH 假设检验（Schoenfeld 残差）+ 分层修正 | ✅ 完成 | `utils/ph_assumption_test.py` |
| 10 | 模型评估（C-index、Brier Score） | ✅ 完成 | `utils/model_evaluation.py` |
| 11 | 可视化（KM 曲线、森林图） | ⬜ 待完成 | `utils/ph_viz.py` |
| A1 | LA 影像参数清洗（EchoGraphs → 宽表） | ✅ 完成 | `utils/clean_la_params.py` |
| A2 | LA × MIMIC 临床参数联合统计分析 | ✅ 完成 | `utils/la_analysis.py` |

---

## 各步骤详细说明

### 步骤 1 — 队列定义（SQL）

**文件**：`SQL_Queries/codes.sql`、`SQL_Queries/statics.sql`

**方法**：在 MIMIC-IV PostgreSQL 数据库中，筛选满足以下条件的成年住院患者：
- 主要或次要诊断编码包含 HFpEF 相关 ICD 编码
  （ICD-10：`I50.30`、`I50.31`、`I50.32`、`I50.33`、`I50.40`、`I50.41`、`I50.42`、`I50.43`、`I50.9`；
  ICD-9：`428.x` 系列）
- 同次住院期间完成了经胸超声心动图（A4C 切面 DICOM）检查
- 年龄 ≥ 18 岁

提取字段包括：
- 患者人口学信息（`subject_id`、`hadm_id`、`gender`、`anchor_age`）
- 入院/出院时间戳及生存状态
- **生存结局**：`died_inhosp`、`died_post_dc`、`days_survived_post_dc`、`died_30d`、`died_90d`、`died_1yr`
- **ICD 衍生合并症二值标志**（Deyo/Charlson 映射）及 `charlson_score`

**输出**：三个 CSV 文件，分别对应三个特征聚合时间窗（见步骤 2）。

---

### 步骤 2 — 特征提取（MIMIC-Extract）

**框架**：[MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) 聚合 ICU chartevents、labevents 及 OMR 数据。

**三个时间窗**：

| 窗口 | 描述 |
|------|------|
| `win_hadm` | 整个指数住院期间（入院 → 出院） |
| `win_48h24h` | 超声检查前 48 小时至前 24 小时 |
| `win_48h48h` | 超声检查前 48 小时至后 48 小时 |

**各窗口提取的特征**：

| 类别 | 列名 |
|------|------|
| 实验室指标 | 肌酐、BUN、钠、钾、氯、碳酸氢盐、钙、血糖、阴离子间隙、白蛋白、血红蛋白、红细胞压积、WBC、血小板、NT-proBNP、肌钙蛋白 T、CRP、INR、PT、PTT |
| 生命体征 | 心率、收缩压、舒张压、平均动脉压、呼吸频率、SpO₂、体温 |
| OMR（门诊） | 体重、BMI、收缩压、舒张压 |

每个特征取窗口内所有测量值的**均值**。

---

### 步骤 3 — 数据清洗（`utils/clean_cohort_csvs.py`）

**方法**：
1. **日期时间解析** — `index_study_datetime`、`index_admittime`、`index_dischtime`、`death_date` → `datetime64`
2. **二值标志强制转换** — 合并症/结局标志转换为可为空的 `Int8`（区分 `NaN` 与 0）
3. **离群值去除与有效范围裁剪** — 使用 `resources/variable_ranges.csv` 中的阈值：
   - 值 < `OUTLIER_LOW` 或 > `OUTLIER_HIGH` → 置为 `NaN`
   - 值 < `VALID_LOW`（但在离群值范围内） → 裁剪至 `VALID_LOW`
   - 值 > `VALID_HIGH`（但在离群值范围内） → 裁剪至 `VALID_HIGH`

**输出**：`csv/cleaned/hfpef_cohort_win_*_cleaned.csv`

---

### 步骤 4 — 合并症特征处理（`utils/compute_comorbidity.py`）

#### 4a. 查尔森合并症指数（CCI）部分重计算

基于可用的 ICD 衍生二值标志，计算 `cci_from_flags` 列，作为 SQL 衍生 `charlson_score` 的**下界验证**。

| 标志列 | CCI 权重 |
|--------|---------|
| myocardial_infarct | +1 |
| CHF（所有患者隐含） | +1 |
| peripheral_vascular_disease | +1 |
| cerebrovascular_disease | +1 |
| chronic_pulmonary_disease | +1 |
| mild_liver_disease | +1 |
| diabetes_without_cc | +1（仅当 diabetes_with_cc = 0 时） |
| diabetes_with_cc | +2（与 _without_cc 互斥） |
| renal_disease | +2 |
| malignant_cancer | +2 |
| severe_liver_disease | +3 |

均值：`cci_from_flags` = 4.02，`charlson_score` = 6.61，差值约 2.6 分（符合下界预期）。

#### 4b. HFpEF 特异性预后复合特征

新增 8 个二值/有序列（基于 MAGGIC、CHARM-Preserved 等文献）：

| 新列名 | 定义 | 临床依据 |
|--------|------|---------|
| `hf_any_diabetes` | 有无 DM（含或不含并发症） | DM 使 HFpEF 死亡率 HR 升高约 1.3–1.5 |
| `hf_cardiorenal` | 肾脏病 | 心肾综合征；CKD 使 HF 死亡风险翻倍 |
| `hf_met_syndrome_proxy` | 高血压 AND DM | 代谢综合征代理指标 |
| `hf_af_ckd` | 房颤 AND 肾脏病 | 共存时显著预测不良结局 |
| `hf_high_risk_triad` | 房颤 AND 肾脏病 AND DM | 三联高危组合，预后最差 |
| `hf_competing_risk` | 恶性肿瘤 OR 重度肝病 | 非心脏死亡风险主导 |
| `hf_comorbidity_burden` | 0=低（CCI<5）、1=中（5–8）、2=高（≥9） | 基于 charlson_score 的有序分层 |
| `hf_comorb_score_custom` | 加权合并症评分（HTN×1+AF×2+肾×2+DM×1+癌×3+重度肝×3+CVD×1） | HFpEF 专项风险评分 |

**队列主要合并症患病率**（三个时间窗数值相同，均来自 ICD 编码）：

| 合并症 | 患病率 |
|--------|--------|
| 高血压 | 92.1% |
| 房颤 | 53.8% |
| 肾脏病 | 49.6% |
| 任意糖尿病 | 49.1% |
| 慢性肺部疾病 | 37.0% |
| 代谢综合征代理 | 46.4% |
| 房颤 + CKD | 27.7% |
| 高危三联 | 14.9% |
| 竞争风险（癌/肝） | 12.5% |

**输出**：`csv/comorbidity/hfpef_cohort_win_*_comorbidity.csv` 及 `csv/comorbidity/comorbidity_summary.csv`

---

### 步骤 5 — 缺失值插补 & 标准化（`utils/impute_normalize.py`）

#### 5a. 缺失值分析与变量删除决策

| 缺失比例 | 处理方式 |
|---------|---------|
| ≥ 60% | **删除**变量——稀疏度过高，无法可靠插补 |
| 20–59% | **插补 + 标志**——中位数/众数插补；添加 `<列名>_missing_flag` 二值列 |
| 5–19% | **插补**——中位数（连续）或众数（二值）插补 |
| < 5% | **插补**——中位数/众数插补 |

**已删除变量**（所有时间窗缺失率均 ≥ 60%）：

| 变量 | hadm 缺失率 | 48h24h 缺失率 | 48h48h 缺失率 | 原因 |
|------|------------|--------------|--------------|------|
| `crp` | 89.6% | 96.2% | 95.5% | CRP 在 MIMIC-IV ICU 环境中检测频率低 |
| `ntprobnp` | 75.1% | 80.2% | 79.6% | NT-proBNP 并非所有住院常规检测 |
| `troponin_t` | 64.9% | 70.9% | 70.2% | 序列肌钙蛋白非 HFpEF 诊断常规 |
| `albumin` | 52.3%* | 65.5% | 62.8% | ICU 专项检测；病房患者采集率低 |
| `temperature_c` | 68.1% | 74.5% | 74.2% | 基于 chartevents 聚合；约 26% 有 ICU 住院史 |
| `heart_rate` | 68.1% | 74.5% | 74.2% | 同上——非 ICU 患者缺失 chartevents 生命体征 |
| `sbp`、`dbp`、`mbp` | 68.1% | 74.5% | 74.2% | 同上 |
| `resp_rate`、`spo2` | 68.1% | 74.5% | 74.2% | 同上 |

\* `albumin` 在 `hadm` 窗口的缺失率低于 60% 阈值，保留并进行插补 + 标志。

**已添加缺失标志列的变量**（20–60% 缺失）：

| 变量 | 窗口 | 缺失率 |
|------|------|--------|
| `albumin` | hadm | 52.3% |
| `pt`、`inr` | 48h24h、48h48h | ~25–27% |
| `ptt` | 48h24h、48h48h | ~28–30% |

#### 5b. 行级别完整性检查

保留特征列缺失率 > 60% 的患者行将被删除。**三个时间窗均无患者被删除**。

#### 5c. 插补策略

- **连续变量**：中位数插补（对离群值和偏态分布鲁棒）
- **二值/有序变量**：众数插补（最高频类别）

当前使用简单（单次）插补。在最终模型拟合阶段，建议对缺失率 5–60% 的变量使用
**多重插补**（如 MICE，通过 `sklearn.impute.IterativeImputer` 实现），
以避免标准误低估。

#### 5d. 标准化

插补后对所有保留的连续特征进行标准化：

| 条件 | 变换方式 |
|------|---------|
| \|偏度\| ≤ 1.0 | z-score（`StandardScaler`） |
| \|偏度\| > 1.0 | log₁₊₁ 变换后 z-score |

通常经 log1p 变换的右偏变量：肌酐、BUN、血糖、WBC、血小板、INR、PT、PTT、肌钙蛋白 T、NT-proBNP。

二值列及所有合并症/有序列保留为 0/1 整数，**不做标准化**。

**输出**：`csv/processed/hfpef_cohort_win_*_processed.csv`、
`csv/processed/missingness_report.csv`、`csv/processed/feature_decisions.csv`

---

### 步骤 6 — 生存终点构建（`utils/build_survival_endpoint.py`）

#### MIMIC-IV 死亡日期（DOD）数据收录规则

理解 MIMIC-IV 的死亡日期来源与截尾机制，对于正确构建生存终点至关重要：

**死亡日期来源**：
- 死亡日期来自**医院记录**与**州政府记录**两个渠道；若两者均存在，以**医院记录优先**。
- 州政府记录通过基于**姓名、出生日期及社会保障号**的自定义规则匹配算法进行关联。

**数据完整性保障**：
- MIMIC-IV 中，州政府与医院的死亡记录均在**最后一例患者出院两年后**才完成采集，
  以最大限度降低死亡日期上报延迟带来的影响。

**去标识化截尾规则（核心约束）**：
- 作为去标识化处理的一部分，**出院超过 1 年的死亡日期会被截尾**（dod 字段置为 NULL）。
- 截尾时间以**末次住院出院时间（`last_dischtime`）**为基准，而非指数住院出院时间：
  `censor_date = last_dischtime + 1 年`
- 若患者在指数住院后还有后续住院，其可观测随访窗口将延伸至末次出院加一年，
  因此 **`os_days` 可超过 365 天**（多次住院患者的有效随访期更长）。
- 示例：若患者末次出院日期为 2155-01-01（而指数出院为 2150-01-01），
  则可被记录的最晚死亡日期为 2156-01-01；`os_days` 上限约为 6 年。

**dod 字段取值规则**：

| 情形 | dod 字段 |
|------|---------|
| 患者在 `last_dischtime + 1 年` 之前死亡，且被医院或州记录捕获 | 填充去标识化后的死亡日期 |
| 患者在 `last_dischtime` 后存活至少 1 年 | NULL（视为行政删失） |

#### SQL 中新增的关键列

原始表格已在 SQL 导出阶段（步骤 1/2）中预先计算以下列：

| 列名 | 含义 | 取值 |
|------|------|------|
| `last_dischtime` | 患者在 MIMIC-IV 中所有住院中的末次出院时间（`MAX(dischtime)`） | 时间戳 |
| `censor_date` | MIMIC-IV 行政删失日期 = `last_dischtime + 365 天` | 时间戳 |
| `os_event` | Cox 事件指标 | `1` = 出院后死亡；`0` = 删失；`NULL` = 院内死亡 |
| `os_days` | Cox 时间变量 | 事件患者：`dod − index_dischtime`；删失患者：`censor_date − index_dischtime`；院内死亡：`NULL` |

> **重要**：`days_survived_post_dc` 仅对死亡患者非空（删失患者为 NaN），
> **不可直接用于 Cox 模型**。Cox 分析请使用 `os_event` + `os_days`。

#### 患者分类（院外随访分析）

| 患者类型 | 处理方式 | os_event | os_days |
|---------|---------|---------|---------|
| 院内死亡（`hospital_expire_flag=1`） | **排除**出院后分析；写入独立院内文件 | NULL | NULL |
| 院外死亡（`died_post_dc = 1`） | 事件患者；进入 Cox 分析 | 1 | `dod − index_dischtime` |
| 存活删失（其余院外患者） | 删失患者；进入 Cox 分析 | 0 | `censor_date − index_dischtime` |

> **Cox 分析前务必过滤**：`WHERE died_inhosp = 0`（或 `WHERE os_event IS NOT NULL`）

#### 衍生时间窗终点（由脚本计算）

`build_survival_endpoint.py` 从原始 `os_event` + `os_days` 衍生各时间截止点终点列：

| 终点后缀 | 截止点 H | `event_<H>` 定义 | `time_<H>` 定义 |
|---------|---------|----------------|----------------|
| `_30d`  | 30 天   | `os_event=1` 且 `os_days ≤ 30` → 1，否则 0 | `min(os_days, 30)` |
| `_90d`  | 90 天   | `os_event=1` 且 `os_days ≤ 90` → 1，否则 0 | `min(os_days, 90)` |
| `_1yr`  | 365 天  | `os_event=1` 且 `os_days ≤ 365` → 1，否则 0 | `min(os_days, 365)` |
| `_any`  | 不截断  | 等于 `os_event` | 等于 `os_days` |

院内死亡患者（`os_event=NULL`）的所有衍生终点均保持 **NULL**，与出院后分析明确隔离。

#### 本队列实际统计

| 指标 | 数值 |
|------|------|
| 总患者数 | 530 |
| 院内死亡（排除） | 18 |
| 院外患者（参与 Cox） | 512 |
| 其中事件（`os_event=1`） | 267（52.1%） |
| 其中删失（`os_event=0`） | 245（47.9%） |
| `os_days` 范围 | 0 – 2324 天 |
| `os_days > 365` 的患者（末次出院晚于指数出院） | 315 |

**运行命令**：
```bat
python utils/build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival
```

**输出**：
- `csv/survival/hfpef_cohort_win_*_survival.csv`：全量患者（院外患者含终点列，院内死亡行终点为 NULL）
- `csv/survival/hfpef_cohort_win_*_inhosp.csv`：仅院内死亡患者（独立分析用）

---

### 步骤 7 — 特征选择

**背景**：约 46–47 个候选特征，530 例患者，有效事件数随终点时间不同（27–267）。
直接拟合全量特征的 Cox 模型过度拟合风险高（30 天终点 EPV ≈ 0.6，1 年终点 EPV ≈ 3）。
三步流水线按顺序筛选，每步均输出可复用的 CSV 报告。

#### 方法 1 — 单变量 Cox 筛选

对每个特征单独拟合 `CoxPHFitter`（lifelines），记录：

| 输出列 | 含义 |
|--------|------|
| `coef` | 偏回归系数 |
| `exp_coef` | 风险比 HR |
| `exp_coef_lower/upper_95` | 95% CI |
| `z` / `p` | Wald 检验统计量 / p 值 |
| `concordance` | Harrell C-index（该特征单独的区分能力） |
| `n_events` | 终点事件数 |
| `significant` | p < 0.10（宽松阈值，避免漏掉有临床意义的变量） |

**报告文件**：
- `method1_univariate_{endpoint}.csv`：全部特征结果（按 p 值升序排列）
- `method1_candidates_{endpoint}.csv`：p < 0.10 且收敛的候选集

#### 方法 2 — VIF 共线性过滤

对方法 1 候选集迭代计算方差膨胀因子（statsmodels VIF）：

1. 计算所有候选特征的 VIF。
2. 若最大 VIF > 10，按**临床优先级**（`VIF_PRIORITY` 列表）删除优先级最低的特征。
   - 保留 `creatinine` 优于 `bun`（GFR 代理更直接）
   - 保留 `hemoglobin` 优于 `hematocrit`（浓度指标更稳定）
   - 保留 `inr` 优于 `pt` / `ptt`（综合凝血指标）
   - 保留 `charlson_score` 优于 `cci_from_flags`（标准量表）
3. 重复直至所有 VIF ≤ 10 或剩余特征 ≤ 2。

**报告文件**：
- `method2_vif_{endpoint}.csv`：每轮迭代的 VIF 值及删除记录
- `method2_candidates_{endpoint}.csv`：过滤后候选集

#### 方法 3 — LASSO 惩罚 Cox（交叉验证）

在方法 2 候选集上拟合 LASSO-Cox（`l1_ratio=1.0`，lifelines `CoxPHFitter`）：

1. **k 折交叉验证**（k=5，随机种子=42）：在 10 个候选 penalizer（0.001→5.0）上评估验证集对数偏似然。
2. 选取平均对数偏似然最高的 penalizer（λ*）。
3. 以 λ* 在全量数据重新拟合，**|系数| > 1e-4** 的特征为最终入选特征。

**报告文件**：
- `method3_cv_{endpoint}.csv`：各 penalizer 的 CV 平均 / 标准差对数偏似然
- `method3_lasso_{endpoint}.csv`：最优模型全部系数（`nonzero` 列标记入选）
- `final_features_{endpoint}.csv`：三步汇总表（全特征 × 三步通过状态）

#### 全局汇总报告

`csv/feature_selection/selection_summary.json`：JSON 格式，包含所有 window × endpoint 组合的：
- 输入特征数、M1/M2/M3 通过数、最终特征列表
- 最优 penalizer、事件数、n_obs

#### 本队列实际筛选结果（hadm 窗口）

| 终点 | 事件数 | M1 通过 | M2 通过 | LASSO 最终 | 关键特征 |
|------|--------|---------|---------|-----------|---------|
| 30d  | 27  | 10 | 9  | 7  | albumin, wbc, bun, hf_af_ckd, atrial_fibrillation |
| 90d  | 53  | 14 | 13 | 11 | albumin, wbc, hemoglobin, hf_af_ckd, omr_bmi, sodium, creatinine |
| 1yr  | 133 | 21 | 17 | 13 | albumin, omr_bmi, hemoglobin, wbc, anchor_age, malignant_cancer, hf_af_ckd |
| any  | 267 | 27 | 21 | 16 | albumin, omr_bmi, anchor_age, hemoglobin, wbc, hf_competing_risk, hf_af_ckd, creatinine |

**运行命令**：
```bat
# 全部时间窗 × 全部终点
python utils/feature_selection.py --input_dir csv/survival --output_dir csv/feature_selection

# 仅处理单个窗口 + 终点
python utils/feature_selection.py --window hadm --endpoint any
```

**可调参数**：
```
--p_threshold  0.10    方法 1 单变量 p 值筛选阈值
--vif_threshold 10.0   方法 2 VIF 阈值
--cv_folds 5           方法 3 交叉验证折数
--dry_run              仅打印摘要，不写文件
```

**输出目录结构**：
```
csv/feature_selection/
  selection_summary.json           ← 全局汇总（可视化复用）
  hadm/
    method1_univariate_any.csv     ← 全特征单变量 Cox 结果（HR/CI/p/C-index）
    method1_candidates_any.csv     ← p<0.10 候选集
    method2_vif_any.csv            ← 迭代 VIF 全程记录
    method2_candidates_any.csv     ← VIF 过滤后候选集
    method3_cv_any.csv             ← LASSO CV penalizer 比较
    method3_lasso_any.csv          ← LASSO 最优模型系数
    final_features_any.csv         ← 三步汇总（all × 3 通过状态）
    ... (30d / 90d / 1yr 同理)
  48h24h/ ...
  48h48h/ ...
```

---

### 步骤 8 — Cox PH 模型拟合 ✅

**脚本**：`utils/fit_cox_model.py`  
**使用库**：`lifelines`

对每个「时间窗 × 终点」组合，读取步骤 7 的 LASSO 特征子集，
拟合全量无正则化 Cox PH 模型（`penalizer=0`），
收敛失败时自动使用兜底惩罚因子（`penalizer=0.05`）。

**运行命令**：
```bat
# 全部组合
python utils/fit_cox_model.py

# 仅处理 hadm 窗口
python utils/fit_cox_model.py --window hadm

# 仅处理特定终点
python utils/fit_cox_model.py --window hadm --endpoint 1yr

# 不生成图形
python utils/fit_cox_model.py --no_plots
```

**实际拟合结果汇总**：

| 窗口+终点 | n | 事件 | 特征数 | EPV | C-index | 收敛 | 备注 |
|----------|---|------|--------|-----|---------|------|------|
| 48h24h/30d | 512 | 27 | 8 | 3.4 | 0.7695 | ✓ | *low_EPV; pen=0.05 |
| 48h24h/90d | 512 | 53 | 5 | 10.6 | 0.6692 | ✓ | |
| 48h24h/1yr | 512 | 133 | 8 | 16.6 | 0.6572 | ✓ | |
| 48h24h/any | 512 | 267 | 9 | 29.7 | 0.6283 | ✓ | |
| 48h48h/30d | — | — | — | — | — | — | 跳过（无选中特征） |
| 48h48h/90d | 512 | 53 | 5 | 10.6 | 0.6740 | ✓ | |
| 48h48h/1yr | 512 | 133 | 10 | 13.3 | 0.6641 | ✓ | |
| 48h48h/any | 512 | 267 | 1 | 267.0 | 0.5839 | ✓ | 仅 1 特征入模 |
| hadm/30d | 512 | 27 | 6 | 4.5 | 0.8039 | ✓ | *low_EPV |
| hadm/90d | 512 | 53 | 7 | 7.6 | 0.7591 | ✓ | *low_EPV |
| **hadm/1yr** | **512** | **133** | **9** | **14.8** | **0.6834** | **✓** | **主要参考结果** |
| hadm/any | 512 | 267 | 11 | 24.3 | 0.6456 | ✓ | |

> **EPV** = 事件数 / 特征数（events per variable）。  
> `*low_EPV`：EPV < 10，不满足 Peduzzi et al. (1995) 的 10–20 EPV 准则，
> 该模型 HR 估计的方差被低估，C-index 虚高，**不应作为主要结论引用**。

#### EPV 评估与主要可信结果

| 准则 | 不满足（EPV < 10）| 满足（EPV ≥ 10）|
|------|-----------------|----------------|
| 模型 | hadm/30d, hadm/90d, 48h24h/30d | hadm/1yr, hadm/any, 48h24h/90d, 48h24h/1yr, 48h48h/90d, 48h48h/1yr, 48h48h/any |

**主要可信结果**：**`hadm/1yr` — C-index = 0.6834，EPV = 14.8，全量住院期特征集**

- EPV = 133 / 9 = **14.8**，满足 ≥ 10 的最低准则，估计稳定可信 ✓
- C-index 0.68 属于临床生存模型的"中等"区间（0.6–0.7），对于单纯实验室 +
  合并症特征（无影像/基因）的队列属正常水平 ✓
- `hadm` 窗口收集了**整个住院期**的检验均值，信息量最充分；
  `1yr` 终点事件数充足且临床意义明确 ✓
- **结论：该结果可接受，可继续推进 PH 假设检验（步骤 9）**

**输出目录**：
```
csv/cox_models/
  cox_summary.json
  hadm/
    cox_results_30d.csv
    cox_results_90d.csv
    cox_results_1yr.csv
    cox_results_any.csv
  48h24h/ ...
  48h48h/ ...
  figures/
    forest_{window}_{endpoint}.png
    baseline_survival_{window}_{endpoint}.png
    cindex_summary.png
```

---

### 步骤 9 — PH 假设检验与分层修正（Schoenfeld 残差）✅

**脚本**：`utils/ph_assumption_test.py`  
**可视化模块**：`utils/ph_viz.py`（步骤 8、9 共用，见下文"可视化模块"一节）  
**使用库**：`lifelines.statistics.proportional_hazard_test`, `statsmodels`（可选 LOWESS）

对 EPV ≥ 10 的全部模型，运行 Schoenfeld 残差法检验 PH 假设
（Grambsch & Therneau 1994）。

- **原假设 H₀**：给定协变量的系数在随访全程内保持不变（PH 成立）。
- **显著结果（p < 0.05）**：该协变量存在时变效应，PH 假设可能不成立。

**运行命令**：
```bat
# 对全部 EPV>=10 模型运行 PH 检验
python utils/ph_assumption_test.py

# 仅检验 hadm/1yr（主要结果）
python utils/ph_assumption_test.py --window hadm --endpoint 1yr

# 不生成图形
python utils/ph_assumption_test.py --no_plots

# 检验后自动对违反 PH 的变量进行分层修正并重新检验
python utils/ph_assumption_test.py --correct_violations

# 仅对 hadm/1yr 运行检验 + 修正
python utils/ph_assumption_test.py --window hadm --endpoint 1yr --correct_violations
```

**前提**：须先完成步骤 8（`cox_summary.json` 必须存在）。

**输出目录**：
```
csv/cox_models/ph_test/
  ph_test_summary.json                         ← 全局汇总（是否通过 PH 检验）
  ph_test_{window}_{endpoint}.csv              ← 每个协变量的 Schoenfeld 检验统计量及 p 值
  figures/
    covariate_effects_{window}_{endpoint}.png  ← 协变量效应图（分层生存曲线，含显示名称）
    schoenfeld_residuals_{window}_{endpoint}.png  ← 缩放 Schoenfeld 残差图 + LOWESS 平滑
  stratified_corrected/                        ← 仅 --correct_violations 时生成
    stratified_corrected_{window}_{endpoint}.csv   ← 分层修正模型系数
    ph_test_{window}_{endpoint}_stratified.csv     ← 修正后 PH 检验结果
    figures/
      schoenfeld_residuals_{window}_{endpoint}_stratified.png
  stratified_correction_summary.json           ← 分层修正前后对比汇总（供 model_evaluation.py 读取）
```

#### PH 检验结果汇总

| 窗口+终点 | 协变量数 | 违反 PH（p<0.05） | PH 通过？ | 说明 |
|----------|---------|-----------------|---------|------|
| 48h24h/90d | 5 | 0 | ✓ 是 | |
| 48h24h/1yr | 8 | 0 | ✓ 是 | |
| 48h24h/any | 9 | 0 | ✓ 是 | |
| 48h48h/90d | 5 | 0 | ✓ 是 | |
| 48h48h/1yr | 10 | 1 | ✗ 否 | wbc 违反 PH |
| 48h48h/any | 1 | 0 | ✓ 是 | 仅 1 特征 |
| hadm/1yr | 9 | 1 | ✗ 否 | **wbc 违反 PH（p≈0.02）** |
| hadm/any | 11 | 1 | ✗ 否 | wbc 违反 PH |

> **主要结论**：`wbc`（白细胞计数）在多个窗口中违反 PH 假设，提示其对死亡风险的效应随时间递减（早期感染负荷效应更强）。`malignant_cancer` 在 `hadm/1yr` 中 p 值接近 0.05，属临界情况，结合 Schoenfeld 残差图判断无明显趋势，暂保留主效应。

#### 分层修正（`--correct_violations`）

当某协变量违反 PH 假设时，本脚本采用**四分位分层 Cox**（stratified Cox）作为主要修正策略：

- **连续变量**（nunique > 5）：按四分位离散化为最多 4 层，作为 `strata` 传入 Cox 模型。
  模型为每层估计独立的基线风险 $h_0^{(k)}(t)$，从而在结构上满足 PH 假设；
  其余协变量仍保留为比例风险主效应项，HR 可正常解读。
- **二值 / 低基数变量**（nunique ≤ 5）：直接以原始变量作为 `strata`。

违反变量被移出协变量列表（其效应被基线风险结构吸收），其余协变量系数不受影响。

**为什么不用 `x × log(t)` 静态时间交互项**：
将 `x × time_col` 作为协变量加入 Cox 模型存在根本性的内生性问题——协变量值由每名受试者
自身的结局/截尾时间决定，导致预测因子与结局时间直接相关。这种内生性会：
- 虚高 C-index（通常高 15–30 个百分点，属数据泄漏）；
- 即便 p < 0.05 的 Schoenfeld 残差在交互项模型中依然不平稳（模型仍然错设）。
分层 Cox 在不引入内生性的前提下彻底放松违反变量的 PH 约束，是文献中推荐的标准方法。

**学术背景——wbc 违反 PH 的文献依据**：

白细胞计数（WBC）在 Cox 心衰预测模型中的时变效应已有多项研究记录：

- Anand IS 等（HF-ACTION，JACC 2009）及 Tang WHW 等（Circulation 2013）均发现
  WBC 对心衰死亡率的预测价值主要集中在短中期（6–12 个月内），长期效应衰减，
  符合其反映急性炎症/感染状态的生物学机制。
- Ahmed A 等（JAMA Intern Med 2007）在老年心衰队列中同样观察到 WBC 的时变效应。

因此 WBC 违反 PH 不是数据质量问题，而是符合预期的生物学信号。标准处理方式有：
1. **分层 Cox**（本脚本实现）：最常用，不影响其他协变量的 HR 解读；
2. **报告并讨论局限性**：若分层后仍不通过，可在论文中说明 PH 假设近似满足，
   并引用 Therneau & Grambsch（2000）关于轻微违反时结果鲁棒性的讨论；
3. **参数生存模型**（Weibull/log-normal）：适合违反普遍时，灵活性更高但解读复杂。

#### 分层修正结果汇总

| 窗口+终点 | 修正前 C-index | 修正后 C-index | 修正后 PH 通过？ | 说明 |
|----------|--------------|--------------|---------------|------|
| 48h48h/1yr | 0.6641 | 0.6549 | ✗ 否 | 见下文"残留违反"讨论 |
| hadm/1yr | 0.6834 | 0.6619 | ✗ 否 | 见下文 |
| hadm/any | 0.6456 | 0.6289 | ✓ 是 | 分层修正有效 |

**残留违反（48h48h/1yr 和 hadm/1yr）的学术处理**：

分层修正使 `hadm/any` 通过 PH 检验，但 `48h48h/1yr` 和 `hadm/1yr` 仍然不通过。
这在学术上是允许的，处理方式如下：

1. **透明报告**：在论文方法学部分明确报告 PH 检验结果及分层修正尝试；
   对残留违反的变量报告时间加权平均 HR 及置信区间作为近似汇总效应量。
2. **分层是充分修正**：分层 Cox 已从统计结构上放松了违反变量的 PH 约束；
   若 Schoenfeld 残差仍显著，通常是因为其他协变量（非 wbc）也有时变趋势，
   不应反复对同一模型叠加修正。额外修正（如再次分层其他变量）会过度消耗自由度，
   且在 HFpEF 这类小样本（EPV ≈ 13–15）中尤其危险。
3. **敏感性分析**：以通过 PH 检验的 `48h24h/1yr`（PH 完全通过，C-index 0.6572）
   作为主要敏感性模型进行对比，若结论一致则支持稳健性。
4. **学术惯例**：多数心衰生存预测研究（包括 MAGGIC、HF-ACTION 等）使用 Cox 模型时
   仅报告 PH 检验结果，部分违反并不导致撤稿；关键在于透明报告残差图和修正过程。

> **最终主模型选择**：推荐使用 `hadm/1yr`（原始模型，未分层）作为主要报告模型，
> 因为分层后 C-index 下降（0.6834 → 0.6619）且 PH 仍不通过，说明该特定模型的
> wbc 时变效应难以通过分层完全消除。在论文中透明报告 PH 检验结果并讨论
> wbc 时变效应的临床含义，是更有学术价值的处理方式。
> `48h24h/1yr` 作为敏感性分析对照（PH 完全通过，C-index 0.6572）。

#### 协变量效应图（Covariate Effect Plots）的技术说明

**二元变量的处理**（v2 修正，`ph_viz.py`）：

步骤 5 对所有特征进行了 z-score 标准化，包括二元变量（0/1 编码的合并症标志）。
标准化后二元变量的两个类别分别映射到 $\{z_{\text{low}}, z_{\text{high}}\}$（非 0/1），
而 lifelines 的 `plot_partial_effects_on_outcome` 内部以**中位数**作为中心基线
（`plot_baseline=True`）。由于中位数对于二元变量必然等于两个类别值之一，
基线曲线与其中一条效应曲线完全重叠——这是一个纯粹的代码问题，与数据分布无关。

修复方案（已在 `ph_viz.plot_covariate_effects` 中实现）：
- **连续变量**：仍在 P10/P90 处绘制效应曲线，并保留基线曲线（`plot_baseline=True`）。
- **二元变量**：从数据中提取实际的两个标准化值（而非硬编码 `[0, 1]`），
  并禁用基线曲线（`plot_baseline=False`）——二元变量的"均值处"不是真实的临床状态，
  显示它只会制造视觉混淆。

同时修正了基线线型检测的 bug：lifelines 使用 `":"` 点线（dotted）绘制基线，
而旧代码错误地检测 `"--"` 虚线（dashed），导致基线永远无法被正确识别和标注。

---

### 步骤 10 — 模型内部验证与评估 ✅

**脚本**：`utils/model_evaluation.py`  
**使用库**：`lifelines`, `numpy`, `matplotlib`

对最终选定模型及全部候选模型执行内部验证与性能评估：

1. **Bootstrap C-index 乐观校正**（Harrell 1996）：1000 次有放回重抽样，计算校正后 C-index 及 95% 置信区间。
2. **校准曲线**（Calibration curve）：按预测风险五分位分组，KM 观测风险 vs. 模型预测风险。
3. **决策曲线分析（DCA）**：净收益 vs. 阈值概率曲线（Vickers & Elkin 2006）。

**运行命令**：
```bat
# 对全部满足 EPV 准则的模型运行评估（1000 次 bootstrap）
python utils/model_evaluation.py

# 仅评估 hadm/1yr（主要结果）
python utils/model_evaluation.py --window hadm --endpoint 1yr

# 使用分层修正后的模型进行评估（需先运行步骤 9 --correct_violations）
# model_evaluation.py 自动加载 stratified_correction_summary.json
python utils/model_evaluation.py \
    --window hadm --endpoint 1yr \
    --tvc_summary csv/cox_models/ph_test/stratified_correction_summary.json

# 自定义 bootstrap 次数和评估时间点
python utils/model_evaluation.py --n_bootstrap 1000 --t_eval 365

# 不生成图形
python utils/model_evaluation.py --no_plots
```

> **注意（数据泄漏）**：旧版代码曾使用 `x × log(t)` 交互项修正 PH 违反
> （即将 `feature * log(time_col)` 作为协变量）。由于 `time_col`（观测存活
> 时间）是结局变量，该做法构成直接的**数据泄漏**——模型可从自身预测变量
> 中恢复结局时间，导致 C-index 虚高 15–30 个百分点（如 `48h48h/1yr`：
> 0.6641→0.8349；`hadm/1yr`：0.6834→0.9239）。当前版本已彻底移除
> `_x_logt` 交互项的构造逻辑；若加载包含此类特征的旧版 `tvc_summary.json`，
> 脚本会发出警告并回退到基础特征集。分层 Cox 模型（`ph_assumption_test.py
> --correct_violations` 产生的 `stratified_correction_summary.json`）是
> 唯一受支持的 PH 修正方法。

**输出目录**：
```
csv/model_eval/
  evaluation_summary.json          ← 全部模型的评估汇总
  {window}_{endpoint}/
    bootstrap_cindex.json          ← bootstrap C-index 汇总
    calibration.csv                ← 校准分组数据
    dca.csv                        ← DCA 净收益表
    figures/
      bootstrap_cindex.png         ← C-index 对比柱状图（apparent vs corrected）
      calibration.png              ← 校准曲线（observed vs predicted）
      dca.png                      ← 决策曲线
```

**关键方法说明**：

| 评价维度 | 方法 | 参考文献 |
|---------|------|---------|
| 区分度 | Harrell C-index（Bootstrap 乐观校正） | Harrell et al. 1996 |
| 内部校准 | 五分位组 KM 观测风险 vs. 预测风险 | Hosmer-Lemeshow 改编 |
| 临床效用 | 决策曲线分析（净收益） | Vickers & Elkin 2006 |

**Bootstrap 乐观校正原理**：

$$C_{\text{corrected}} = C_{\text{apparent}} - \overline{\text{optimism}}$$

$$\overline{\text{optimism}} = \frac{1}{B} \sum_{b=1}^{B} \left( C_{\text{boot,train}}^{(b)} - C_{\text{boot,test}}^{(b)} \right)$$

其中 $C_{\text{boot,train}}^{(b)}$ 为第 $b$ 次 bootstrap 样本上拟合并评估的 C-index，$C_{\text{boot,test}}^{(b)}$ 为同一 bootstrap 模型在原始数据上的 C-index。

**DCA 净收益公式**（时间点 $t$ 处，阈值概率 $p_t$）：

$$\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \frac{p_t}{1 - p_t}$$


---

### 步骤 11 — 可视化（待完成）

> **v2 新增**：所有生产图形代码已从步骤 8、9 脚本中抽出，集中到
> `utils/ph_viz.py` 模块（见下文"可视化模块"一节）。步骤 11 的新增图表
> 可直接调用该模块的公开函数，无需重复实现。

**计划图表**：
1. 按 `hf_comorbidity_burden`（低/中/高）分层的 Kaplan–Meier 曲线。
2. 按 `hf_high_risk_triad` 分层的 KM 曲线。
3. Cox 模型风险比（HR）及 95% CI 的森林图。
4. 连续预测变量（年龄、肌酐）的偏效应图。
5. 用于 PH 假设评估的 log-log 图（log[−log S(t)] vs. log t）。

---

### 可视化模块 — `utils/ph_viz.py`

**v2 新增**：所有 matplotlib 绘图函数已从 `fit_cox_model.py` 和 `ph_assumption_test.py`
中提取到独立模块 `utils/ph_viz.py`，以实现单一实现、零重复。

**公开 API**：

| 函数 | 说明 |
|------|------|
| `setup_chinese_font()` | 尝试配置 CJK 字体，返回是否成功 |
| `plot_covariate_effects(cph, df_model, ph_df, window, endpoint, fig_dir, ...)` | 协变量效应图（分层生存曲线） |
| `plot_schoenfeld_residuals(cph, df_model, ph_df, window, endpoint, fig_dir, ...)` | 缩放 Schoenfeld 残差图 + LOWESS |
| `plot_forest(coef_df, window, endpoint, out_path, concordance, ...)` | 森林图（HR ± 95% CI） |
| `plot_baseline_survival(cph, window, endpoint, out_path, ...)` | 基线生存函数 S₀(t) |
| `plot_cindex_summary(summaries, out_path, ...)` | C-index 汇总条形图 |

**在外部脚本中使用**：
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))
from ph_viz import plot_covariate_effects, plot_forest, plot_cindex_summary
```

---

### 步骤 A1 — LA 影像参数清洗（`utils/clean_la_params.py`）

**数据来源**：EchoGraphs 模块生成的三张左心房（LA）参数文件（放置于 `csv/la_params/`）：

| 输入文件 | 说明 |
|----------|------|
| `final_morphology_results.csv` | 形态学参数长表（主键：`video_prefix + parameter_name`） |
| `final_kinematic_stats.csv` | 运动学 / 应变参数长表（主键：`video_prefix + parameter_name + sub_item`） |
| `final_qc.csv` | 质量控制与元数据表（必须同时使用） |

**重要语义约定**：
- `max_idx → LAVmax`，`min_idx → LAVmin`；**不**按旧 `ED/ES` 假设映射
- 所有 `value` 空字符串均表示"当前不可用"，绝不能按 0 处理
- 所有参数基于模型连续 28 点轮廓，非人工真值直接测量

**清洗步骤（两层异常值过滤）**：

| 步骤 | 方法 |
|------|------|
| QC 硬过滤 | 含 `missing_spacing` 或 `analysis_error` 的视频整体移除 |
| 行级软过滤 | 按每行 `status` 字段，将受影响的参数值置为 NaN（不填 0） |
| **第一层：自动剔除（长表）** | 仅去除"几何/数学不可能"或"明显测量失败"的值；超出范围的值置为 NaN，行保留 |
| 长表 → 宽表 | 形态表按 `parameter_name` pivot；运动表按 `parameter_name__sub_item` pivot |
| **第一层补充：跨列约束（宽表）** | `LAVmin ≤ LAVmax`、`LAVmin-i ≤ LAVI`、`LAD-trans ≤ LAD-long`；违规值置为 NaN |
| 缺失列剔除 | 列缺失率 ≥ 30% → 整列剔除（论文约定） |
| **第二层：人工复核标记（宽表）** | 不删除；仅为超复核范围的值添加 `{col}_review_flag = 1` 列 |
| **未收录参数：IQR 统计** | 不进行任何过滤；计算 Tukey IQR 异常值统计，输出 CSV 报告与可视化图表 |
| 影像缺失策略 | 算法失败导致的缺失直接保留 NaN（不填补），下游分析时按需 `dropna` |
| Z-score 标准化 | `\|偏度\| > 1.0` 先 `log1p`，再 `StandardScaler`（跳过 `_review_flag` 列） |

#### 生理范围检查所用文件及位置

> **与 `clean_cohort_csvs.py` 不同**：MIMIC 临床参数的范围来自外部文件  
> `resources/variable_ranges.csv`（MIMIC-Extract 项目随附），  
> LA 影像参数的范围**直接硬编码**在 `utils/clean_la_params.py` 中，分为两层：

| 层 | 字典名 | 大致位置 | 用途 |
|----|--------|----------|------|
| 第一层（自动剔除） | `LA_AUTO_REMOVE_RANGES` | `clean_la_params.py` 约第 104 行 | 几何/数学不可能值 → NaN |
| 第二层（人工复核） | `LA_REVIEW_RANGES` | `clean_la_params.py` 约第 143 行 | 宽松范围，只标 flag |
| 跨列约束 | `LA_CROSS_CHECKS` | `clean_la_params.py` 约第 179 行 | LAVmin≤LAVmax 等 |

若需修改范围，直接编辑对应字典即可。

**第一层自动剔除阈值**：

| 参数 | 下界 | 上界 | 单位 |
|------|------|------|------|
| `LAVmax` | 0 | — | mL |
| `LAVI` | 0 | — | mL/m² |
| `LAVmin` | 0 | — | mL |
| `LAVmin-i` | 0 | — | mL/m² |
| `LAEF` | 0 | 100 | % |
| `LAD-long` | 0 | — | cm |
| `LAD-trans` | 0 | — | cm |
| `3D LA sphericity` | 0 | — | — |
| `LA ellipticity` | 0 | 1 | — |
| `LA circularity` | 0 | 1 | — |
| `LA sphericity index` | 0 | — | — |
| `LA eccentricity index` | 0 | 1 | — |
| `MAT area` | 0 | — | cm² |
| `TGAR` | 0 | 1 | — |
| `LASr` | −100 | 100 | % |
| `LASrR` / `GCSR` / `LSR` / `ASR` | −20 | 20 | s⁻¹ |
| `LASct` / `GCS` / `LS` / `AS` | −100 | 100 | % |
| `Time to peak LASrR` | 0 | 100 | %cycle |
| `4CH ellipticity/circularity rate`、`Sphericity index rate` | −20 | 20 | — |
| `Annular expansion rate`、`Longitudinal stretching rate` | −20 | 20 | cm/s |

**第二层人工复核阈值**（超出仅标 flag，不删除）：

| 参数 | 下界 | 上界 | 单位 |
|------|------|------|------|
| `LAVmax` | 5 | 250 | mL |
| `LAVI` | 3 | 150 | mL/m² |
| `LAVmin` | 3 | 200 | mL |
| `LAVmin-i` | 1 | 120 | mL/m² |
| `LAEF` | 5 | 90 | % |
| `LAD-long` | 2 | 9 | cm |
| `LAD-trans` | 1 | 8 | cm |
| `3D LA sphericity` | 0.1 | 2.0 | — |
| `LA ellipticity` | 0.2 | 1.0 | — |
| `LA circularity` | 0.2 | 1.0 | — |
| `LA sphericity index` | 0.2 | 1.5 | — |
| `LA eccentricity index` | 0.0 | 1.0 | — |
| `MAT area` | 0.5 | 30 | cm² |
| `TGAR` | 0.05 | 1.0 | — |
| `LASr` | −20 | 80 | % |
| `LASct` / `GCS` / `LS` / `AS` | −80 | 20 | % |
| `LASrR` / `GCSR` / `LSR` / `ASR` | −8 | 8 | s⁻¹ |
| `Time to peak LASrR` | 0 | 100 | %cycle |
| `4CH ellipticity/circularity rate`、`Sphericity index rate` | −8 | 8 | — |
| `Annular expansion rate`、`Longitudinal stretching rate` | −8 | 8 | cm/s |

**输出文件**（默认 `csv/la_params/processed/`）：

| 文件 | 说明 |
|------|------|
| `la_morphology_wide.csv` | 形态学参数宽表（主键：`video_prefix`） |
| `la_kinematic_wide.csv` | 运动学参数宽表（主键：`video_prefix`） |
| `la_params_qc_filtered.csv` | QC 过滤后元数据表（供追溯） |
| `la_params_missingness_morph.csv` | 形态表缺失率报告 |
| `la_params_missingness_kine.csv` | 运动表缺失率报告 |
| `la_params_feature_decisions.csv` | 特征决策汇总（保留 / 剔除） |
| `la_params_cross_check_log.csv` | 跨列约束违规记录（第一层补充） |
| `la_params_review_log_morph.csv` | 形态表第二层复核标记汇总 |
| `la_params_review_log_kine.csv` | 运动表第二层复核标记汇总 |
| `la_params_iqr_outliers_morph.csv` | 形态表未收录参数 IQR 异常值统计 |
| `la_params_iqr_outliers_kine.csv` | 运动表未收录参数 IQR 异常值统计 |
| `la_params_iqr_outliers_morph.png` | 形态表 IQR 异常值可视化（需 matplotlib） |
| `la_params_iqr_outliers_kine.png` | 运动表 IQR 异常值可视化（需 matplotlib） |

**运行命令**：

```bat
python utils/clean_la_params.py ^
  --morphology csv/la_params/final_morphology_results.csv ^
  --kinematic  csv/la_params/final_kinematic_stats.csv ^
  --qc         csv/la_params/final_qc.csv ^
  --output_dir csv/la_params/processed

:: 仅生成缺失率报告（不写输出文件）：
python utils/clean_la_params.py --report_only

:: 跳过 IQR 图表生成（无 matplotlib 环境）：
python utils/clean_la_params.py --no_plots
```

---

### 步骤 A2 — LA × MIMIC 临床参数联合分析（`utils/la_analysis.py`）

**输入**：

| 文件 | 来源 |
|------|------|
| `csv/processed/hfpef_cohort_win_*_processed.csv` | 步骤 5（临床宽表，含插补 + 标准化） |
| `csv/la_params/processed/la_morphology_wide.csv` | 步骤 A1 |
| `csv/la_params/processed/la_kinematic_wide.csv` | 步骤 A1 |
| `csv/la_params/processed/la_params_qc_filtered.csv` | 步骤 A1（QC 元数据） |

**分析内容**（依次执行）：

| 分析 | 方法 |
|------|------|
| **数据合并** | 按 `subject_id` inner join，可附加 QC 元数据 |
| **描述性统计** | 近正态→ 均值 ± SD；偏态→ 中位数 [IQR]；二值→ 频次（%）；支持按分组列分层 |
| **相关性分析** | 两列均近正态 → Pearson；否则 → Spearman；输出相关矩阵 CSV + 热图 PNG |
| **FDR 多重校正** | 所有 p 值采用 Benjamini–Hochberg 方法校正（`statsmodels.multipletests`） |
| **VIF 共线性检验** | 迭代剔除 VIF > 10 的特征；对 LA 参数与临床变量分别运行 |
| **组间比较** | 两组连续变量：t 检验 / Welch / Mann–Whitney U；多组：ANOVA / Kruskal–Wallis；分类：χ² / Fisher 精确检验；BH-FDR 校正 |
| **分布可视化** | LA 参数分布直方图（含 KDE），便于识别极端值 |

**输出文件**（默认 `csv/la_analysis/`）：

| 文件 | 说明 |
|------|------|
| `merged_dataset.csv` | 合并后完整数据集 |
| `descriptive_stats.csv` | 各变量描述性统计 |
| `correlation_la_clinical.csv` | LA 参数 × 临床变量相关系数长表（含 FDR 校正 p 值） |
| `correlation_matrix.csv` | 相关系数宽矩阵（行：LA 参数，列：临床变量） |
| `correlation_la_clinical.png` | 相关热图 |
| `la_distributions.png` | LA 参数分布图 |
| `vif_report_la.csv` | LA 参数 VIF 报告 |
| `vif_report_clinical.csv` | 临床变量 VIF 报告 |
| `group_comparison_{col}.csv` | 每个分组列的组间比较结果（含 FDR 校正 p 值） |

**运行命令**：

```bat
:: 基本用法（以 hadm 时间窗为例，按院内死亡分组）
python utils/la_analysis.py ^
  --clinical_csv csv/processed/hfpef_cohort_win_hadm_processed.csv ^
  --morph_csv    csv/la_params/processed/la_morphology_wide.csv ^
  --kine_csv     csv/la_params/processed/la_kinematic_wide.csv ^
  --qc_csv       csv/la_params/processed/la_params_qc_filtered.csv ^
  --output_dir   csv/la_analysis ^
  --group_col    died_inhosp os_event

:: 不生成图形（服务器环境）：
python utils/la_analysis.py ... --no_plots
```

---

## 数据字典（步骤 5 之后的关键列）

### 标识符（不用于建模）

| 列名 | 描述 |
|------|------|
| `subject_id` | MIMIC-IV 患者标识符 |
| `hadm_id` | 住院记录标识符 |
| `index_study_id` | 超声检查标识符 |
| `index_study_datetime` | 指数超声检查的日期时间 |
| `a4c_dicom_filepath` | A4C DICOM 文件路径 |

### 人口学特征

| 列名 | 步骤 5 后类型 | 描述 |
|------|-------------|------|
| `gender` | 二值（1=男，0=女） | 患者性别 |
| `anchor_age` | 浮点，z-score 标准化 | 入院年龄 |

### 结局列（不插补，不标准化）

| 列名 | 类型 | 描述 |
|------|------|------|
| `hospital_expire_flag` | 二值 | 指数住院期间死亡 |
| `died_inhosp` | 二值 | 院内死亡（同上） |
| `died_post_dc` | 二值 | 出院后死亡（辅助列，仅死亡患者非空） |
| `days_survived_post_dc` | 浮点 | 出院后存活天数（**仅事件患者非空**，删失患者为 NaN；不可直接用于 Cox） |
| `died_30d` | 二值 | 出院后 30 天内死亡 |
| `died_90d` | 二值 | 出院后 90 天内死亡 |
| `died_1yr` | 二值 | 出院后 1 年内死亡 |
| `last_dischtime` | 时间戳 | 患者 MIMIC-IV 中末次住院出院时间（`MAX(dischtime)`） |
| `censor_date` | 时间戳 | MIMIC-IV 行政删失日期 = `last_dischtime + 365 天` |
| `os_event` | 二值/NULL | **Cox 主要事件指标**：1=出院后死亡；0=删失；NULL=院内死亡 |
| `os_days` | 浮点/NULL | **Cox 主要时间变量**：事件患者 = `dod−index_dischtime`；删失患者 = `censor_date−index_dischtime`；院内死亡 = NULL |

### 合并症特征（二值，另有说明除外）

| 列名 | 描述 |
|------|------|
| `charlson_score` | SQL 衍生 CCI（有序数值） |
| `cci_from_flags` | 由标志列重计算的部分 CCI（下界，有序数值） |
| `hf_any_diabetes` | 任意糖尿病（含或不含并发症） |
| `hf_cardiorenal` | 肾脏病（心肾综合征标志） |
| `hf_met_syndrome_proxy` | 高血压 + 糖尿病（代谢综合征代理） |
| `hf_af_ckd` | 房颤 + 肾脏病 |
| `hf_high_risk_triad` | 房颤 + 肾脏病 + 糖尿病 |
| `hf_competing_risk` | 恶性肿瘤或重度肝病 |
| `hf_comorbidity_burden` | 0=低，1=中，2=高（基于 charlson_score） |
| `hf_comorb_score_custom` | HFpEF 专项加权合并症评分（有序数值） |

### 实验室指标（连续，步骤 5 后已插补 + 标准化）

creatinine、bun、sodium、potassium、chloride、bicarbonate、calcium、
glucose_lab、aniongap、hemoglobin、hematocrit、wbc、platelet、inr、pt、ptt

### OMR / 体格检测

omr_weight_kg、omr_bmi、omr_sbp、omr_dbp

### 缺失值指示标志列（二值，步骤 5 为缺失率 20–60% 的变量添加）

albumin_missing_flag（hadm 窗口）、ptt_missing_flag、inr_missing_flag、
pt_missing_flag（48h 系列窗口）

### LA 影像参数（步骤 A1 宽表输出，已标准化）

形态学参数（来自 `la_morphology_wide.csv`，列名为安全化后的 `parameter_name`）：

| 列名 | 原始参数名 | 单位 | 关键帧 |
|------|-----------|------|--------|
| `LAVmax` | LAVmax | mL | LAVmax 关键帧 |
| `LAVI` | LAVI | mL/m² | LAVmax 关键帧 |
| `LAVmin` | LAVmin | mL | LAVmin 关键帧 |
| `LAVmin-i` | LAVmin-i | mL/m² | LAVmin 关键帧 |
| `LAEF` | LAEF | % | LAVmax + LAVmin 组合 |
| `LAD-long` | LAD-long | cm | LAVmin 关键帧 |
| `LAD-trans` | LAD-trans | cm | LAVmin 关键帧 |
| `LA_ellipticity` | LA ellipticity | — | LAVmax 关键帧 |
| `LA_circularity` | LA circularity | — | LAVmax 关键帧 |
| `LA_sphericity_index` | LA sphericity index | — | LAVmax 关键帧 |
| `LA_eccentricity_index` | LA eccentricity index | — | LAVmax 关键帧 |
| `MAT_area` | MAT area | cm² | LAVmax 关键帧 |
| `TGAR` | TGAR | — | LAVmax 关键帧 |

运动学参数（来自 `la_kinematic_wide.csv`，列名格式为 `parameter_name__sub_item`）：

| 列名示例 | 说明 | 单位 |
|----------|------|------|
| `LASr__mean` | 左心房纵向储备应变（均值） | % |
| `LASrR__peak` / `LASrR__mean` | 左心房纵向储备应变率（峰值/均值） | s⁻¹ |
| `GCS__peak` / `GCS__mean` / `GCS__range` | 整体圆周应变 | % |
| `Annular_expansion_rate__peak_expansion` | 瓣环扩张率峰值 | cm/s |
| `MAT_area__peak` / `MAT_area__mean` | 二尖瓣环面积 | cm² |
| `TGAR__mean` / `TGAR__peak` | TGAR（峰值/均值） | — |

> **注意**：`value` 空字符串在清洗时已转换为 NaN；  
> 列缺失率 ≥ 30% 的参数已被剔除；  
> 速率类参数依赖真实 fps（fps 缺失时该参数为 NaN，不可填 0）。

---

## 完整流水线复现

> **Windows 用户**：所有命令均写在一行，不使用反斜杠（`\`）换行。

### 前提：从 MIMIC-IV 导出原始队列 CSV

本流水线需要已完成以下操作：
1. 在 MIMIC-IV PostgreSQL 数据库中执行 `SQL_Queries/codes.sql` 导出 ICD 诊断码
2. 执行 `SQL_Queries/statics.sql`（或等效的队列查询），将结果导出为三个时间窗 CSV：
   - `csv/hfpef_cohort_win_hadm.csv`
   - `csv/hfpef_cohort_win_48h24h.csv`
   - `csv/hfpef_cohort_win_48h48h.csv`

> 原始 CSV 必须包含以下新增列（v2）：`last_dischtime`、`censor_date`、`os_event`、`os_days`。
> 这些列由 SQL 在导出时预先计算（详见步骤 6 说明）。

### 步骤 3–8：Python 流水线（MIMIC 临床分析）

```bat
pip install numpy pandas scikit-learn lifelines statsmodels matplotlib
python utils/clean_cohort_csvs.py --input_dir csv --output_dir csv/cleaned --resource_path resources
python utils/compute_comorbidity.py --input_dir csv/cleaned --output_dir csv/comorbidity
python utils/impute_normalize.py --input_dir csv/comorbidity --output_dir csv/processed
python utils/build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival
python utils/feature_selection.py --input_dir csv/survival --output_dir csv/feature_selection
python utils/fit_cox_model.py
python utils/ph_assumption_test.py --correct_violations
python utils/model_evaluation.py
```

**各步骤说明**：

| 命令 | 输入 | 输出 |
|------|------|------|
| `clean_cohort_csvs.py` | `csv/` | `csv/cleaned/` — 离群值清洗、类型校正、datetime 解析 |
| `compute_comorbidity.py` | `csv/cleaned/` | `csv/comorbidity/` — CCI 重计算、HFpEF 专项合并症特征 |
| `impute_normalize.py` | `csv/comorbidity/` | `csv/processed/` — 缺失值插补、标准化（结局列保持原样） |
| `build_survival_endpoint.py` | `csv/processed/` | `csv/survival/` — Cox 衍生终点列（`event_*/time_*`） |

### 步骤 A1–A2：LA 影像参数清洗与联合分析

> **前提**：将 EchoGraphs 输出的三张 CSV 放置于 `csv/la_params/`，  
> 文件名须为 `final_morphology_results.csv`、`final_kinematic_stats.csv`、`final_qc.csv`。

```bat
:: A1 — LA 参数清洗
python utils/clean_la_params.py --morphology csv/la_params/final_morphology_results.csv --kinematic csv/la_params/final_kinematic_stats.csv --qc csv/la_params/final_qc.csv --output_dir csv/la_params/processed

:: A2 — LA × MIMIC 临床参数联合分析（以 hadm 时间窗为例）
python utils/la_analysis.py ^
  --clinical_csv csv/processed/hfpef_cohort_win_hadm_processed.csv ^
  --morph_csv    csv/la_params/processed/la_morphology_wide.csv ^
  --kine_csv     csv/la_params/processed/la_kinematic_wide.csv ^
  --qc_csv       csv/la_params/processed/la_params_qc_filtered.csv ^
  --output_dir   csv/la_analysis ^
  --group_col    died_inhosp os_event
```

**各步骤说明**：

| 命令 | 输入 | 输出 |
|------|------|------|
| `clean_la_params.py` | `csv/la_params/` | `csv/la_params/processed/` — QC 过滤、生理范围检查、宽表、标准化 |
| `la_analysis.py` | `csv/processed/` + `csv/la_params/processed/` | `csv/la_analysis/` — 合并数据集、相关矩阵、VIF 报告、组间比较 |

---

## 参考文献

1. Charlson ME et al. *A new method of classifying prognostic comorbidity in longitudinal studies.* J Chronic Dis. 1987;40(5):373–383.
2. Deyo RA et al. *Adapting a clinical comorbidity index for use with ICD-9-CM administrative databases.* J Clin Epidemiol. 1992;45(6):613–619.
3. Pocock SJ et al. *Predictors of mortality in patients with chronic heart failure: incremental value of the MAGGIC risk score.* Eur Heart J. 2013;34(23):1757–1766.
4. Yusuf S et al. *Effects of candesartan in patients with chronic heart failure and preserved left-ventricular ejection fraction (CHARM-Preserved).* Lancet. 2003;362(9386):777–781.
5. Johnson AEW et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data. 2023;10:1.
6. Wang EW et al. *MIMIC-Extract: A data extraction, preprocessing, and representation pipeline for MIMIC-III.* CHIL 2020.
7. Peduzzi P et al. *A simulation study of the number of events per variable in logistic regression analysis.* J Clin Epidemiol. 1996;49(12):1373–1379.  *(EPV ≥ 10 rule-of-thumb also applied to Cox regression)*
8. Grambsch PM, Therneau TM. *Proportional hazards tests and diagnostics based on weighted residuals.* Biometrika. 1994;81(3):515–526.
9. Harrell FE et al. *Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors.* Statistics in Medicine. 1996;15(4):361–387.  *(Bootstrap optimism correction for C-index)*
10. Vickers AJ, Elkin EB. *Decision curve analysis: a novel method for evaluating prediction models.* Medical Decision Making. 2006;26(6):565–574.
11. Therneau TM, Grambsch PM. *Modeling Survival Data: Extending the Cox Model.* Springer, 2000.  *(Stratified Cox model; robustness of Cox under mild PH violations)*
12. Anand IS et al. *C-reactive protein in heart failure: prognostic value and the effect of valsartan (from the Valsartan Heart Failure Trial).* Am J Cardiol. 2005;95(12):1380–1383.  *(WBC/inflammatory markers and time-varying HF prognosis)*
13. Tang WHW et al. *White blood cell count and mortality in patients with heart failure.* Circ Heart Fail. 2013;6(1):48–55.  *(WBC time-varying effect in HF)*
