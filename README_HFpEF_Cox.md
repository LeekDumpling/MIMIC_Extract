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
resources/                    # MIMIC-Extract variable_ranges.csv 等资源文件
utils/
  clean_cohort_csvs.py        # 步骤 3
  compute_comorbidity.py      # 步骤 4
  impute_normalize.py         # 步骤 5
  build_survival_endpoint.py  # 步骤 6
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
| 7 | 特征选择（单变量 Cox + VIF + LASSO） | ⬜ 待完成 | — |
| 8 | Cox PH 模型拟合（终点 × 时间窗） | ⬜ 待完成 | — |
| 9 | PH 假设检验（Schoenfeld 残差） | ⬜ 待完成 | — |
| 10 | 模型评估（C-index、Brier Score） | ⬜ 待完成 | — |
| 11 | 可视化（KM 曲线、森林图） | ⬜ 待完成 | — |

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

### 步骤 8 — Cox PH 模型拟合（待完成）

**计划使用库**：`lifelines`（首选）或 `scikit-survival`。

对 15 个组合（3 时间窗 × 5 终点），各评估三种模型规格：

| 模型 | 特征 | 用途 |
|------|------|------|
| 基础模型 | 年龄 + 性别 + charlson_score | 临床参考基准 |
| 实验室模型 | 基础模型 + 保留实验室指标 | 实验室增强 |
| 完整模型 | 步骤 7 保留的所有特征 | 最大预测能力 |

**实现示意**（每个终点 × 时间窗循环）：
```python
from lifelines import CoxPHFitter
for endpoint in ['died_post_dc', 'died_30d', 'died_90d', 'died_1yr']:
    for window in ['hadm', '48h24h', '48h48h']:
        df_model = build_model_df(window, endpoint)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_model, duration_col=f'time_{endpoint}', event_col=f'event_{endpoint}')
        cph.print_summary()
```

---

### 步骤 9 — PH 假设检验（待完成）

**方法**：Schoenfeld 残差检验（全局及逐变量）。
- `lifelines.statistics.proportional_hazard_test(cph, df_model)`
- 对违反 PH 假设的变量：考虑时变系数或分层
  （如 `strata=['hf_comorbidity_burden']`）。

---

### 步骤 10 — 模型评估（待完成）

**评价指标**：
- **C-index（一致性指数）**：主要区分度指标（等价于生存数据的 AUC-ROC）。
- **综合 Brier 评分（IBS）**：整个随访期内的校准度评估。
- **校准曲线**：30 天/90 天/1 年时观测生存率 vs. 预测生存率。
- **交叉验证**：5 折 CV 获得无偏 C-index 估计。

---

### 步骤 11 — 可视化（待完成）

**计划图表**：
1. 按 `hf_comorbidity_burden`（低/中/高）分层的 Kaplan–Meier 曲线。
2. 按 `hf_high_risk_triad` 分层的 KM 曲线。
3. Cox 模型风险比（HR）及 95% CI 的森林图。
4. 连续预测变量（年龄、肌酐）的偏效应图。
5. 用于 PH 假设评估的 log-log 图（log[−log S(t)] vs. log t）。

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

### 步骤 3–6：Python 流水线

```bat
pip install numpy pandas scikit-learn
python utils/clean_cohort_csvs.py --input_dir csv --output_dir csv/cleaned --resource_path resources
python utils/compute_comorbidity.py --input_dir csv/cleaned --output_dir csv/comorbidity
python utils/impute_normalize.py --input_dir csv/comorbidity --output_dir csv/processed
python utils/build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival
```

**各步骤说明**：

| 命令 | 输入 | 输出 |
|------|------|------|
| `clean_cohort_csvs.py` | `csv/` | `csv/cleaned/` — 离群值清洗、类型校正、datetime 解析 |
| `compute_comorbidity.py` | `csv/cleaned/` | `csv/comorbidity/` — CCI 重计算、HFpEF 专项合并症特征 |
| `impute_normalize.py` | `csv/comorbidity/` | `csv/processed/` — 缺失值插补、标准化（结局列保持原样） |
| `build_survival_endpoint.py` | `csv/processed/` | `csv/survival/` — Cox 衍生终点列（`event_*/time_*`） |

---

## 参考文献

1. Charlson ME et al. *A new method of classifying prognostic comorbidity in longitudinal studies.* J Chronic Dis. 1987;40(5):373–383.
2. Deyo RA et al. *Adapting a clinical comorbidity index for use with ICD-9-CM administrative databases.* J Clin Epidemiol. 1992;45(6):613–619.
3. Pocock SJ et al. *Predictors of mortality in patients with chronic heart failure: incremental value of the MAGGIC risk score.* Eur Heart J. 2013;34(23):1757–1766.
4. Yusuf S et al. *Effects of candesartan in patients with chronic heart failure and preserved left-ventricular ejection fraction (CHARM-Preserved).* Lancet. 2003;362(9386):777–781.
5. Johnson AEW et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data. 2023;10:1.
6. Wang EW et al. *MIMIC-Extract: A data extraction, preprocessing, and representation pipeline for MIMIC-III.* CHIL 2020.
