# **MIMIC-Extract**：面向 MIMIC-III 的数据提取、预处理与表征流程

# 关于
本仓库包含 **MIMIC-Extract** 相关代码。代码按功能分为以下文件夹：
* Data：本地存放待提取的数据。
* Notebooks：Jupyter  Notebook，用于演示测试用例，以及在风险预测与干预预测任务中如何使用输出数据。
* Resources：包含以下文件：
  - `Rohit_itemid.txt`：描述 MIMIC-III 中条目 ID 与 Rohit 所用 MIMIC II 条目 ID 的对应关系；
  - `itemid_to_variable_map.csv`：数据提取的核心文件，包含条目 ID 分组及可直接提取的条目 ID 列表；
  - `variable_ranges.csv`：描述各变量的正常取值范围，辅助提取合规数据；
  同时还包含输出表格的预期结构。
* Utils：运行 **MIMIC-Extract** 数据流程的脚本与详细说明。
* `mimic_direct_extract.py`：数据提取脚本。

# 引用论文
如果你在研究中使用了本代码，请引用以下文献：

```
Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, 
and Marzyeh Ghassemi. MIMIC-Extract: A Data Extraction, Preprocessing, and Representation 
Pipeline for MIMIC-III. arXiv:1907.08322. 
```

# 预处理后输出结果
如果你希望直接在研究中使用本流程的输出结果，可通过 GCP 获取默认参数下的预处理版本：
[点击访问](https://console.cloud.google.com/storage/browser/mimic_extract)

访问该数据需要通过 PhysioNet 完成 MIMIC-III GCP 权限认证。
相关说明见 [PhysioNet 文档](https://mimic.physionet.org/gettingstarted/cloud/)。

本输出按“现状”提供，不做任何保证。如发现问题，可通过 GitHub Issues 反馈。

# 分步使用说明
前面若干步骤与上文一致。
本说明基于版本号为 `762943eab64deb30bdb2abcf7db43602ccb25908` 的 `mimic-code` 测试通过。

## 步骤 0：所需软件与依赖
本地系统需将以下可执行文件加入环境变量 `PATH`：
* conda
* psql（PostgreSQL 9.4 及以上）
* git
* MIMIC-III PostgreSQL 数据库（参考 [MIT-LCP 仓库](https://github.com/MIT-LCP/mimic-code)）

以下所有命令均在终端执行，且当前目录为 `utils/`。

## 步骤 1：创建 conda 环境
通过 [mimic_extract_env_py36.yml](../mimic_extract_env_py36.yml) 创建新 conda 环境并激活：

```
conda env create --force -f ../mimic_extract_env_py36.yml
```

该步骤**在 pip 安装阶段可能提示失败**，属于正常现象。
只需照常激活环境（即使提示“失败”，环境通常仍可激活）：

```
conda activate mimic_data_extraction
```

然后使用 `pip` 手动安装失败的包（例如 `pip install [package]`），
常见需要手动安装的包：`datapackage`、`spacy`、`scispacy`。

同时还需要为 spacy 安装英文语言模型：
`python -m spacy download en_core_web_sm`

#### 预期结果
所需环境创建并激活成功。

#### 预期资源消耗
通常耗时 < 5 分钟，需要稳定网络。

## 步骤 3：构建特征提取所需视图
将在 MIMIC PostgreSQL 库中生成物化视图，
包括 [MIT-LCP 仓库](https://github.com/MIT-LCP/mimic-code) 中的所有概念表，
以及用于提取非机械通气、晶体液推注、胶体液推注的相关表。

注意：你的 Postgres 用户需要拥有模式（schema）修改权限，才能按此方式构建概念表。
首先将本 GitHub 仓库克隆至本地，假设路径存放在环境变量 `$MIMIC_CODE_DIR` 中。
克隆完成后执行：

```
cd $MIMIC_CODE_DIR/concepts
psql -d mimic -f postgres-functions.sql
bash postgres_make_concepts.sh
```

接下来需要为本流程额外构建 3 个物化视图（同样需要 schema 修改权限）：
进入 `utils` 目录，依次运行：
`bash postgres_make_extended_concepts.sh`
`psql -d mimic -f niv-durations.sql`

## 步骤 4：设置队列筛选与提取规则
进入本仓库根目录，激活 conda 环境，
根据需要传入参数运行：
`python mimic_direct_extract.py ...`

#### 预期结果
默认配置会在 `MIMIC_EXTRACT_OUTPUT_DIR` 下生成一个 HDF5 文件，包含 4 张表：
* **patients**：静态人口学信息与静态结局
  * 每行对应：`(subj_id, hadm_id, icustay_id)`
* **vitals_labs**：时变生命体征与检验指标（按小时统计均值、计数、标准差）
  * 每行对应：`(subj_id, hadm_id, icustay_id, hours_in)`
* **vitals_labs_mean**：时变生命体征与检验指标（仅按小时均值）
  * 每行对应：`(subj_id, hadm_id, icustay_id, hours_in)`
* **interventions**：按小时的二分类干预执行标记
  * 每行对应：`(subj_id, hadm_id, icustay_id, hours_in)`

#### 预期资源消耗
耗时约 5–10 小时，
需要性能较好的机器，至少 50GB 内存。

#### 设置样本量
默认会构建所有符合条件患者的数据集。
若仅需小样本（如调试），可设置环境变量 `POP_SIZE`。

例如仅使用前 1000 名患者构建数据集：

# 常见错误 / 常见问题
1. 运行 `mimic_direct_extract.py` 时出现如下错误：
   ```
   psycopg2.OperationalError: could not connect to server: No such file or directory
   Is the server running locally and accepting
   connections on Unix domain socket "/tmp/.s.PGSQL.5432"?
   ```
   或
   ```
   psycopg2.OperationalError: could not connect to server: No such file or directory
   Is the server running locally and accepting
   connections on Unix domain socket "/var/run/postgresql/..."?
   ```
   解决方案可参考 [Stack Overflow 帖子](https://stackoverflow.com/questions/5500332/cant-connect-the-postgresql-with-psycopg2)，
   并使用 `--psql_host` 参数：
   可在调用 `mimic_direct_extract.py` 时直接传入，
   或在 Makefile 中设置环境变量 `HOST`。

2. `relation "code_status" does not exist`
   表示 `code_status` 表未成功构建，需要重新构建 MIMIC-III 概念表。
   具体步骤见两种说明文档中的**步骤 3**，也可参考下方“构建概念表常见错误”。

## 构建概念表常见错误
1. 构建概念表时提示无权限修改 `mimiciii` schema。
   说明默认 psql 用户无构建概念表的权限，需要使用更高权限的 Postgres 用户执行命令。
   这种情况常见于多用户只读环境。
   完成后可能需要额外配置，使其他用户可访问生成的概念表。

2. 已构建概念表，但代码无法识别。
   可能原因：
   - 无读取新表的权限；
   - 表位于错误的命名空间。
   本项目要求表在 `mimiciii` 命名空间下且完全可见。
   可使用高权限用户登录，手动调整权限与命名空间，示例命令：
   * `ALTER TABLE code_status SET SCHEMA mimiciii;`
   * `GRANT SELECT ON mimiciii.code_status TO [USER];`
   注意：需要对脚本访问的**每一张概念表**执行以上操作。