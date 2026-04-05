# Style Sample

Use this sample only as a reference for paragraph rhythm, density, and texture.

Do not treat it as a factual source.

```latex
\BiSubsection{MIMIC-IV-ECHO数据集}{MIMIC-IV-ECHO Dataset}
研究数据取自MIMIC-IV-ECHO模块。该模块属于大型重症监护数据库MIMIC-IV。MIMIC（Medical Information Mart for Intensive Care）由麻省理工学院计算生理学实验室与贝斯以色列女执事医疗中心联合发布。数据采集经机构审查委员会批准，患者知情同意获得豁免。所有受保护的健康信息都经过处理，符合HIPAA（Health Insurance Portability and Accountability Act）安全港标准。MIMIC-IV-ECHO收录了2017年至2019年间的超声心动图数据。这些患者同样出现在MIMIC-IV临床数据库中。两个模块通过subject\_id这一唯一标识符实现跨平台关联。

MIMIC-IV-ECHO中的数据采集使用GE医疗的Vivid E90、Vivid E95和Vivid S7超声设备。原始数据超过50万份超声心动图，来自4579名患者，涵盖7243次检查，每次检查由多个图像序列构成，每个序列对应一个心脏特定切面。约5\%的数据被预留为隐藏测试集，未包含在当前公开版本。图像以未压缩的DICOM（Digital Imaging and Communications in Medicine）格式，即医学图像存储的国际标准进行存储，每个DICOM文件包含一个连续图像序列，完整记录心脏在一个或多个心动周期的动态变化。文件按subject\_id分层存放，方便批量数据读取和按患者筛选。以路径files/p10/p10690270/s95240362/95240362\_0004.dcm为例：p10是subject\_id前两位，p10690270是完整subject\_id，s95240362是检查study\_id，末尾数字为视图编号。

MIMIC-IV-ECHO的一大优势是与MIMIC-IV临床数据库紧密耦合。借助subject\_id，超声影像数据能够关联到患者的人口统计学信息、诊断记录、用药信息、实验室检查和心电图等。超声检查的时间戳和患者在急诊科（ED，Emergency Department）或重症监护室（ICU，Intensive Care Unit）的记录时间不一定重合。部分超声检查是在住院时间之外采集的。当前版本还没有包含机器自动生成的测量报告和心内科医师的判读报告。后者预计在后续的MIMIC-IV Note模块发布。目前，只有大约12\%的检查能通过衍生表关联到两日内完成的医师报告。

本研究基于左心房心功能分析的临床实践期待，对于MIMIC-IV-ECHO中的数据进行了初步筛选。鉴于MIMIC-IV中的人群多为一种或多种疾病患者，为最大化左心房测量在诊断中的作用，结合MIMIC-IV中的患者数据，筛选了部分HFpEF人群作为初步的研究对象，具体纳排标准详见5.1.2。
```
