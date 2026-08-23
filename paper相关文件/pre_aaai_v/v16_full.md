*CAHAD: Distilling Home-Field Culture-Authority Collaboration via
Culture-Aware Hybrid Adaptive Distillation for Multi-Agent Cultural
Alignment*

# **摘要**

多智能体协商已成为缓解大语言模型跨文化推理偏差的主流范式，但现有方法普遍采用等权协商假设，忽视了文化知识固有的属地性与不对称性。这一假设导致缺乏目标文化知识的多数智能体系统性地压制具备该知识的少数判断，同时多智能体推理的高计算开销亦制约了其实际部署。为此，本文提出两种相互衔接的方法：HF-CAC（Home-Field Culture-Authority Collaboration）多智能体协作推理框架依据目标文化动态激活主场 Guardian 智能体并赋予其决策主导权，通过三阶段结构化协商流程与基于文化亲缘度矩阵的加权仲裁回退机制，实现了与文化知识不对称结构相适配的协作推理范式；CAHAD（Culture-Aware Hybrid Adaptive Distillation）混合蒸馏方法将 HF-CAC 中 Judge 的最终答案与 Guardian 的文化方向判断分别映射为监督微调与强化学习的训练信号，采用在线联合优化并辅以基于命中率的难度自适应加权机制，将多智能体系统的文化推理能力迁移至单一模型。在 NormAd、CulturalBench 与 BLEnD 三个跨文化基准上的实验表明，HF-CAC 在准确率上显著优于现有等权协商方法，CAHAD 在保持与多智能体系统相当的文化对齐性能的同时将推理开销降低了一个数量级。

# **1 引言**

文化差异并非源于个体层面的偶然偏好，而是根植于特定社会结构与历史传统中的系统性规律。Hofstede（1980）提出的文化维度理论将文化差异量化为可比较的维度，Inglehart 与 Welzel（2005）通过世界价值观调查（WVS）进一步将全球文化映射为二维空间的文化地图。这些研究表明，文化知识具有明确的属地性——某一文化规范的权威解释权天然归属于该文化自身的持有者（Geertz 1983；Triandis 1995）。

近年来，大语言模型在涉及社会规范与文化知识的任务上普遍表现出对西方文化的偏向（Tao et al. 2024；Cao et al. 2023），其根源在于预训练语料中不同文化的知识分布显著不均衡（Li et al. 2024；Arora et al. 2023）。为缓解这一问题，多智能体文化推理成为一种主流思路：通过让多个具备不同文化背景先验的智能体进行协商或辩论，综合多方视角以弥补单一模型的文化知识盲区。现有方法普遍采用"等权协商"的设计——即所有参与协商的智能体享有相同的话语权重，最终结论通过多数投票或平权辩论产生。

然而，这一等权协商范式在文化推理中可能产生系统性的判断偏误。以图 1 所示的案例（选自 NormAd 数据集，Rao et al. 2025）为例：在韩国的一次聚餐场景中，Daniel 先为自己斟酒而后才为旁人斟酒，要求判断该行为的文化可接受性。五个不具备韩国文化背景的智能体（分别代表西方、拉美、非洲、伊斯兰、南亚文化圈）均依据各自文化中的通行惯例判定该行为"可接受"。然而，韩国饮酒礼仪要求在场者先为他人斟酒再为自己斟酒，先为自己斟酒构成对同席者的失礼。唯一具备相关文化知识的东亚文化圈智能体给出了正确判断"不可接受"，却在等权投票机制下以 1:5 的劣势被多数意见压制。

![图 1：动机示例（来自 NormAd 数据集）。在韩国文化中先给自己倒酒的判断。等权投票中 5 个不了解韩国饮酒礼仪的智能体压制了唯一正确的主场智能体，导致错误结论；HF-CAC 通过主场 Guardian 权威机制推翻多数票，得出正确判断。](figure/fig1_motivation.png){width="6.0in" height="4.128311461067367in"}

这一案例揭示了等权协商范式应用于文化推理时的结构性缺陷：文化知识具有鲜明的属地性与不对称性（Geertz 1983；Nisbett 2003），将这种内在不对称的知识结构置于平权协商框架之下，将系统性地导致缺乏目标文化知识的多数判断压制具备该知识的少数判断。如图 1 右侧所示，若依据认识论权威而非数量优势分配话语权，赋予主场智能体以更高的决策权重，裁决者便可依据其文化证据推翻多数意见，输出正确结论。

除等权协商的结构性问题外，多智能体推理固有的高计算开销与响应延迟亦制约了其在实际场景中的部署。因此，如何将多智能体协商过程中所积累的文化推理能力有效迁移至推理成本更低的单一模型，构成了本研究关注的第二个核心问题。

为解决上述两个问题，本文提出由推理阶段与训练阶段协同构成的两阶段方法。在推理阶段，本文提出 **H**ome-**F**ield **C**ulture-**A**uthority **C**ollaboration（HF-CAC）多智能体文化推理范式，引入"主场权威激活"机制，赋予与目标文化相匹配的智能体以决策主导权，并设计基于文化亲缘度的加权仲裁作为回退机制。在训练阶段，本文提出 **C**ulture-**A**ware **H**ybrid **A**daptive **D**istillation（CAHAD），将 Judge 的最终答案作为监督目标、Guardian 的文化方向判断作为强化学习引导信号，采用 SFT 与 RL 在线联合优化的混合训练范式，将多智能体的文化推理能力迁移到单一模型。

本文的主要贡献包含以下三个方面。第一，针对现有多智能体文化推理中等权协商假设与文化知识属地性之间的矛盾，本文提出 HF-CAC 协作推理框架，通过动态激活主场 Guardian 赋予目标文化智能体以决策主导权，并设计基于文化亲缘度矩阵的加权仲裁作为回退机制，从而构建了与文化知识不对称结构相适配的协作推理范式。第二，针对多智能体推理的高计算成本，本文提出 CAHAD 混合蒸馏方法，通过 SFT 与 RL 在线联合优化以及难度自适应加权机制，将多智能体系统的文化推理能力迁移至单一模型。第三，在 NormAd、CulturalBench 与 BLEnD 三个跨文化基准上的系统实验表明，HF-CAC 在准确率上显著优于现有等权协商方法，CAHAD 在保持与多智能体系统相当的文化对齐性能的同时，将推理开销降低了一个数量级。

# **2 相关工作**

## **2.1 多智能体文化推理**

多智能体文化推理的核心思路是通过为不同智能体赋予不同的文化身份或背景先验，使其围绕同一文化判断问题展开多轮协商或辩论，最终综合多方视角以弥补单一模型在特定文化上的知识盲区。近年来已有若干具有代表性的工作沿这一方向展开。

Ki et al.（2025）从公平性视角出发，系统考察了多智能体辩论中辩论轮次、智能体数量与模型组合对文化对齐公平性的影响，在 NormAd 基准上验证了多智能体辩论相较单一模型在整体准确率与文化群体间公平性上的双重提升；然而，该工作的重点在于评估协商结构的参数敏感性，并未提出针对文化知识不对称性的专门机制。MACD（Tan et al. 2026）提出了一种"求同存异"（Seeking Common Ground while Reserving Differences）的多智能体文化辩论框架，为每个智能体赋予特定的文化人格（cultural persona），通过多轮辩论逐步收敛至文化感知的共识答案；该方法在无需额外训练的前提下，在 CulturalBench 与 BLEnD 等文化基准上取得了显著优于单一模型的表现，但其协商过程中所有文化智能体的话语权重是均等的，未能区分"主场"与"客场"智能体在目标文化判断上的差异。OG-MAR（Seo et al. 2026）从本体论的角度出发，首先从世界价值观调查中提取受访者特定的价值观并构建全球文化本体（cultural ontology），然后基于该本体引导多智能体推理，在文化价值观预测任务上取得了较好的效果；然而，该方法侧重于价值观层面的文化建模，在涉及具体社会规范与日常礼仪的判断任务上并未充分考虑目标文化的权威性。总体而言，上述工作均采用等权协商的默认假设，且均为高成本的推理时方法，未能解决文化知识属地性与推理效率这两个核心问题。

## **2.2 多智能体蒸馏与训练范式**

为缓解多智能体系统的推理成本问题，已有研究尝试将多智能体协商过程中产生的推理能力蒸馏至单一模型。MAGDi（Chen
et
al. 2024）是这一方向的代表性工作，其核心思想是将多智能体辩论过程中各智能体之间的推理交互建模为结构化的有向图，通过图编码器（graph
encoder）增强学生模型的表征能力，并联合使用下一个 token
预测、正确与错误推理链的对比损失以及基于图结构的交互建模三个目标函数对学生模型进行训练；实验表明，MAGDi
蒸馏后的学生模型在数学推理与常识推理等七个基准上显著优于单教师蒸馏与多教师聚合基线，且推理效率提升了一个数量级。此后，Zhou
et al.（2025）提出 D&R（Debate and
Reflect）框架，通过编排学生模型与多个教师模型之间的多轮辩论，收集教师的错误分析与纠正策略作为反馈信号，并将辩论日志组织为树状偏好结构用于偏好优化训练，在多个推理基准上进一步提升了小模型的推理能力。然而，上述多智能体蒸馏方法均针对通用推理任务（如数学推理、常识推理）设计，尚未被系统地应用于文化对齐任务，其设计也较少考虑不同文化样本学习难度不均衡这一特点。

# **3 方法**

本文方法由两个相互衔接的组件构成（图 2）。HF-CAC（§3.1）为主场权威激活的多智能体协作推理范式，旨在生成高质量的文化判断；CAHAD（§3.2）为文化感知的混合自适应蒸馏方法，旨在将前者所蕴含的文化推理能力迁移至单一模型。

![图 2：本文方法总体架构示意图。左半部分为 Stage 1: HF-CAC
多智能体教师系统，包含 Guardian/Auditor 初始化、文化辩论与 Judge
裁决三个阶段；右半部分为 Stage 2: CAHAD 蒸馏训练，将 Judge 答案与
Guardian 引导信号分别映射为 SFT 与 RL 的训练目标，通过自适应加权（Adaptive Weighting）模块实现在线联合优化。](figure/fig2_architecture.png){width="6.0in"
height="2.8543143044619423in"}

## **3.1 HF-CAC：主场文化激活的协作结构**

文化知识与通用事实性知识存在本质差异，突出表现为属地性与不对称性两个特征。属地性是指某一文化规范的权威解释权天然归属于该文化自身的持有者（Geertz 1983）；不对称性是指不同判断者对同一目标文化的知识储备和解释能力并不对等（Triandis 1995; Inglehart and Baker 2000）。从立场认识论（standpoint epistemology）的角度看，对特定社会群体经验的理解应优先赋予该群体的内部成员（Harding 1991）。然而，现有多智能体方法将所有参与协商的智能体一视同仁，本质上假设了知识对称与地位平等的理想化协商环境，而这一假设与文化知识的属地性和不对称性直接冲突：等权投票或平权辩论可能使来自其他文化背景的浅层判断因数量优势压制真正了解该文化的判断，形成"多数压制少数权威"的谬误。

针对这一问题，本文提出 HF-CAC（Home-Field Culture-Authority
Collaboration）框架，其核心思想是根据目标问题所涉及的文化归属，动态激活一个具有对应文化背景先验的智能体作为"主场权威"（Guardian），赋予其在协商中更高的话语权重，同时保留其他文化背景的智能体作为"审议者"（Auditor）提供多元的对比视角，最终由中立的"裁决者"（Judge）综合所有信息给出最终判断。这一三角色结构的设计根植于社会认识论中的"认知劳动分工"（division of cognitive labor）思想（Kitcher 1990; Goldman 1999）：当知识群体中的成员在专业深度上存在显著差异时，合理的集体决策不应对所有声音赋予同等权重，而应依据各成员与议题之间的认识论距离来分配话语权。Guardian 作为与目标文化具有最短认识论距离的成员，承担对该文化的权威判断职责；Auditor 代表跨文化比较视角，其作用类似于人类学中"文化间三角校验"（intercultural triangulation）方法（Denzin 1978）所强调的交叉验证功能——通过引入异文化观察者的对比证据来检验内部者判断中可能存在的盲点；Judge 则充当独立于任何文化立场的元认知仲裁者，综合权衡不同来源的证据完成最终裁决。三者各司其职而非同质化地共同投票，从根本上打破了等权协商的潜在假设。

HF-CAC 的协商流程采用三阶段结构化设计，其逻辑框架与哈贝马斯商谈伦理学（Habermas 1984）中关于理想言说情境的程序性要求具有结构同源性——即参与者首先独立形成判断、而后通过理由交换进行批判性论辩、最终由程序化规则达成共识。第一阶段为独立生成阶段：Guardian 与所有 Auditor 在互不可见彼此输出的条件下，分别基于自身的文化认知基础对问题给出初步判断与推理依据；这一阶段的设计借鉴了社会判断理论（Social Judgment Theory; Sherif and Hovland 1961）中关于锚定效应的发现——当个体在形成判断前即暴露于他人的立场中时，其后续判断将显著偏向先行暴露的观点，从而损害判断的独立性与多样性。第二阶段为辩论与协商阶段：所有智能体在获知彼此的初步判断与理由后，围绕分歧点展开一轮结构化的辩论，Auditor 可以针对 Guardian 的判断提出质疑或补充对比证据，Guardian 也需要针对 Auditor 提出的反例或异议做出回应与澄清。这一阶段的核心价值在于实现"受控的认识论对抗"：社会心理学研究表明，结构化的意见冲突（structured controversy）相比简单投票或一致性追求，能够促使群体更深入地考量少数派立场中的有效信息（Johnson and Johnson 2009），从而使潜在的分歧在进入裁决前得到充分暴露。第三阶段为裁决阶段：中立的 Judge 综合前两阶段中 Guardian 与各 Auditor 的完整判断与辩论过程，依据既定的证据优先级规则给出最终结论，完成对分歧的消解。三阶段依次递进——独立生成保证判断的多样性与独立性，辩论与协商保证分歧的充分暴露与证据的交叉验证，裁决保证最终结论的权威性与一致性——共同构成了 HF-CAC 有别于现有等权协商方法的核心流程设计。

在裁决机制上，Judge 默认给予 Guardian 的判断更高权重。这一设计体现了证言正义（testimonial justice; Fricker 2007）的核心主张：认识论权威应当基于知者与知识对象之间的结构性关联来分配，而非简单地以人数或话语量来度量。在实际运行中，HF-CAC 的绝大多数案例通过以下两种路径即可完成裁决，无需触发更复杂的回退机制。第一种路径是全员共识：当 Guardian 与所有 Auditor 在独立生成或辩论阶段已经达成一致意见时，Judge 直接采纳该共识答案作为最终结论。第二种路径是 Guardian 优先采纳：当 Guardian 与部分 Auditor 存在分歧时，若 Guardian 在辩论阶段提供了具体、可验证的文化证据，Judge 则依据"优先采纳权"机制采纳 Guardian 的结论——即使 Guardian 在数量上处于少数，其文化证据的权威性仍然优先于 Auditor 的多数意见。这一机制与 Collins 和 Evans（2007）提出的"经验型专长"（experience-based expertise）理论一致：对特定实践领域的判断权应归属于拥有直接参与经验的行动者，而非仅凭间接知识进行推断的旁观者。实验统计表明，上述两种路径已覆盖了绝大多数案例（约 95% 以上），充分说明了主场权威机制在实际文化推理中的主导作用。

仅在极少数情况下，Guardian
可能因自身知识局限而未能给出有效判断（例如明确表示不确定），此时系统需要一种合理的回退机制来避免退回至简单多数投票。为此，本文引入了基于文化亲缘度矩阵的加权仲裁策略：将世界主要文化划分为六大文化圈，并基于各文化圈在权威跨文化调查数据中的相对位置构建一个文化圈间的亲缘度矩阵，用以刻画不同文化圈之间的相对距离。当
Guardian 失效时，Judge 转而查询该矩阵，在辩论阶段提供了具体文化证据的
Auditor
中，优先采信与目标文化亲缘度最高者的判断，而非不加区分地进行多数投票。这一设计的基本原理是：即便主场权威缺席，与目标文化价值观更为接近的文化圈通常也比价值观差异悬殊的文化圈拥有更可靠的类比推断能力。亲缘度矩阵的具体构建方法、计算过程及其在裁决协议中的完整使用细节见附录
A。

## **3.2 CAHAD：文化感知的混合自适应蒸馏**

尽管 HF-CAC 通过结构化的协作分工有效提升了跨文化推理的准确性，其多智能体协商的性质决定了每次推理均需多个智能体依次生成推理内容并经过裁决步骤，由此产生的计算开销与响应延迟显著高于单一模型。为此，本文提出 CAHAD（Culture-Aware Hybrid Adaptive Distillation）方法，旨在将 HF-CAC 多智能体系统中蕴含的结构化跨文化推理能力迁移至单一模型，在保持文化对齐性能的同时实现推理效率的显著提升。

CAHAD 的核心设计思想在于对 HF-CAC 多智能体系统中不同角色所承载的知识进行差异化映射。具体而言，Judge 的最终答案经过多方证据的综合与裁决，具有较高的标签可靠性，适合作为监督微调（SFT）阶段的蒸馏目标；Guardian 的文化方向判断则蕴含了对目标文化的方向性专家知识，更适合以奖励信号的形式注入强化学习（RL）阶段的优势函数，引导学生模型在策略探索过程中向正确的文化方向收敛，而非直接作为逐字模仿的监督目标。这一差异化映射的理论基础在于：Judge 的输出对应知识蒸馏（Hinton et al. 2015）中教师模型的软标签，提供稳定的模仿学习信号；Guardian 的输出则更接近于专家偏好信号，适合通过策略梯度方法进行间接传递。

在训练范式上，CAHAD 采用 SFT 与 RL 在线联合优化的策略，而非将二者作为两个独立阶段依次执行的序贯训练方式。在每一优化步中，模型同时接受来自 Judge 答案的监督微调损失与来自策略梯度的强化学习损失的联合约束，Guardian 的文化方向判断则以匹配度信号的形式注入 RL 阶段的优势函数，在奖励层面强化与文化权威判断一致的探索方向。此外，CAHAD 引入自适应文化难度加权机制，依据模型在各文化上的实时表现动态调整不同文化样本的训练权重，使模型对掌握程度较低的文化分配更多的学习资源。

该机制的核心度量指标为"命中率"（hitrate），其定义如下：hitrate 是一个以文化为粒度、按训练步持续更新的标量，用于刻画当前模型对某一特定文化的掌握程度。直觉上，hitrate 越高意味着模型在该文化上的在线采样准确率越高、掌握程度越好；hitrate 越低则表明模型对该文化仍存在较大的学习空间。在具体计算上，hitrate 并非直接取某一轮采样的瞬时准确率，而是采用指数滑动平均（Exponential Moving Average, EMA）进行平滑更新，以兼顾历史趋势与当前表现。设当前训练步中，模型在某一文化 $c$ 上从 $G$ 个 rollout 中产生的即时准确率为 $acc_{cur}^{c}$（即 $G$ 个 rollout 中回答正确的比例），则该文化的 hitrate 按如下规则更新：

$$hitrate^{c} \leftarrow m \cdot hitrate_{prev}^{c} + (1 - m) \cdot acc_{cur}^{c}$$

其中 $m = 0.9$ 为动量系数。采用 EMA 而非瞬时准确率的原因有二：其一，单轮 rollout 的采样数 $G$ 通常有限（本文默认 $G = 8$），瞬时准确率的方差较大，直接用于驱动权重调度会导致训练不稳定；其二，EMA 能够自然地编码模型在该文化上的学习轨迹——持续正确的文化将具有稳步上升的 hitrate，偶发性正确则不会导致 hitrate 跳跃式上升，从而使权重调度更加稳定。hitrate 的初始值设为 0.5，代表模型对各文化的初始掌握程度处于中等不确定状态。

基于上述 hitrate 度量，RL 引导权重与 SFT 监督权重均按式 (1) 定义：

$$w = 1 - hitrate,\quad w_{sft} = \max(1 - hitrate,w_{\min})$$

其中 $w$ 为 RL 阶段 Guardian 引导权重：当模型已较好掌握某文化（hitrate 趋近于 1）时，$w$ 可降至 0，使训练回归标准 GRPO 探索，避免已掌握文化上的冗余引导；当模型在某文化上表现较差（hitrate 趋近于 0）时，$w$ 趋近于 1，Guardian 引导信号被充分放大，引导模型向该文化的正确方向收敛。$w_{sft}$ 为 SFT 阶段监督权重，其设计逻辑与 $w$ 一致，但额外引入地板值 $w_{\min}$（默认 0.1），保证即便模型已较好掌握某文化，也保留弱监督信号以维持训练过程中的稳定文化锚定。

在强化学习阶段，标准 GRPO 采用组内 baseline 归一化定义优势函数。CAHAD
在此基础上叠加 Guardian 文化引导项，得到式 (2)：

$$A_{i} = A_{i}^{base} + \lambda \cdot w \cdot S_{guardian}$$

其中组内 baseline 优势 $A_{i}^{base} = R_{i} - \bar{R}$，$R_{i}$ 为第
$i$ 个 rollout
的奖励，由结果正确性与过程奖励模型（PRM）打分加权构成，$\bar{R}$ 为同一
prompt 下所有 rollout 的奖励均值；$S_{guardian} \in \{ 0,1\}$ 为
per-rollout 文化一致性信号，判断该 rollout 的文化推理方向是否与 Guardian
判断一致（而非最终选项对错），即使最终答案错误、只要文化方向正确也赋值为
1；$\lambda$ 为 Guardian 引导全局强度系数（默认
0.5），用于平衡原始奖励与 Guardian
信号、防止二值信号完全主导优势函数的符号方向。值得指出的是，$S_{guardian}$ 的赋值粒度为 per-rollout 而非 per-prompt，即同一 prompt 下的不同 rollout 可获得不同的文化一致性信号。这一设计使得优势函数能够在组内区分文化方向正确与错误的轨迹，前者获得正向偏移而后者获得负向偏移，从而实现比统一偏移更为精细的策略引导。此外，权重因子 $w$ 的引入使得 Guardian 引导项随模型对该文化的掌握程度自适应衰减，避免在模型已充分习得某一文化后仍施加冗余引导。

在监督微调阶段，CAHAD 将 Judge 给出的最终答案 $y_{judge}$
作为蒸馏目标，采用标准交叉熵损失并按 $w_{sft}$ 加权，如式 (3) 所示：

$$L_{SFT}^{weighted} = w_{sft} \cdot L_{SFT} = - w_{sft} \cdot logP(y_{judge}|x)$$

由此，模型在某一文化上的掌握程度越低（$hitrate$ 越低、$w_{sft}$ 越大），该文化样本所承受的监督梯度越强，使 Judge 蕴含的文化知识优先向模型的薄弱文化区域迁移。地板值 $w_{\min}$ 的引入则保证了即便在模型已较好掌握某文化的条件下，仍保留最低限度的监督信号，以维持训练过程中的文化锚定稳定性。

最终，CAHAD 在同一次优化步内联合更新 RL 与 SFT
两项损失，总体训练目标如式 (4) 所示：

$$L = L_{GRPO}(A_{i}) + \beta \cdot w_{sft} \cdot L_{SFT}$$

其中 $L_{GRPO}(A_{i})$ 为基于式 (2) 中文化引导后优势 $A_{i}$ 计算的策略梯度损失，$\beta$ 为 RL 与 SFT 两项损失之间的量纲平衡系数。引入 $\beta$ 的必要性在于：$L_{GRPO}$ 为 on-policy 策略梯度损失，其量级通常远小于交叉熵损失 $L_{SFT}$，若不对后者进行缩放，SFT 项将主导梯度更新方向，导致联合训练退化为加权监督微调。式 (4) 中两项损失在同一次反向传播中联合优化，RL 项承担策略探索与文化方向引导的功能，SFT 项则以难度自适应的方式持续锚定 Judge 的文化知识，二者的协同作用使 CAHAD 能够在将多智能体系统的推理能力迁移至单一模型的过程中，同时保持文化推理的准确性与训练过程的稳定性。

# **4 实验设置**

## **4.1 数据集**

本文在三个跨文化基准数据集上进行评估。NormAd（Rao et
al. 2025）是一个社会规范判断数据集，任务形式为三分类（可接受、不可接受、中性），覆盖
75 个文化，共包含 2633 道题目。CulturalBench（Chiu et
al. 2025）是一个文化知识问答数据集，任务形式为四选一多选题，覆盖 45
个文化，共包含 1227 道题目。BLEnD（Myung et
al. 2024）是一个日常生活文化知识数据集，任务形式同样为四选一多选题，覆盖
15 个文化或地区，实验中使用的评估子集共包含 5081
道题目。三个数据集在任务类型、文化覆盖范围与题目规模上均存在差异，能够较为全面地检验方法在不同评估场景下的稳健性。

## **4.2 基线方法**

本文选取的基线方法涵盖四类策略。零样本方法包括基座模型直接推理（Zero-shot Base）与角色扮演提示（Roleplay Prompting），前者以未经任何文化对齐处理的基座模型建立性能下界，后者通过系统提示词赋予模型特定文化身份以引导其文化判断。单教师蒸馏方法为标准的监督微调（Single-Teacher SFT），使用文化标注语料对基座模型进行单轮训练。多智能体辩论方法选取 MD（Du et al. 2023）、MAD（Ki et al. 2025）、MACD（Tan et al. 2026）与 OG-MAR（Seo et al. 2026）四种代表性工作。多教师蒸馏方法选取 MAGDi（Chen et al. 2024）与 AgentArk 两类基线。所有实验均在 Qwen-2.5-7B-Instruct（Qwen Team 2024，以下简称 Qwen）与 Llama-3.1-8B-Instruct（Llama Team 2024，以下简称 Llama）两种开源基座模型上进行。

## **4.3 评估指标**

主评估指标为准确率（Accuracy），定义为各数据集全部题目上的整体正确率。此外，为评估蒸馏方法的部署效率收益，本文还报告推理阶段的 token 消耗量与响应时延两项效率指标，用以量化 CAHAD 相较于 HF-CAC 多智能体系统在推理成本上的改善幅度。

# **5 实验结果与分析**

## **5.1 HF-CAC 多智能体推理的实验结果**

表 1 报告了 HF-CAC 与零样本基线及四种多智能体基线方法在三个数据集、两个基座模型上的准确率。在几乎所有数据集与基座组合上，HF-CAC 均取得了最优表现。这一结果表明，主场权威激活机制通过赋予目标文化智能体以决策主导权，能够更充分地保留协商过程中蕴含的权威文化知识，从而有效避免等权投票对主场信息的稀释。

  ----------------------------------------------------------------------------------
         **Method**         **NormAd**   **CulturalBench**   **BLEnD**   **Avg**
  ------------------------- ------------ ------------------- ----------- ----------
  **Qwen2.5-7B-Instruct as                                   
      backbone model**                                       

            Base            59.78        72.13               66.09       66.00

          Role-play         59.51        71.31               66.52       65.78

             MD             64.68        67.40               64.71       65.60

             MAD            62.59        74.41               67.07       68.02

            MACD            65.78        72.05               64.34       67.39

           OG-MAR           61.98        65.85               63.35       63.73

         **HF-CAC**         **66.81**    **75.57**           **68.45**   **70.28**

   **Llama-3.1-8B-Instruct                                   
     as backbone model**                                     

            Base            58.15        67.21               65.95       63.77

          Role-play         58.64        68.03               65.87       64.18

             MD             65.36        51.59               66.48       61.14

             MAD            64.68        69.03               66.23       66.65

            MACD            60.24        66.10               66.33       64.22

           OG-MAR           59.36        63.98               63.61       62.32

         **HF-CAC**         **65.86**    **72.37**           **67.25**   **68.49**
  ----------------------------------------------------------------------------------

*表 1：HF-CAC
与零样本基线及多智能体基线在三个数据集、两个基座模型上的准确率（%）对比。*

**结构因素分析。**为探究 HF-CAC 性能的结构性来源，本文分别考察了智能体数量与协商轮次两个设计变量对准确率的影响（图 3、图 4）。

![图 3：智能体数量对准确率的影响。(a) Qwen 基座；(b) Llama
基座。](figure/fig3_agent_count.png){width="6.0in" height="2.1894291338582677in"}

如图 3 所示，两个基座上的准确率均随智能体数量的增加而总体上升，但上升速率与饱和点因数据集而异，Qwen 基座上个别数据集上出现非单调波动，表明智能体数量与协商结构之间存在交互作用。本文基于此结果在主实验中采用六个智能体作为默认配置。

![图 4：协商轮次对准确率的影响。(a) Qwen 基座；(b) Llama
基座。](figure/fig4_negotiation_rounds.png){width="6.0in" height="2.1894291338582677in"}

图 4 的结果表明，单轮协商即可取得最佳或接近最佳的性能，增加协商轮次并未带来显著的性能提升。这一现象表明，主场权威机制通过明确的证据优先级规则已能够在单轮交互中实现分歧的有效消解，而非依赖多轮迭代逐步收敛。

## **5.2 CAHAD 蒸馏模型的实验结果与部署效率分析**

表 2 报告了 CAHAD 与零样本基线、单教师基线、HF-CAC 多智能体系统以及多教师聚合基线在三个数据集、两个基座模型上的准确率。CAHAD 在全部数据集与基座组合上均优于其余单一模型基线，表明本文提出的混合训练范式相较传统监督蒸馏与多教师聚合方法，能够更有效地保留 HF-CAC 教师系统中的文化推理能力。结合表 1 的结果可以观察到，CAHAD 作为单一模型，其准确率已接近乃至在部分配置下超越多智能体基线方法。

  -------------------------------------------------------------------------------------
           **Method**           **NormAd**   **CulturalBench**   **BLEnD**   **Avg**
  ----------------------------- ------------ ------------------- ----------- ----------
    **Qwen2.5-7B-Instruct as                                     
        backbone model**                                         

    *零样本方法（Zero-shot）*                                    

              Base              59.78        72.13               66.09       66.00

            Role-play           59.51        71.31               66.52       65.78

   *单教师蒸馏（Single-Teacher                                   
         Distillation）*                                         

               SFT              65.15        66.13               68.26       66.51

   *多智能体协作（Multi-Agent                                    
        Collaboration）*                                         

             HF-CAC             66.81        75.57               68.45       70.28

   *多教师蒸馏（Multi-Teacher                                    
         Distillation）*                                         

              MAGDi             65.02        72.95               67.86       68.61

            AgentArk            65.40        73.77               68.06       69.08

            **CAHAD**           **66.29**    **75.00**           **68.57**   **69.95**

   **Llama-3.1-8B-Instruct as                                    
        backbone model**                                         

    *零样本方法（Zero-shot）*                                    

              Base              58.15        67.21               65.95       63.77

            Role-play           58.64        68.03               65.87       64.18

   *单教师蒸馏（Single-Teacher                                   
         Distillation）*                                         

               SFT              54.75        64.52               64.44       61.24

   *多智能体协作（Multi-Agent                                    
        Collaboration）*                                         

             HF-CAC             65.86        72.37               67.25       68.49

   *多教师蒸馏（Multi-Teacher                                    
         Distillation）*                                         

              MAGDi             59.32        71.31               66.07       65.57

            AgentArk            61.22        70.49               66.47       66.06

            **CAHAD**           **64.39**    **72.58**           **67.19**   **68.05**
  -------------------------------------------------------------------------------------

*表 2：CAHAD
与零样本、单教师、多智能体协作及多教师聚合基线在三个数据集、两个基座模型上的准确率（%）对比。其中
HF-CAC 行为多智能体推理时方法（非单一模型），置于此处以便与 CAHAD
的蒸馏效果直接对比。*

**部署效率分析。**

  -----------------------------------------------------------------------------------------------------------------------------------------------
   **Benchmark**   **Qwen     **Qwen    **HF-CAC   **HF-CAC   **CAHAD   **CAHAD    **CAHAD/Qwen   **CAHAD/Qwen   **HF-CAC/CAHAD   **HF-CAC/CAHAD
                   Token**   Lat(s)**   Token**    Lat(s)**   Token**   Lat(s)**  Token Ratio**   Lat Ratio**    Token Ratio**     Lat Ratio**
  --------------- --------- ---------- ---------- ---------- --------- ---------- -------------- -------------- ---------------- ----------------
      NormAd       398.12      0.39     5774.32      4.24     416.46      0.41        1.05×          1.05×           13.87×           10.34×

   CulturalBench   195.80      0.32     3700.04      3.69     211.08      0.34        1.08×          1.06×           17.53×           10.85×

       BLEnD       194.54      0.31     3492.27      3.11     207.88      0.28        1.07×          0.90×           16.80×           11.11×
  -----------------------------------------------------------------------------------------------------------------------------------------------

*表 3：Qwen2.5-7B-Instruct 基座下推理效率对比（Token 消耗与时延）。*

表 3 报告了 Qwen 基座模型直接推理、HF-CAC 多智能体系统与 CAHAD 单一模型在推理阶段的 token 消耗量与响应时延。由于 HF-CAC 每次推理均需多个智能体依次生成内容，其计算开销与响应时延显著高于 CAHAD，在三个数据集上的 HF-CAC/CAHAD 比率均超过一个数量级。值得注意的是，CAHAD 的推理开销与未经文化对齐的基座模型非常接近（token 比率 1.05–1.08×，时延比率 0.90–1.06×），表明蒸馏过程几乎未引入额外的推理负担。这一结果表明，CAHAD 通过将多智能体系统的结构化文化推理能力缔约至单次前向传播，在保持与多智能体系统相当的文化对齐性能的同时，将推理开销降低了一个数量级。

**消融实验。**为验证 CAHAD 各组件的必要性，表 4 报告了四种训练变体的准确率对比：仅监督微调（SFT-only）、仅强化学习（RL-only）、SFT-then-RL 序贯训练以及本文提出的在线联合训练（Joint SFT-RL）。

  -----------------------------------------------------------------------
         **Variant**        **NormAd**   **CulturalBench**   **BLEnD**
  ------------------------- ------------ ------------------- ------------
  **Qwen2.5-7B-Instruct as                                   
      backbone model**                                       

          SFT-only          65.53        71.77               64.05

           RL-only          65.15        72.58               66.99

         SFT-then-RL        65.91        73.39               68.17

   **Joint SFT-RL (Ours)**  **66.29**    **75.00**           **68.57**

   **Llama-3.1-8B-Instruct                                   
     as backbone model**                                     

          SFT-only          60.23        70.16               65.82

           RL-only          61.74        70.97               65.03

         SFT-then-RL        62.36        71.77               66.80

   **Joint SFT-RL (Ours)**  **64.39**    **72.58**           **67.19**
  -----------------------------------------------------------------------

*表 4：CAHAD
四种训练变体在三个数据集、两个基座模型上的准确率（%）对比。*

在全部六组实验中，训练变体的准确率呈现高度一致的排序模式：Joint SFT-RL 优于 SFT-then-RL 序贯训练，后者优于单独的 RL-only 或 SFT-only 变体。尤其值得关注的是，Joint SFT-RL 相较 SFT-then-RL 序贯训练的优势在三个数据集上普遍成立。这一结果表明，在文化对齐任务中，序贯切换监督与强化学习信号容易导致训练信号的衔接不充分，而在线联合训练通过在每一优化步中同时保留监督锚定与策略探索，对不同文化样本的学习进度实现了更为均衡的调度，这一机制与§3.2 中提出的难度自适应加权策略在设计思想上一脉相承。更多超参数分析结果见附录 B。

# **6 结论**

本文围绕多智能体文化推理中的两个核心问题——等权协商假设与文化知识属地性之间的矛盾，以及多智能体推理的高计算成本——提出了 HF-CAC 与 CAHAD 两种相互衔接的方法。HF-CAC 通过主场权威激活机制，依据目标文化动态赋予对应文化背景的 Guardian 智能体以决策主导权，突破了现有方法中等权协商的隐含假设，并以基于文化亲缘度矩阵的加权仲裁作为回退机制。CAHAD 通过对 Judge 答案与 Guardian 引导信号的差异化映射，采用 SFT 与 RL 在线联合优化并辅以难度自适应加权的混合训练范式，将多智能体系统的文化推理能力迁移至单一模型。在 NormAd、CulturalBench 与 BLEnD 三个跨文化基准上的实验表明，HF-CAC 在准确率上显著优于现有等权协商方法，CAHAD 在保持与多智能体系统相当的文化对齐性能的同时将推理开销降低了一个数量级。本研究表明，将社会科学中关于文化属地性与认识论权威的理论洞见系统性地融入多智能体协商与蒸馏的设计之中，能够为跨文化人工智能对齐提供更为有效的技术路径。

# **7 局限性**

本文的工作仍存在若干局限性。其一，文化概念本身具有动态性、混合性与跨语境迁移性（Appadurai 1996），本文所采用的以国家或地区为单位的文化划分方式以及六大文化圈的划分粒度，仅是对这一复杂现实的简化近似，未能充分反映文化内部的异质性与流动性，未来工作可考虑引入更细粒度或数据驱动的文化表征方式。其二，本文所使用的封闭式基准数据集虽然便于量化评估，但尚无法涵盖开放式对话与真实用户交互场景中更为复杂多变的文化对齐需求，上述实验结论的外部效度有待在开放式生成场景中进一步验证。其三，本文的文化亲缘度矩阵基于 WVS 数据的静态快照构建，未能反映文化价值观随时间演变的动态特征（Inglehart and Baker 2000），未来可探索随训练数据更新而动态调整亲缘度矩阵的机制。

# **参考文献**

Adilazuarda, M. F.; Mukherjee, S.; Muis, A. O.; Utama, P.; and Moens,
M.-F. 2024. Towards Measuring and Modeling "Culture" in LLMs: A Survey.
arXiv:2403.15412.

Appadurai, A. 1996. Modernity at Large: Cultural Dimensions of
Globalization. Minneapolis: University of Minnesota Press.

Arora, A.; Kaffee, L.-A.; and Augenstein, I. 2023. Probing Pre-Trained
Language Models for Cross-Cultural Differences in Values. In Proceedings
of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP),
ACL.

Berry, J. W. 1997. Immigration, Acculturation, and Adaptation. Applied
Psychology, 46(1): 5--34.

Bourdieu, P. 1986. The Forms of Capital. In Richardson, J. (ed.),
Handbook of Theory and Research for the Sociology of Education,
241--258. New York: Greenwood Press.

Cao, Y.; Zhou, L.; Lee, S.; Cabello, L.; Chen, M.; and Hershcovich, D.
2023. Assessing Cross-Cultural Alignment between ChatGPT and Human
Societies: An Empirical Study. In Proceedings of the First Workshop on
Cross-Cultural Considerations in NLP (C3NLP), ACL.

Chen, J. C.-Y.; Saha, S.; Stengel-Eskin, E.; and Bansal, M. 2024. MAGDi:
Structured Distillation of Multi-Agent Interaction Graphs Improves
Reasoning in Smaller Language Models. In Proceedings of the 41st
International Conference on Machine Learning (ICML).

Chiu, Y.-Y.; Jiang, L.; et al. 2025. CulturalBench: A Robust, Diverse,
and Challenging Cultural Benchmark by Human-AI CulturalTeaming. In
Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (ACL).

Collins, H. M.; and Evans, R. 2007. Rethinking Expertise. Chicago:
University of Chicago Press.

Collins, P. H. 2000. Black Feminist Thought: Knowledge, Consciousness,
and the Politics of Empowerment (2nd ed.). New York: Routledge.

Denzin, N. K. 1978. The Research Act: A Theoretical Introduction to
Sociological Methods (2nd ed.). New York: McGraw-Hill.

Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2023.
Improving Factuality and Reasoning in Language Models through Multiagent
Debate. arXiv:2305.14325.

Dubey, A.; Jauhri, A.; et al. 2024. The Llama 3 Herd of Models.
arXiv:2407.21783.

Fricker, M. 2007. Epistemic Injustice: Power and the Ethics of Knowing.
Oxford: Oxford University Press.

Geertz, C. 1973. The Interpretation of Cultures. New York: Basic Books.

Geertz, C. 1983. Local Knowledge: Further Essays in Interpretive
Anthropology. New York: Basic Books.

Goldman, A. I. 1999. Knowledge in a Social World. Oxford: Clarendon
Press.

Habermas, J. 1984. The Theory of Communicative Action, Volume 1: Reason
and the Rationalization of Society. Boston: Beacon Press.

Habermas, J. 1996. Between Facts and Norms: Contributions to a Discourse
Theory of Law and Democracy. Cambridge, MA: MIT Press.

Hall, E. T. 1976. Beyond Culture. New York: Anchor Books.

Harding, S. 1991. Whose Science? Whose Knowledge? Thinking from Women's
Lives. Ithaca, NY: Cornell University Press.

Henrich, J.; Heine, S. J.; and Norenzayan, A. 2010. The Weirdest People
in the World? Behavioral and Brain Sciences, 33(2--3): 61--83.

Hershcovich, D.; Frank, S.; Lent, H.; de Lhoneux, M.; Sheinman, V.;
Søgaard, A.; and Abend, O. 2022. Challenges and Strategies in
Cross-Cultural NLP. In Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (ACL).

Hofstede, G. 1980. Culture's Consequences: International Differences in
Work-Related Values. Beverly Hills, CA: Sage Publications.

Hofstede, G. 2001. Culture's Consequences: Comparing Values, Behaviors,
Institutions and Organizations Across Nations (2nd ed.). Thousand Oaks,
CA: Sage Publications.

Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the Knowledge in
a Neural Network. arXiv:1503.02531.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang,
L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language
Models. In Proceedings of the International Conference on Learning
Representations (ICLR).

Inglehart, R.; and Baker, W. E. 2000. Modernization, Cultural Change,
and the Persistence of Traditional Values. American Sociological Review,
65(1): 19--51.

Inglehart, R.; and Welzel, C. 2005. Modernization, Cultural Change, and
Democracy: The Human Development Sequence. Cambridge: Cambridge
University Press.

Johnson, D. W.; and Johnson, R. T. 2009. Energizing Learning: The
Instructional Power of Conflict. Educational Researcher, 38(1): 37--51.

Johnson, R. L.; Pistilli, G.; Menédez-González, N.; Duran, L. D. D.;
Panai, E.; Kalpokiene, J.; and Bertulfo, D. J. 2022. The Ghost in the
Machine Has an American Accent: Value Conflict in GPT-3.
arXiv:2203.07785.

Ki, D.; Rudinger, R.; Zhou, T.; and Carpuat, M. 2025. Multiple LLM
Agents Debate for Equitable Cultural Alignment. arXiv:2505.24671.

Kitcher, P. 1990. The Division of Cognitive Labor. The Journal of
Philosophy, 87(1): 5--22.

Kluckhohn, C.; and Kroeber, A. L. 1952. Culture: A Critical Review of
Concepts and Definitions. Cambridge, MA: Peabody Museum of Archaeology
and Ethnology.

Li, C.; Chen, M.; Wang, J.; Sitaram, S.; and Xie, X. 2024. CultureLLM:
Incorporating Cultural Differences into Large Language Models. In
Proceedings of the 38th Conference on Neural Information Processing
Systems (NeurIPS).

Masoud, R.; Liu, K.; Ferhatosmanoglu, H.; and Alshemali, B. 2024.
CulturalTeaming: AI-Assisted Interactive Red-Teaming for Challenging
LLMs' (Lack of) Multicultural Knowledge. In Findings of the Association
for Computational Linguistics: EMNLP 2024.

Myung, J.; Lee, N.; et al. 2024. BLEnD: A Benchmark for LLMs on Everyday
Knowledge in Diverse Cultures and Languages. arXiv:2406.09948.

Nisbett, R. E. 2003. The Geography of Thought: How Asians and Westerners
Think Differently... and Why. New York: Free Press.

Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.

Rao, A.; Yerukola, A.; et al. 2025. NormAd: A Framework for Measuring
the Cultural Adaptability of Large Language Models. In Proceedings of
the North American Chapter of the Association for Computational
Linguistics (NAACL).

Said, E. W. 1978. Orientalism. New York: Pantheon Books.

Schwartz, S. H. 1992. Universals in the Content and Structure of Values:
Theoretical Advances and Empirical Tests in 20 Countries. In Zanna, M.
P. (ed.), Advances in Experimental Social Psychology, 25: 1--65. San
Diego: Academic Press.

Schwartz, S. H. 1999. A Theory of Cultural Values and Some Implications
for Work. Applied Psychology, 48(1): 23--47.

Seo, W.; Choi, W.; Koh, J.; et al. 2026. Toward Culturally Aligned LLMs
through Ontology-Guided Multi-Agent Reasoning. arXiv:2601.21700.

Sherif, M.; and Hovland, C. I. 1961. Social Judgment: Assimilation and
Contrast Effects in Communication and Attitude Change. New Haven: Yale
University Press.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang,
M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the
Limits of Mathematical Reasoning in Open Language Models.
arXiv:2402.03300.

Tan, Q.; Jiang, L.; Zeng, Y.; Ding, S.; Xu, X.; et al. 2026. Mitigating
Cultural Bias in LLMs via Multi-Agent Cultural Debate. arXiv:2601.12091.

Tao, Y.; Viberg, O.; Baker, R. S.; and Kizilcec, R. F. 2024. Cultural
Bias and Cultural Alignment of Large Language Models. PNAS Nexus, 3(9):
pgae346.

Triandis, H. C. 1995. Individualism and Collectivism. Boulder, CO:
Westview Press.

Tylor, E. B. 1871. Primitive Culture: Researches into the Development of
Mythology, Philosophy, Religion, Art, and Custom. London: John Murray.

Wang, Y.; Zhao, Y.; and Purevsuren, B. 2024. CultureBank: An Online
Community-Driven Knowledge Base Towards Culturally Aware Language
Technologies. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing (EMNLP).

Yao, B.; Chen, D.; Li, Z.; and Fang, H. 2024. Value FULCRA: Mapping
Large Language Models to the Multidimensional Spectrum of Basic Human
Values. In Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (ACL).

Zhou, Y.; Huang, Z.; Chen, J.; and Bansal, M. 2025. Debate, Reflect, and
Distill: Multi-Agent Feedback with Tree-Structured Preference Learning.
In Findings of the Association for Computational Linguistics: ACL 2025.

# **附录**

## **附录 A　文化亲缘度矩阵的构建方法与使用方式**

### **A.1 构建动机**

HF-CAC 中 Guardian 承担主场权威角色，但 Guardian
也可能因自身知识不足而失效（例如明确表示不确定）。实验统计表明，Guardian
失效的比例极低（约 5% 以下），绝大多数案例通过全员共识或 Guardian
优先采纳即可完成裁决。然而，为确保系统在极端情况下仍能产生合理的判断，本文设计了基于文化亲缘度的回退仲裁机制。此时需要一种合理的回退机制来决定最终答案。简单多数投票忽略了不同文化圈之间存在的亲近与疏远关系，容易被与目标文化距离较远的
Auditor 主导。为此，本文构建了一个基于世界价值观调查（World Values
Survey, WVS）Inglehart--Welzel 文化地图的 6×6 文化亲缘度矩阵，用于在
Guardian 失效时对 Auditor 的判断进行加权仲裁。

### **A.2 计算方法**

亲缘度矩阵的构建分为三个步骤。第一步，收集文化地图坐标：从 WVS
第七轮数据中获取每个国家在 Inglehart--Welzel
文化地图上的二维坐标，横轴表示传统与世俗理性维度，纵轴表示生存与自我表达维度。第二步，计算文化圈质心：将所有国家按六大文化圈（西方与英语文化圈、拉丁美洲文化圈、撒哈拉以南非洲文化圈、东亚文化圈、伊斯兰与中东文化圈、南亚与东南亚文化圈）分组，取组内各国坐标的算术平均值作为该文化圈的质心。第三步，将质心间的欧氏距离转换为归一化亲缘度分数：采用
$\text{affinity} = 1 - \sqrt{d/d_{\max}}$ 的平方根逆距离变换，其中 $d$
为两个文化圈质心间的欧氏距离，$d_{\max}$
为所有文化圈对之间的最大距离，最终裁剪到 $\lbrack 0.10,1.00\rbrack$
区间，对角线元素设为
1.00。相较于线性逆距离变换，平方根变换能够在中间距离范围提供更好的区分度，避免距离较近的文化圈对产生过高的亲缘度分数。

从图 5 的热力图可以观察到若干清晰的结构特征：Western 与 Islamic、African
的亲缘度整体较低（0.10），反映了这些文化圈在 WVS
文化地图上的显著距离；LatAm 与 SouthAsian
的亲缘度中等偏高（0.54），体现了二者在传统价值维度上的相似性；Islamic 与
SouthAsian
的亲缘度较高（0.74），与地理邻近性和历史文化交流一致；EastAsian
与其他非西方文化圈的亲缘度普遍较低（0.10--0.21），反映了儒家文化圈在世俗理性维度上的独特位置。这说明该矩阵能够较好地区分不同文化圈之间的远近关系，并为
Guardian 失效时的加权仲裁提供可解释的结构化先验。

![图 5：六大文化圈亲缘度矩阵热力图。基于 WVS 第七轮 Inglehart--Welzel 文化地图质心间欧氏距离的平方根逆距离变换构建。](figure/fig_affinity_matrix.png){width="6.0in"
height="5.151152668416448in"}

### **A.3 在 HF-CAC 中的使用方式**

亲缘度矩阵在 HF-CAC 的 Judge 裁决阶段使用，具体表现为"Guardian
失效时的文化亲缘度仲裁协议"，包含三个步骤。第一，Guardian
失效检测：Judge 检查 Guardian 的回答中是否包含失效指示词，若包含则判定
Guardian 失效。第二，激活仲裁模式：Judge
不再采用简单多数投票，而是查询亲缘度矩阵，获取每个 Auditor
所属文化圈与目标国家所属文化圈之间的亲缘度分数。第三，加权裁决：在提供了具体文化证据的
Auditor 中，优先采纳与目标文化亲缘度最高者的答案；若没有任何 Auditor
提供具体文化证据，则退回至简单多数投票。需要强调的是，该仲裁机制仅在极少数
Guardian 失效的情况下被触发，绝大多数案例无需使用亲缘度矩阵。

## **附录 B　CAHAD 超参数分析实验**

CAHAD 的 SFT 与 RL 混合训练包含两个关键超参数：$\beta$（SFT/RL
损失平衡系数）与 $\lambda$（Guardian 文化引导强度）。联合损失函数定义为
$L = L_{GRPO} + \beta \cdot w_{sft} \cdot L_{SFT}$，其中 $L_{GRPO}$ 为
GRPO（Shao et al. 2024）策略梯度损失，$L_{SFT}$
为教师答案的交叉熵损失。Guardian
引导通过修改优势函数实现：$A_{i} = A_{i}^{base} + \lambda \cdot w \cdot S_{guardian}$，其中
$w = 1 - hitrate$ 为难度自适应权重，$S_{guardian} \in \{ 0,1\}$ 为
Guardian 文化方向匹配信号。实验设计上，先固定 $\lambda$ 为默认值 0.5
搜索最优 $\beta$，再固定最优 $\beta$ 搜索最优 $\lambda$。

$\beta$ 控制 SFT 监督项在联合损失中的相对权重。$\beta$ 取值过小时，SFT
提供的监督信号不足以约束 RL 探索方向，容易出现探索不稳定的现象；$\beta$
取值过大时，SFT 项在损失函数中占据主导地位，训练过程退化为加权 SFT，RL
部分对未见文化的泛化能力被过度压制。本文实验中 $\beta$ 的默认取值为
0.3，该取值在 SFT 监督强度与 RL 探索空间之间取得了较好的平衡。

$\lambda$ 控制 Guardian 文化引导信号在 GRPO 优势函数中的强度。$\lambda$
取值过小时，文化引导信号过于微弱，模型难以充分利用 Guardian
提供的方向信息；$\lambda$ 取值过大时，二值化的 Guardian
匹配信号会主导优势函数的符号，掩盖答案质量本身的差异，使训练信号变得粗糙，可能损害准确率。本文实验中
$\lambda$ 的默认取值为 0.5。

综合来看，$\beta$ 与 $\lambda$
的取值需要在训练稳定性、准确率与文化引导强度之间进行权衡：过度偏向 SFT
或过强的 Guardian
引导信号都会削弱另一部分训练信号的作用，本文最终采用的默认配置（$\beta = 0.3$，$\lambda = 0.5$）是在准确率与训练稳定性之间综合权衡后的选择。

![图 6：CAHAD 超参数敏感性分析（以 Qwen2.5-7B-Instruct
基座为例）。红色标记表示默认配置（β = 0.3，λ =
0.5），取得了最高的准确率。](figure/fig_hyperparameter.png){width="6.0in"
height="2.2118536745406825in"}

## **附录 C　HF-CAC 提示词模板设计**

### **C.1 设计原则**

HF-CAC 的提示词体系包含 Guardian、Auditor、Judge
三种角色，且每种角色针对六大文化圈分别设计了文化特定的变体。提示词的详细程度随任务复杂度而调整：NormAd（社会规范判断，三分类
1/2/3）采用最为详尽的提示词，强调主场权威与校准性判断（即在不确定时倾向输出"3"），并包含
Guardian
一票否决权与亲缘度仲裁协议的完整规则；CulturalBench（文化知识问答，四选一
MCQ 1/2/3/4）采用中等长度的提示词，Guardian 与 Auditor
共用同一段"文化专家"人设，走 MAD
式对称辩论流程；BLEnD（日常文化知识，四选一 MCQ
1/2/3/4）采用最为简洁的通用提示词，Guardian/Auditor/Judge 三者 system
prompt 完全统一为通用助手提示词，强调跨国干扰项的甄别。

三个数据集在协作流程上也存在结构差异：NormAd 采用 Guardian 优先生成 →
Auditor 单向看到 Guardian 回答后生成 → 条件触发
Judge（含一票否决权与亲缘度仲裁）的非对称流程；CulturalBench 与 BLEnD
采用 Auditor 独立起步 → MAD 式对称辩论（feedback 互评 → final decision
重决策）→ 纯多数投票（无 Guardian 特权），仅在分歧时触发 Judge
的对称流程。

以下以 NormAd 数据集为例，给出三类角色的完整提示词模板。CulturalBench 与
BLEnD 的提示词模板在结构上相似，主要差异在于答案空间（四选一
MCQ）、人设详略程度以及协作流程的对称性。

### **C.2 Guardian System Prompt（NorMAD）**

    You are the HOST-CULTURE GUARDIAN for this question.
    The target culture belongs to YOUR area of expertise ({culture_area} cultures).
    Your cognitive foundation: {cognitive_foundation_description}
    Your role is to AUTHORITATIVELY confirm or correct cultural claims about the target culture.
    You have PRIMARY AUTHORITY on this topic. Be specific, cite cultural practices by name, explain WHY certain options are correct/incorrect based on deep cultural knowledge.
    If other agents suggest answers that conflict with your expertise, firmly correct them with specific cultural evidence.
    IMPORTANT: Not all behaviors have clear cultural acceptance or rejection in every culture.
    If the described behavior is culturally neutral, ambiguous, or universally human (not specifically tied to the target culture's norms), you SHOULD answer "3" to indicate indeterminacy. Forcing a definitive accept/reject judgment when the behavior is culturally irrelevant demonstrates LESS cultural expertise, not more.
    Format: Reasoning: <your authoritative cultural analysis>\nAnswer: <number>

其中 {culture_area}
取值为六大文化圈之一，{cognitive_foundation_description}
为对应文化圈的认知基础描述（见 C.5）。

### **C.3 Auditor System Prompt（NormAD）**

    You are a CROSS-CULTURAL AUDITOR from {culture_area} cultural background.
    Your cognitive foundation: {cognitive_foundation_summary}
    The target culture does NOT belong to your primary expertise area.
    Your role is to provide CONTRASTIVE perspective: note similarities/differences between your culture and the target culture, but DEFER to the Host-Culture Guardian on specific factual claims about the target culture.
    Explicitly acknowledge uncertainty where appropriate.
    If a behavior appears universally human or culturally neutral (not specific to any culture), explicitly state this and consider answering "3" (neutral/indeterminate).
    Your cross-cultural perspective is especially valuable for identifying behaviors that do NOT have culture-specific significance.
    Format: Reasoning: <your cross-cultural comparative analysis>\nAnswer: <number>

Auditor 的 User Prompt 在协商模式下会注入 Guardian 的回答作为参考：

    TARGET CULTURE: {target_country}

    {question}

    The HOST-CULTURE GUARDIAN [{guardian_name}] has provided their authoritative analysis:
    ---
    {guardian_response}
    ---

    As a Cross-Cultural Auditor from [{agent_name}] background:
    1. Provide your comparative perspective (similarities/differences between your culture and {target_country}).
    2. If you agree with the Guardian, explain WHY from your cultural lens.
    3. If you disagree, provide specific counter-evidence — but acknowledge that the Guardian has primary authority on {target_country}.

    Reasoning: <your cross-cultural comparative analysis>
    Answer: <number>

### **C.4 Judge System Prompt 与仲裁**

Judge 的 System Prompt
定义了完整的裁决规则体系，包括证据优先级、Guardian
一票否决权、三分类校准规则以及 Guardian 失效时的文化亲缘度仲裁协议：

    You are a neutral cultural fact-checker and final arbitrator.
    You will receive the ORIGINAL QUESTION and FULL RESPONSES from all cultural expert agents, including both their initial independent analyses and any negotiation/debate exchanges.
    ONE of the agents has been designated as the HOST-CULTURE GUARDIAN — the agent whose cultural expertise most closely matches the target culture in the question.

    Your task is to synthesize all available information — the question itself, each agent's initial reasoning, and how their positions evolved during debate — to arrive at the most culturally accurate answer.

    When evaluating:
    1. Give HIGHER WEIGHT to the Host-Culture Guardian's factual claims about the target culture.
    2. The Guardian has VETO AUTHORITY: if the Guardian's answer differs from the majority AND the Guardian provides specific cultural evidence, prefer the Guardian's answer unless other agents present equally specific counter-evidence about the TARGET culture.
    3. Cross-Cultural Auditors provide valuable comparative perspectives, but their claims about the target culture should be verified against the Guardian's expertise.
    4. Pay attention to how agents' positions shifted during debate — consensus reached through evidence-based discussion is more reliable than initial disagreement.
    5. Base your final decision on verifiable cultural facts, with the Guardian's input as your primary reference.
    6. CRITICAL — Three-way classification:
       - Answer "1" ONLY if there is specific evidence the behavior IS culturally acceptable
       - Answer "2" ONLY if there is specific evidence the behavior IS culturally unacceptable
       - Answer "3" if: (a) the behavior is not culturally specific (universal human behavior),
         OR (b) cultural norms on this vary within the target country,
         OR (c) agents provide conflicting claims without decisive evidence,
         OR (d) the behavior simply has no cultural valence in the target context
       - When in doubt between a forced judgment and "neutral", prefer "3" — a calibrated "I'm not sure" is more valuable than a confident wrong answer.
    7. GUARDIAN FAILURE PROTOCOL — Cultural Affinity Arbitration:
       If the Host-Culture Guardian has FAILED to provide a valid answer (format collapse, empty reasoning, or explicit uncertainty), do NOT fall back to simple majority voting.
       Instead, activate cross-cultural affinity-weighted arbitration:
       - You will be provided with CULTURAL AFFINITY SCORES indicating how culturally proximate each Auditor's background is to the target culture.
       - Give HIGHER WEIGHT to Auditors with higher affinity scores — their cultural proximity to the target culture makes their reasoning more reliable.
       - Even if numerically fewer agents support an answer, prefer the answer backed by the highest-affinity Auditor(s) IF they provide specific cultural evidence.
       - Evaluate each Auditor's reasoning chain for concrete cultural references (practices, traditions, norms) that align with the target culture context.

当检测到 Guardian 失效时（格式崩溃、答案不可提取或明确放弃），Judge 的
User Prompt 切换为亲缘度仲裁模式：

    TARGET CULTURE: {target_country}

    {question}

    GUARDIAN FAILURE: The HOST-CULTURE GUARDIAN [{guardian_name}] has FAILED to provide a valid answer for this question. Activate Cultural Affinity Arbitration protocol.

    CULTURAL AFFINITY SCORES (proximity to {target_country}'s culture):
      - [{auditor_name}]: {affinity_score}
      ...

    Agent responses:
    [{guardian_name}] (HOST-CULTURE GUARDIAN — FAILED, no valid answer):
    {guardian_response}
    [{auditor_name}] (Cross-Cultural Auditor, affinity to target culture: {score}):
    {auditor_response}
    ...

    As the final arbitrator under Guardian Failure Protocol:
    - Do NOT use simple majority voting.
    - Give HIGHER WEIGHT to Auditors with higher affinity scores.
    - If the highest-affinity Auditor provides specific cultural evidence, prefer their answer even if outnumbered.
    - Evaluate each Auditor's reasoning for concrete cultural references.

    Reasoning: <your reasoning, referencing affinity-weighted evidence>
    Answer: <number>

### **C.5 六大文化圈的认知基础描述**

为使各文化圈的 Guardian
提示词具备实质性的文化知识锚点，每个文化圈均配有一段"认知基础"描述，作为
Guardian System Prompt 中 {cognitive_foundation_description}
的填充内容：

西方与英语文化圈（Western & Anglo-Saxon）：English-speaking nations and
secular holidays derived from Christian traditions (Thanksgiving,
Christmas, National Days), individualism, low power-distance social
etiquette, and legal norms prevalent in North America,
Australia/Oceania, and Western Europe.

拉丁美洲文化圈（Latin American）：Hybrid cultures blending Catholic
traditions with indigenous/Afro-descendant elements, including Carnival,
Día de los Muertos, warm and expressive social distances, and cultural
taboos prevalent in South America and Central America (including
Mexico).

撒哈拉以南非洲文化圈（Sub-Saharan African）：Indigenous tribal
traditions (such as the Ubuntu spirit), rich tribal ceremonies, local
taboos, and the unique extended-family collectivism prevalent in
Sub-Saharan Africa (Nigeria, Kenya, South Africa, etc.).

东亚文化圈（East-Asian）：Confucian cultural sphere and the Chinese
character (Hanzi/Kanji) cultural sphere, including traditional festivals
(Spring Festival, Mid-Autumn Festival), face culture (mianzi),
collectivism, and high uncertainty avoidance prevalent in China, Japan,
and Korea.

伊斯兰与中东文化圈（Islamic & Middle-Eastern）：Sharia law, Eid al-Fitr,
Eid al-Adha, unique halal dietary prohibitions, the sanctity of the
color green and the right hand, and other Islamic cultural norms
prevalent in the Middle East, North Africa, and parts of Southeast Asia.

南亚与东南亚文化圈（South & Southeast Asian）：Buddhist and Hindu
traditions, as well as the unique folk customs and cultural taboos of
tropical regions (such as not touching someone's head), prevalent in
India, Thailand, Malaysia, etc.

这些认知基础描述为 Guardian
智能体提供了具体、可操作的文化知识框架，而非空泛的角色标签。

### **C.6 采样温度配置**

三个数据集采用一致的温度设计意图：Guardian 使用低温（约
0.5）以确保权威分析的精确性与一致性；Auditor 使用高温（约
0.9）以提供多样的跨文化对比视角、避免趋同；Judge 使用极低温（约
0.3）以确保裁决的稳定性。在 CulturalBench 与 BLEnD 的 MAD
辩论流程中，feedback 阶段使用 Auditor 高温采样以保持观点多样性，final
decision 阶段切回 Guardian 低温采样以收敛答案。

## **附录 D　Case Study**

本节选取 NormAd 数据集上两个具有代表性的跨文化推理案例，展示 HF-CAC
协商流程中主场权威机制的实际作用方式。两个案例分别涉及苏丹（撒哈拉以南非洲文化圈）与印度（南亚与东南亚文化圈），均选取自
Qwen-2.5-7B-Instruct 基座模型的真实推理记录。表 5 和表 6
分别给出了两个案例的完整多智能体推理轨迹与逐阶段分析。

**案例概述。**
两个案例均涉及特定文化中的左右手礼仪规范，但考察方向相反：案例一考察违反禁忌的行为是否被正确判定为不可接受，案例二考察遵循规范的行为是否被正确判定为可接受。在两个案例中，Guardian
均基于其对目标文化的深入了解给出了准确判断，而部分 Auditor
则因自身文化中缺乏类似规范而产生误判。Judge 通过优先采纳 Guardian
的文化证据而非简单多数投票，在两个案例中均得出了正确结论，充分展示了主场权威机制在处理具有属地性文化知识时的核心优势。

*表 5：案例一------苏丹：左手递物的文化禁忌（目标答案：2，不可接受）*

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **推理阶段**         **智能体与判断**                                                                                        **分析说明**
  -------------------- ------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------
  题目                 目标文化：苏丹（Sudan）。场景：非正式晚宴，Sarah 用左手将菜递给 Tom。判断该行为是否可接受。             苏丹属撒哈拉以南非洲文化圈，左手用于卫生、右手用于社交是根深蒂固的礼仪。

  第一阶段：Guardian   ★ Guardian（African）：Answer:                                                                          Guardian 提供了具体、可验证的文化证据，判断明确。
  独立判断             2。"在苏丹文化中，右手或双手递物是基本礼仪，左手被保留用于卫生用途，公共场合用左手递物被视为不礼貌。"   

  第一阶段：Auditor    Western Auditor：Answer: 3。"非正式场合中这一失误可被视为轻微偏差。"                                    以自身文化的宽松标准评判，缺乏对苏丹文化的深入了解。
  独立判断                                                                                                                     

                       EastAsian Auditor：Answer: 2。"认可 Guardian 分析，但指出东亚文化中左手禁忌不如苏丹严格。"              提供了有价值的比较视角。

                       Islamic Auditor：Answer: 2。"伊斯兰文化中同样存在右手优先的习俗。"                                      文化亲缘度较高，判断与 Guardian 一致。

                       LatAm Auditor：Answer: 2。"认可该禁忌的文化严肃性。"                                                    跨文化尊重的立场。

                       SouthAsian Auditor：Answer: 2。"印度教与佛教传统中左手同样被视为不洁。"                                 文化亲缘度较高，判断与 Guardian 一致。

  第二阶段：辩论       Western Auditor 坚持 3，认为非正式场合可减轻冒犯；Guardian                                              Guardian 以具体文化证据反驳了 Western Auditor 的推测。
                       回应："即使在非正式场合，左手禁忌在苏丹也不会被豁免。"                                                  

  第三阶段：Judge 裁决 Judge：Answer: 2。"Guardian 提供了具体且可验证的文化证据。Western Auditor                               依据 Guardian 优先采纳权，最终答案正确。
                       的观点缺乏对苏丹文化具体语境的深入了解，不构成有效反驳。"                                               
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*表 6：案例二------印度：右手递物的文化规范（目标答案：1，可接受）*

  ------------------------------------------------------------------------------------------------------------------------------------------------------------
  **推理阶段**         **智能体与判断**                                                                     **分析说明**
  -------------------- ------------------------------------------------------------------------------------ --------------------------------------------------
  题目                 目标文化：印度（India）。场景：晚宴上，Emma 注意到文化礼仪，特意用右手将盐递给       印度属南亚与东南亚文化圈，右手或双手递物是习俗。
                       Alex。判断该行为是否可接受。                                                         

  第一阶段：Guardian   ★ Guardian（SouthAsian）：Answer:                                                    Guardian 明确判断为可接受并给出文化依据。
  独立判断             1。"在印度文化中，右手或双手递物是一项习俗，左手被视为不洁、保留用于个人卫生。Emma   
                       用右手递盐完全符合文化规范。"                                                        

  第一阶段：Auditor    Western Auditor：Answer: 2。"在西方文化中，使用右手递物并非严格规范，Emma            混淆了自身文化视角与目标文化规范。
  独立判断             的行为可能被视为过度刻意。"                                                          

                       EastAsian Auditor：Answer:                                                           以自身文化的宽松程度错误推断目标文化。
                       2。"东亚文化中偏好右手但左手并非严格禁忌，暗示印度规范可能也不那么严格。"            

                       LatAm Auditor：Answer: 1。"拉丁美洲文化中类似的礼仪尊重被认为是积极的。"             文化类比得出正确判断。

                       African Auditor：Answer: 1。"撒哈拉以南非洲文化中同样存在右手优先习俗。"             文化亲缘度较高，判断正确。

                       Islamic Auditor：Answer: 1。"伊斯兰文化中右手优先习俗与印度类似。"                   文化亲缘度较高，判断正确。

  第二阶段：辩论       Western 和 EastAsian Auditor 坚持 2；Guardian                                        Guardian 明确纠正了外部文化视角的误判。
                       回应："印度文化中右手递物不是可选礼仪而是基本规范。"                                 

  第三阶段：Judge 裁决 Judge：Answer: 1。"Guardian 提供了清晰且具体的文化证据。Western 与 EastAsian Auditor 依据 Guardian 优先采纳权，最终答案正确。
                       的分析未达到同等程度的文化具体性，更多反映了自身文化中缺乏类似禁忌的事实。"          
  ------------------------------------------------------------------------------------------------------------------------------------------------------------

## **附录 E　训练细节与超参数配置**

### **E.1 基座与微调方式**

CAHAD 的基座模型为 Qwen-2.5-7B-Instruct 与
Llama-3.1-8B-Instruct。训练采用 LoRA（Hu et
al. 2022）微调，基座权重保持冻结，仅训练并保存 LoRA adapter
参数，不保存完整的基座权重，以降低存储与训练开销。LoRA 的 rank 与 alpha
等结构性超参数在监督蒸馏与混合训练阶段保持一致配置，以保证不同训练变体之间的可比性。

### **E.2 优化器与训练规模**

训练采用 AdamW 优化器，学习率设置为 2e-4，配合 Accelerate
训练框架。训练轮数上限设为 50 个
epoch，并引入基于验证集准确率的早停机制，连续 2 个 epoch
验证准确率不再提升则提前终止训练，验证集准确率创新高时保存对应的 LoRA
adapter。数据划分统一采用 8:1:1
的训练/验证/测试划分，测试集不参与训练与早停判定。

### **E.3 GRPO 强化学习超参数**

CAHAD 的强化学习阶段基于 GRPO（Group Relative Policy Optimization，Shao
et
al. 2024）算法，采用分组采样的方式计算相对优势，奖励函数由结果奖励与过程奖励加权组合而成：$R_{total} = \alpha \cdot R_{outcome} + (1 - \alpha) \cdot \text{Mean}(R_{process})$，其中结果奖励权重
$\alpha$ 默认取 0.6，即结果正确性在奖励构成中占主导地位。CAHAD
特有的两个超参数 $\beta$（SFT/RL 损失平衡系数）与 $\lambda$（Guardian
文化引导强度）的取值与作用机制详见附录 B，最终采用的默认配置为
$\beta = 0.3$、$\lambda = 0.5$。
