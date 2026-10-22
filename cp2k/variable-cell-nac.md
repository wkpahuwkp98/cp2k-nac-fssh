# 变胞 NAC：为什么必须用局域基组

> 本文说明一件事：**当晶胞随轨迹变化（变胞 / variable cell）时，平面波基组的软件（VASP、Quantum ESPRESSO 等）无法直接计算非绝热耦合，必须换用高斯基或数值原子轨道基组的软件，例如 CP2K。** 然后给出 CP2K 实现中三个必须处理的要点：S 矩阵、cross-time 重叠、相位。

## 结论先行

| 软件 | 基组 | 基函数数目是否随晶胞变化 | 能否做变胞 NAC |
| --- | --- | --- | --- |
| VASP、Quantum ESPRESSO、CASTEP 等 | 平面波 | **会变** | 否 |
| CP2K | GTH 高斯基（可加平面波辅助） | 不变 | **可以** |
| OpenMX、ABACUS、FHI-aims | 数值原子轨道 | 不变 | 可以 |

原因不是精度问题，是**数学上内积无定义**。下面展开。

## 1. 变胞 NAC 在算什么

非绝热耦合的时间离散形式（见 [FSSH.md §5](../FSSH.md)）绕开了对核坐标的梯度，改用相邻两帧波函数的重叠：

$$
d_{ij}(t) \approx \frac{1}{4\Delta t}\Big[\,S_{ji}(t,t+\Delta t) - S_{ij}(t,t+\Delta t)\Big],
\qquad S_{ij}(t,t+\Delta t) = \langle \psi_i(t) \mid \psi_j(t+\Delta t)\rangle
$$

（三点格式再多用一对相邻帧做差分，形式上更平滑，见文末代码对照。）

关键在于：$|\psi_i(t)\rangle$ 和 $|\psi_j(t+\Delta t)\rangle$ 是**在两个不同几何下**求解得到的。当晶胞参数在两步之间发生变化时，这两个波函数属于两个不同的哈密顿量、也可能是两个不同的基组。要让上式有意义，前提是这两个波函数能被放进同一个内积里比较。

## 2. 平面波基组为什么在这里失效

平面波基组的定义是"动能低于截断能 $E_{\text{cut}}$ 的所有平面波"：

$$
\chi_{\mathbf{G}}(\mathbf{r}) = \frac{1}{\sqrt{\Omega}} e^{i\mathbf{G}\cdot\mathbf{r}}, \qquad \frac{\hbar^2 \mathbf{G}^2}{2m} \le E_{\text{cut}}
$$

其中 $\mathbf{G}$ 是倒格矢，由晶胞决定。**晶胞一变，倒格矢整体缩放，同一个 $E_{\text{cut}}$ 下能容纳的平面波数目随之改变。**

于是两帧的波函数维数不同：

$$
|\psi(t)\rangle \in \mathbb{C}^{N_t}, \quad |\psi(t+\Delta t)\rangle \in \mathbb{C}^{N_{t+\Delta t}}, \quad N_t \ne N_{t+\Delta t}
$$

两个维数不同的向量，内积没有定义。强行补零或插值相当于**换了一组基**，此时算出来的不再是"同一个物理态在两帧之间的重叠"，数值没有物理意义。

## 3. 局域基组为什么可以

高斯基与数值原子轨道的基函数挂在原子核上：

$$
\chi_\mu(\mathbf{r}) = R_n(r)\, Y_{lm}(\hat{\mathbf{r}}), \qquad \mathbf{r} \text{ 相对于某个原子中心}
$$

**基函数的数目只由"有哪些原子、用什么基组"决定，与晶胞大小无关。** 变胞只移动基函数的中心（以及周期镜像的排布），不改变基函数集合的规模。因此

$$
\langle \chi_\mu(t) \mid \chi_\nu(t+\Delta t)\rangle
$$

始终有定义，两帧的 MO 系数矩阵可以直接相乘。这就是变胞 NAC 必须换软件的根因。

## 4. 代价：S 矩阵与广义本征值问题

平面波基组是正交归一的，所以 SCF 解的是普通本征值问题 $H\mathbf{c} = \varepsilon \mathbf{c}$。高斯基**不正交**，必须解广义本征值问题：

$$
H\mathbf{c} = \varepsilon S\mathbf{c}, \qquad S_{\mu\nu} = \langle \chi_\mu \mid \chi_\nu \rangle
$$

$S$ 是这个非正交函数空间的**度规**，$S_{\mu\nu}$ 度量两个基函数"不正交的程度"。平面波情形下 $S = I$，问题退化为普通本征值问题——这正是"平面波不需要 S 矩阵"的原因。

CP2K 通过 `&PRINT / &AO_MATRICES` 把 $S$ 打印出来，脚本读入后使用。

MO 之间的重叠则不能直接写 $\mathbf{c}_i^\dagger \mathbf{c}_j$，必须在中间夹上 S：

$$
\langle \psi_i \mid \psi_j \rangle = \mathbf{c}_i^\dagger S\, \mathbf{c}_j
$$

**注意这里是单帧的 S。** 做 NAC 需要的是两帧之间的 cross-time 重叠，见 §6。

## 5. Löwdin 正交化的视角（为什么不必显式构造正交基）

Löwdin 正交化用旧基组构造一组正交归一的新基：$\tilde{\chi} = S^{-1/2}\chi$。在这组基下系数变为 $\tilde{\mathbf{c}} = S^{1/2}\mathbf{c}$，于是

$$
\langle \psi_i \mid \psi_j \rangle = \tilde{\mathbf{c}}_i^\dagger \tilde{\mathbf{c}}_j = \mathbf{c}_i^\dagger S^{1/2} S^{1/2} \mathbf{c}_j = \mathbf{c}_i^\dagger S \mathbf{c}_j
$$

结果与直接夹 S 一致。**实际计算中不需要真的求出 $S^{-1/2}$、也不需要知道 Löwdin 基长什么样**，只要 $S$ 和原始系数 $\mathbf{c}$ 就够。这既省事，也避免了 $S^{-1/2}$ 在近线性相关基组下的数值不稳定。

完整的推导（正交性证明、系数变换、与基组无关性的说明、数值注意）见文末**附录 A**。

## 6. 必须用 cross-time 的 S，不能用单帧 S

这是最容易出错的一步。NAC 需要的是

$$
S_{ij}(t,t+\Delta t) = \mathbf{c}_i(t)^\dagger\, S_{\text{cross}}\, \mathbf{c}_j(t+\Delta t),
\qquad (S_{\text{cross}})_{\mu\nu} = \langle \chi_\mu(t) \mid \chi_\nu(t+\Delta t)\rangle
$$

如果偷懒只用单帧的 $S(t)$，会丢掉**基函数移动项**。用一个极端情形检验：

> 整个体系做完全刚性平移，原子相对位置与电子态都不变。此时真实的 NAC 全部来自基函数随原子移动的那一项；而用单帧 $S$ 计算会得到 **NAC 恒等于 0**。

一个能通过刚性平移检验的实现，才谈得上正确。

### 本仓库的做法：合并体系

CP2K 一次 SCF 只能有一个晶胞，所以不能直接"同时算两帧的基函数"。这里的处理是把两帧原子**拼成一个 2N 原子的体系**写进同一个输入：

- 帧 $t+\Delta t$ 的原子摆到相对帧 $t$ 的最近周期镜像位置
- 两帧共用一个晶胞（可选 $t$、$t+\Delta t$ 或两者的平均）
- 跑一次 SCF，得到 $2N \times 2N$ 的 AO 重叠矩阵
- 只取其中的跨时块 $S_{\text{cross}} = S[\,0{:}n_{\text{ao}},\ n_{\text{ao}}{:}\,]$

> **这是一个近似**：真实的两帧各自有各自的晶胞，合并时必须选定一个。脚本因此提供 `--cell-choice t|tdt|average|all`，选 `all` 会把三种都算一遍，用差值把这项近似**量化**出来，而不是假装它不存在。
>
> 代价也要说清楚：合并体系的 SCF 是整条计算链里最贵的一步。

## 7. 相位：不做修正，结果不可比

即使上述都做对，还有一个坑：**每个离子步 SCF 收敛后得到的 MO 整体相位是随机的**（$\psi_i \to e^{i\theta}\psi_i$ 不改变任何物理量）。相位随机会让相邻帧的重叠矩阵元素符号乱跳，算出的 NAC 也就没有可比性。

处理方式是两层：

1. **参考帧对齐**：用某一帧作为参考，逐帧把每个 MO 的相位对齐到参考帧，得到 $\langle \psi_i(0)|\psi_i(t)\rangle / |\langle \psi_i(0)|\psi_i(t)\rangle|$
2. **链式传播 + 实规范**：相邻帧之间按 $\text{phase}_i(t+\Delta t) = \operatorname{sign}(S_{ii}(t)) \cdot \text{phase}_i(t)$ 逐帧传播，再把相位因子乘回重叠矩阵

这一步没做的话，同一个体系用 VASP 和用 CP2K 算出来的 NAC **完全不可比**——不是因为哪边算错了，而是两边的相位约定不同。

## 8. 概念与代码的对应

| 概念 | 位置 |
| --- | --- |
| 解析 CP2K 打印的 AO 重叠矩阵 | `parse_cp2k_ao_overlap()` |
| $\langle \psi_i \mid \psi_j\rangle = \mathbf{c}_i^\dagger S \mathbf{c}_j$ | `mo_overlap()` |
| 单帧 MO 相位对齐 | `phase_factors()` |
| 由 cross-time 重叠算 NAC | `nac_gaussian()` |
| 合并体系构造（含最近镜像、晶胞选择） | `cmd_build_merged` |
| 重叠矩阵健康检查（对角 $\lvert S_{ii}\rvert$ 应 ≈ 1） | `diag_overlap_health()` |
| V/C 分块最大重叠指派（防止 VBM 与 CBM 误配） | `block_hungarian()` |
| V/C 块间耦合的规范不变量 $\lVert\hbar d_{VC}\rVert_F$ | `vc_block_metrics()` |
| 相位链式传播 / 修正 | `nq_compute_phases()`、`nq_correct_phases()` |
| 二点 / 三点差分格式 | `nq_couplings_levine()`、`nq_couplings_3points()` |

三点差分格式（`nq_couplings_3points`，与 nano-qmflows 一致）：

$$
d_{ji} = \frac{3\big(S_{ji}(t_1) - S_{ij}(t_1)\big) + \big(S_{ij}(t_0) - S_{ji}(t_0)\big)}{4\,\Delta t}
$$

其中 $t_0$ 为 $(t,\,t+\Delta t)$ 的重叠、$t_1$ 为 $(t+\Delta t,\,t+2\Delta t)$ 的重叠。

## 9. 适用边界

- **基组不完备性**：高斯基的完备性不如平面波，NAC 对基组敏感，建议做基组收敛测试。
- **BSSE**：两帧基组重叠时存在基组叠加误差，跨时重叠在原子间距很近时需留意。
- **合并体系的晶胞选择**：见 §6，是显式近似，用 `--cell-choice all` 量化。
- **计算成本**：合并体系 SCF 是瓶颈；脚本按帧对缓存结果，避免重复计算。

---

# 附录 A：Löwdin 正交化与 S 矩阵的推导

## A.1 S 矩阵的定义与性质

给定一组非正交的实基函数 $\{\chi_\mu\}$，重叠矩阵定义为

$$
S_{\mu\nu} = \langle \chi_\mu \mid \chi_\nu \rangle = \int \chi_\mu^*(\mathbf{r})\,\chi_\nu(\mathbf{r})\,\mathrm{d}\mathbf{r}
$$

它有三个性质，后面都要用到：

1. **对称**：$S = S^{\mathsf T}$（内积对称）
2. **正定**：对任意非零系数向量 $\mathbf{v}$，$\mathbf{v}^{\mathsf T}S\mathbf{v} = \langle \sum_\mu v_\mu\chi_\mu \mid \sum_\nu v_\nu\chi_\nu\rangle > 0$。这正是"这组基函数线性无关"的等价表述。
3. **度规**：它把系数空间的内积与函数空间的内积对应起来。对两个由同一组基展开的函数

$$
|\psi_i\rangle = \sum_\mu c_{\mu i}|\chi_\mu\rangle,\quad
|\psi_j\rangle = \sum_\nu c_{\nu j}|\chi_\nu\rangle
$$

有

$$
\langle \psi_i \mid \psi_j \rangle = \sum_{\mu\nu} c_{\mu i}^* \langle \chi_\mu \mid \chi_\nu \rangle c_{\nu j} = \mathbf{c}_i^\dagger S\, \mathbf{c}_j
$$

这就是"不能直接写 $\mathbf{c}_i^\dagger\mathbf{c}_j$"的原因：$S \ne I$ 时，系数向量的普通点积不等于函数的内积。

平面波基组之所以可以偷懒，是因为 $\langle e^{i\mathbf{G}\mathbf{r}}|e^{i\mathbf{G}'\mathbf{r}}\rangle \propto \delta_{\mathbf{G}\mathbf{G}'}$，$S$ 就是单位阵，度规退化为欧氏度规。

## A.2 广义本征值问题

把 $\hat H$ 在同一组基下展开，得到的是

$$
H\mathbf{c} = \varepsilon S\mathbf{c}, \qquad H_{\mu\nu} = \langle \chi_\mu | \hat H | \chi_\nu \rangle
$$

因为 $S$ 正定，可以令 $\mathbf{c} = S^{-1/2}\mathbf{u}$，代入得

$$
H S^{-1/2}\mathbf{u} = \varepsilon S^{1/2}\mathbf{u}
\;\Longrightarrow\;
\underbrace{S^{-1/2} H S^{-1/2}}_{\tilde H}\,\mathbf{u} = \varepsilon\,\mathbf{u}
$$

于是问题化为对 $\tilde H$ 的普通本征值问题。注意 $\tilde H$ 仍是厄米的（$S^{-1/2}$ 厄米且与自身对易），所以本征值 $\varepsilon$ 仍为实数——**非正交基组并不破坏能量的实数性**，只是把求解过程变复杂了。

## A.3 Löwdin 正交化

思路是：用旧基组造一组新的、正交归一的基函数。

$$
\tilde\chi_\mu = \sum_\nu \left(S^{-1/2}\right)_{\nu\mu} \chi_\nu
$$

**正交归一性证明**：直接代入定义

$$
\langle \tilde\chi_\mu \mid \tilde\chi_\nu \rangle
= \sum_{\alpha\beta} \left(S^{-1/2}\right)_{\alpha\mu}^* \underbrace{\langle \chi_\alpha \mid \chi_\beta\rangle}_{S_{\alpha\beta}} \left(S^{-1/2}\right)_{\beta\nu}
$$

$S$ 实对称，$S^{-1/2}$ 也实对称，所以

$$
= \sum_{\alpha\beta} \left(S^{-1/2}\right)_{\mu\alpha} S_{\alpha\beta} \left(S^{-1/2}\right)_{\beta\nu}
= \left(S^{-1/2} S S^{-1/2}\right)_{\mu\nu}
= \left(S^{-1/2} S^{1/2} S^{1/2} S^{-1/2}\right)_{\mu\nu}
= \delta_{\mu\nu}
$$

即新基组的重叠矩阵是单位阵。这一步的关键是 $S$ 与 $S^{-1/2}$ 对易——它们是同一个矩阵的函数，因此可以自由合并指数。

## A.4 系数如何变换

同一个态在两套基下展开：

$$
|\psi_i\rangle = \sum_\mu c_{\mu i}|\chi_\mu\rangle = \sum_\mu \tilde c_{\mu i}|\tilde\chi_\mu\rangle
$$

把 $\tilde\chi_\mu$ 的定义代回去：

$$
\sum_\mu \tilde c_{\mu i}\sum_\nu \left(S^{-1/2}\right)_{\nu\mu}\chi_\nu
= \sum_\nu \Big(\sum_\mu \left(S^{-1/2}\right)_{\nu\mu}\tilde c_{\mu i}\Big)\chi_\nu
$$

与左边逐项比较（基函数线性无关，系数唯一）：

$$
c_{\nu i} = \sum_\mu \left(S^{-1/2}\right)_{\nu\mu}\tilde c_{\mu i}
\;\Longrightarrow\;
\tilde{\mathbf{c}}_i = S^{1/2}\mathbf{c}_i
$$

## A.5 关键结论：重叠与是否正交化无关

在新基下重叠矩阵是单位阵，所以重叠可以直接用系数点积：

$$
\langle \psi_i \mid \psi_j\rangle
= \tilde{\mathbf{c}}_i^\dagger \tilde{\mathbf{c}}_j
= \left(S^{1/2}\mathbf{c}_i\right)^\dagger \left(S^{1/2}\mathbf{c}_j\right)
= \mathbf{c}_i^\dagger S^{1/2} S^{1/2} \mathbf{c}_j
= \mathbf{c}_i^\dagger S\,\mathbf{c}_j
$$

**结果与 §A.1 直接夹 $S$ 完全一致。** 这说明：

- 正交化只是**推导工具**，不是必须执行的步骤；
- 实际计算只需要 $S$ 和原始系数 $\mathbf{c}$，**不需要求出 $S^{-1/2}$，也不需要知道 Löwdin 基函数的显式形式**；
- 因此也不受 $S^{-1/2}$ 数值稳定性的拖累（见 A.7）。

## A.6 推广到 cross-time

NAC 需要的是两帧之间的重叠，把上面的 $S$ 换成跨时的即可：

$$
S_{ij}(t,t+\Delta t) = \mathbf{c}_i(t)^\dagger\, S_{\text{cross}}\, \mathbf{c}_j(t+\Delta t),
\qquad (S_{\text{cross}})_{\mu\nu} = \langle \chi_\mu(t) \mid \chi_\nu(t+\Delta t)\rangle
$$

注意此时 $S_{\text{cross}}$ **不是方阵的对称矩阵**（行属于帧 $t$ 的基组，列属于帧 $t+\Delta t$ 的基组），但两个基组的规模相同，所以它仍是 $n_{\text{ao}}\times n_{\text{ao}}$ 的方阵，只是不满足 $S_{\text{cross}} = S_{\text{cross}}^{\mathsf T}$。这个不对称正是 NAC 的来源——若对称则 §1 的差分为零。

## A.7 数值注意

虽然不必显式求 $S^{-1/2}$，但 $S$ 的条件数仍然重要：

- 高斯基在弥散函数较多、或两帧原子靠得很近时，$S$ 会出现接近 1 的本征值 → **近线性相关**。此时 $S$ 条件数变差，任何涉及 $S^{-1}$ 的操作都会放大误差。
- 本仓库的做法是绕开 $S^{-1}$：只用 $S$ 本身做双线性型 $\mathbf{c}^\dagger S\mathbf{c}$，不做求逆。这也是 §5 结论的实际价值。
- 一个实用的健康检查：单帧情形下应有 $\lVert C^\dagger S C - I\rVert \approx 0$。若明显偏离，说明读入的 $S$ 与波函数不匹配（例如取错了 `&AO_MATRICES` 的段落、或帧的原子顺序不一致）。脚本里的 `diag_overlap_health()` 就是做这件事。

## A.8 一个容易混淆的点

"$S$ 是度规"这句话常被误解成"$S$ 描述了基函数之间的距离"。更准确的说法是：$S$ 定义了**系数空间上的内积**，$\langle \mathbf{u},\mathbf{v}\rangle_S = \mathbf{u}^\dagger S \mathbf{v}$。基函数本身是函数，没有"距离"可言；是系数向量需要借 $S$ 才能正确地对应回函数空间的内积。平面波情形 $S = I$，两种内积恰好重合，所以这个区别平时被掩盖了。

---

## 延伸

- 本仓库 [FSSH.md](../FSSH.md) §5 相位修正——数值实现的来龙去脉
- 时间离散形式的 NAC 与二点/三点差分：[nano-qmflows](https://github.com/SCM-NV/nano-qmflows)（Apache-2.0），本仓库的对应改编见 [NOTICE](NOTICE)
