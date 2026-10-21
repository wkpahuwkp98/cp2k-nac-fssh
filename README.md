# 非绝热载流子动力学理论计算方法

> 一份从 Tully 最小面跳跃（FSSH）推导到 PYXAID 实操参数的中文教程，覆盖 KS 轨道近似、经典路径近似（CPA）、DISH 退相干修正与 VASP 相位修正。

本仓库是一份**学习型笔记 / 教程**，目标是把"面跳跃 + 非绝热分子动力学"这条链路从第一性原理一路推到能真正跑起来的 PYXAID 计算：公式尽量给全、每一步推导的动机都写清楚、数值实现中容易踩的坑（尤其是相位随机性问题）单独成章。

## 内容概览

正文见 **[FSSH.md](FSSH.md)**（GitHub 上可直接渲染，含公式与插图）。

| 章节 | 内容 | 建议 |
| --- | --- | --- |
| 一、最小面跳跃（FSSH）主要推导 | 哈密顿量 → 含时薛定谔方程与波函数展开 → 投影 → 密度矩阵 → 跃迁概率与速度调整，公式 (1)–(57) | 核心理论，建议完整精读 |
| 二、含时密度泛函理论下的面跳跃 | KS 轨道近似（Slater 行列式简化）、经典路径近似 CPA、量子退相干修正 DISH（随机演化算符、退相干时间、随机判断） | 做 PYXAID 计算请读 §2.2、§2.3 |
| 三、结论 | FSSH 的局限性与适用边界 | 快速浏览 |
| 四、PYXAID 设置 | `params[...]` 各关键词含义与取值建议 | 实操时对照查阅 |
| 五、相位修正 | VASP 每个离子步对角化得到的相位随机，直接套公式会引入虚部噪声，给出脚本中的修正算法 | 理解"为什么必须做" |
| 六、全电子校正 | 占位，待补 | — |
| 七、附录：所需量子力学基础 | 波函数、算符、测量、薛定谔方程、全同粒子假设 | 查阅用，不必顺读 |

## 按目的阅读

- **第一次接触 FSSH，想完整走一遍推导**：§1 → §2 → §3。
- **只想跑 PYXAID 出载流子寿命**：§2.2（CPA 与玻尔兹曼修正）→ §2.3（DISH 物理图像，公式略读）→ §4（参数）→ §5（相位修正）。
- **写文章需要引用与公式出处**：直接看文末"参考文献"，并按章节回查原始公式编号。

## 仓库结构

```
.
├── README.md     # 本文件
├── FSSH.md       # 教程正文（主文档）
├── LICENSE       # CC BY 4.0 许可全文
├── fig/          # 正文插图（image1.png … image17.png）
├── cp2k/         # CP2K 高斯基 NAC 计算脚本与输入模板
└── .gitignore    # 排除本地 PDF 参考资料、Python 缓存与运行产物
```

> 说明：原始 PDF 文献仅作为本地写作参考，未纳入仓库；如需阅读请自行从期刊获取。

## CP2K 高斯基 NAC 脚本

`cp2k/` 目录是正文 §2 方法的一个可运行实现：把 NAC 从平面波（VASP / PYXAID）搬到高斯基组（CP2K）。核心是给 overlap 矩阵夹上 AO 重叠：

$$\langle \psi_i(t) \mid \psi_j(t+\Delta t) \rangle = C_i^\dagger S_{\text{cross}} C_j$$

| 文件 | 作用 |
| --- | --- |
| `cp2k_gaussian_nac_parallel.py` | 高斯基 time-overlap NAC 主程序，含 V/C 分块 gauge 搬运、MO 相位对齐、活性窗检查 |
| `cp2k_nac_parallel.sh` | SLURM 驱动脚本，逐帧对提交作业并缓存结果 |
| `input.inp` | CP2K 模板输入（`RUN_TYPE ENERGY`，通过 `&PRINT / &AO_MATRICES` 输出 AO 重叠矩阵供脚本解析） |

**依赖**：Python 3 + NumPy；CP2K 需开启 `&AO_MATRICES` 打印重叠矩阵。

**自检**（验证基组、赝势、MO 重叠与相位对齐是否正确）：

```bash
python cp2k_gaussian_nac_parallel.py selftest --xyz frame.extxyz --frame 0 \
    --basis gth-dzvp-molopt-sr --pseudo gth-pbe --xc pbe
```

**批量计算**：

```bash
python cp2k_gaussian_nac_parallel.py nac --xyz frame.extxyz --frame-t 0 --frame-tdt 1 \
    --dt 1.0 --bmin 160 --bmax 200 --basis gth-dzvp-molopt-sr --pseudo gth-pbe --xc pbe

sbatch cp2k_nac_parallel.sh
```

> `cp2k_nac_parallel.sh` 顶部的 `FRAMES_DIR`、`SBATCH` 分区名、模块加载与环境变量均为作者所在集群的配置，使用前请替换为你自己的；`BMIN/BMAX` 是活性空间窗口，需覆盖占据前沿（HOMO±5 是常用起点）。

## 阅读方式

- **在线**：直接点开 [FSSH.md](FSSH.md)。GitHub 支持 LaTeX 公式渲染，但复杂公式偶有排版偏差。
- **本地（推荐，公式排版最稳）**：用支持 MathJax/KaTeX 的编辑器打开，例如 VS Code + Markdown Preview Enhanced、Typora、Obsidian。

```bash
git clone <本仓库地址>
```

## 贡献

欢迎提 Issue 指正推导错误、补充 PYXAID 实际参数经验，或补全"六、全电子校正"一节。提交 PR 前请保持公式编号与正文引用一致。

## 许可

本仓库的文档内容（`FSSH.md` 及全部插图）采用 **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)**（署名 4.0 国际）协议授权，许可全文见 [LICENSE](LICENSE)。

© 2026 wkpahuwkp98

你可以自由地**共享**（以任何媒介或格式复制、发行）与**演绎**（修改、转换、再创作），包括商业用途，但须遵守：

- **署名**：注明作者 `wkpahuwkp98`、给出本仓库链接，并标明是否作了修改
- **不得增加限制**：不得对他人施加法律条款或技术措施，限制他人行使许可协议允许的权利

建议引用格式：

> wkpahuwkp98. 非绝热载流子动力学理论计算方法[EB/OL]. GitHub, 2026. https://github.com/wkpahuwkp98/fssh-tutorial

若后续在本仓库中加入代码，代码部分建议另行采用 MIT 许可。
