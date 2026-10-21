#!/bin/sh
#SBATCH -J cp2k_nac
#SBATCH -p your_partition
#SBATCH -o job_%j.log
#SBATCH -n 36
# ============================================================
# CP2K 高斯基 变胞 NAC 驱动(小胞测试版)—— 直接吃 VASP .vasp 帧文件
# ============================================================
set -u
rm -rf .nac_notes
NAC_PAIR_CACHE_ROOT="${NAC_PAIR_CACHE_ROOT:-$PWD/pair_cache}"
mkdir -p "$NAC_PAIR_CACHE_ROOT"
echo "host $(hostname)  time $(date)  dir $PWD  nodes $SLURM_JOB_NODELIST"
# 加载你自己集群的 cp2k 环境(如 module load cp2k, 或 source 你的环境脚本)
export OMP_NUM_THREADS=1                   
# ============================ 可配置区 ============================
PYTHON_BIN="python"
NAC_PY="./cp2k_gaussian_nac_parallel.py"
TEMPLATE="./input.inp"                     # cp2k
FRAMES_DIR="./frames"                      # vasp 的 poscar 文件所在目录
FRAME_NAME_FMT="p%05d"           # 帧文件命名格式
PROJECT="PROJECT"                          # 必须和 input.inp 里 &GLOBAL PROJECT 一样
FRAME_START=7001                              # 起始帧
FRAME_END=11001                              # 结束帧
FRAME_STRIDE=1                            # 抽稀步长
GLOBAL_FIRST_FRAME="${GLOBAL_FIRST_FRAME:-1}"

BMIN=122                                   # 活性空间起始 
BMAX=142                                   # 活性空间结束 
MD_DT=1                                  # MD步长
DT_PER_USED_FRAME=""                       # 下面自动 = MD_DT × FRAME_STRIDE
CELL_CHOICE=average                        # build-merged 镜像求和用哪帧胞: t/tdt/average/all
                                           # all = 三种各算一整套, 用差值把这个近似【量出来】。
                                           #   输出: average → run_nac/<sub>/        (原路径不变)
                                           #         t       → run_nac/cell_t/<sub>/
                                           #         tdt     → run_nac/cell_tdt/<sub>/
                                           #   代价: 合并 SCF 跑 3 遍(这是整条链最贵的一步)。
MO_CUBES="on"                              # on=每帧在 &DFT/&PRINT &MO_CUBES 打cube
NHOMO=5                                    # 打几条 HOMO 侧轨道(HOMO, HOMO-1, ... 共 NHOMO 条)
NLUMO=5                                    # 打几条 LUMO 侧轨道(LUMO, LUMO+1, ... 共 NLUMO 条)
CUBE_STRIDE="2 2 2"                        # cube精度
KEEP_CUBE=0                                # 算完 overlap 立刻删 cube
# ───────── python 后处理的成本开关(★这是"CP2K 很快但 python 巨慢"的根源) ─────────
# —— 简并簇 Procrustes(wfn-nac 的第 ② 层对齐)——
DEGEN_THRESH=0.050                         # 能隙 < 该值(eV)的态归为一簇, 簇内做 SVD 子空间对齐,Hungarian 只能修【置换】(两态互换位置); 简并子空间里发生的是【连续正交旋转】, 置换修不了 —— 特征是 swaps=0, 但 |O_ii| 远离 1。0.050=50meV 对应本项目已知的准简并区。设 0 或给 wfn-nac 加 --no-procrustes 可关掉(做对照用)
ANALYSIS_STRIDE=1                         # 每隔几帧做一次"要单帧S"的分析。1=每帧(最贵); 10=推荐; 0=只在第1帧做一次自检
# ── 多核: python 侧的两种并行(互相独立, 可同时开) ─────────────────────────────────────
NAC_THREADS=8                              # 线程数
NAC_JOBS=4                                 # 算overlap的进程数
AO_NDIGITS=12                               # AO overlap 打印位数
KEEP_MERGED_OUT=0                          # 0=NAC 算完立刻删 merged.out 1=保留
KEEP_WFN=0                                  # 0=只留最近2帧的wfn, 1=全部保留
KEEP_OUT=0                                  # 0=只留最近2帧的point.out, 1=全部保留
SCF_MODE="diag"                            # "diag"=直接对角化
                                           # "ot+diag"=先OT收敛, 再DIAG(读OT的wfn)出虚轨道, 两段式;
                                           # "ot"=只跑OT,无虚轨, 不能接 NAC
SPIN="off"                                 # "on"=开自旋极化; "off"=关。
MULTIPLICITY=""                            #
CHARGE=0 

ensure_dir(){ [ -d "$1" ] || mkdir -p "$1"; }

scf_ok(){
    d="$1"
    [ -s "$d/point.out" ] || return 1                        # ① 非空(空 = 起手就崩)
    [ -f "$d/${PROJECT}-RESTART.wfn" ] || return 1           # ② wfn 真的写出来了
    grep -q "PROGRAM ENDED AT" "$d/point.out" || return 1     # ③ CP2K 正常收尾(不是被 kill/写一半)
    grep -q "SCF run converged" "$d/point.out" || return 1    # ④ SCF 收敛了("NOT converged" 不含此串)
    return 0
}
scf_fail(){
    t="$1"; d="$2"; why="$3"
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "✗✗ frame $t 的 SCF 失败: $why"
    [ -f "$d/point.out" ] && { mv "$d/point.out" "$d/point.out.FAILED"; \
        echo "   out 已改名 → $d/point.out.FAILED (保留排查; 且不会再被误判成'已完成')"; }
    echo "   ---- err 末尾 ----";              [ -f "$d/err" ] && tail -n 12 "$d/err"
    echo "   ---- point.out.FAILED 末尾 ----"; [ -f "$d/point.out.FAILED" ] && tail -n 12 "$d/point.out.FAILED"
    echo "════════════════════════════════════════════════════════════════════"
    echo "★ 中止整个作业。带着一帧坏数据继续跑, 后面所有 NAC/Hvib 都是垃圾——而且不会有任何报错。"
    exit 1
}
_dec(){ _n="${1#${1%%[!0]*}}"; printf "%s" "${_n:-0}"; }      # "0008"→"8"  "8"→"8"  "0000"→"0"
step_tag(){ printf "%04d" "$(_dec "$1")"; }
frame_file(){ printf "$FRAMES_DIR/$FRAME_NAME_FMT" "$(_dec "$1")"; }   # 按 FRAME_NAME_FMT 拼帧文件路径
require(){ [ -f "$1" ] || { echo "缺文件: $1"; exit 1; }; }

require "$TEMPLATE"
require "$NAC_PY"

# dt 自动推导(sh 没有浮点, 借 awk)
DT_PER_USED_FRAME=$(awk -v a="$MD_DT" -v b="$FRAME_STRIDE" 'BEGIN{printf "%.10g", a*b}')
echo "dt: 原始 MD 步长 ${MD_DT} fs × 抽稀 FRAME_STRIDE=${FRAME_STRIDE} → 相邻被用帧间隔 ${DT_PER_USED_FRAME} fs"
if [ $(( (FRAME_START - GLOBAL_FIRST_FRAME) % FRAME_STRIDE )) -ne 0 ]; then
    echo "✗✗ (FRAME_START=$FRAME_START − GLOBAL_FIRST_FRAME=$GLOBAL_FIRST_FRAME) 不能被 FRAME_STRIDE=$FRAME_STRIDE 整除。"
    echo "   本段的帧不在全局抽稀网格上, 配对编号会与其他段错位。请调整 FRAME_START 或 FRAME_STRIDE。"
    exit 1
fi
if [ "$FRAME_START" -lt "$GLOBAL_FIRST_FRAME" ]; then
    echo "✗✗ FRAME_START($FRAME_START) < GLOBAL_FIRST_FRAME($GLOBAL_FIRST_FRAME) → PAIR 会是负数。"; exit 1
fi
FRAME_IDS=$(seq "$FRAME_START" "$FRAME_STRIDE" "$FRAME_END")
echo "被用帧: $FRAME_IDS"
for t in $FRAME_IDS; do require "$(frame_file $t)"; done

ensure_dir scf_run; ensure_dir run_nac/normal; ensure_dir run_nac/phase_corr; ensure_dir analysis
ensure_dir Ipr_Overlap/ipr; ensure_dir Ipr_Overlap/overlap
if [ "$MO_CUBES" = "on" ]; then
    ensure_dir Cube_Overlap/eh_dens; ensure_dir Cube_Overlap/eh_amp
    ensure_dir Cube_Overlap/eh_cos;  ensure_dir Cube_Overlap/index
    NSP=1; [ "$SPIN" = "on" ] && NSP=2
    NCUBE=$(( (NHOMO + NLUMO) * NSP ))
    echo "MO_CUBES: 每帧 $NCUBE 个 cube (NHOMO=$NHOMO + NLUMO=$NLUMO, 自旋通道 $NSP), STRIDE=$CUBE_STRIDE"
    echo "  峰值占用 ≈ $((NCUBE * 5))MB/帧(实测 4.3MB/cube@stride4); KEEP_CUBE=$KEEP_CUBE"
    [ "$KEEP_CUBE" -eq 1 ] && echo "  ⚠ KEEP_CUBE=1: cube 不删, 全程累计 ≈ $((NCUBE * 5))MB × 帧数"
fi

# ── python 调用统一入口: 注入 BLAS 线程数 + 解析进程数 ──
PY(){
    python_exe="$PYTHON_BIN"; nac_py="$NAC_PY"
    env OMP_NUM_THREADS="$NAC_THREADS" OPENBLAS_NUM_THREADS="$NAC_THREADS" \
        MKL_NUM_THREADS="$NAC_THREADS" NUMEXPR_NUM_THREADS="$NAC_THREADS" \
        NAC_JOBS="$NAC_JOBS" "$python_exe" "$nac_py" "$@"
}

# ---------- 每帧 SCF ----------
need_S(){                                   # $1 = 第几帧(1基计数, 不是帧号)
    [ "$ANALYSIS_STRIDE" -le 0 ] && { [ "$1" -eq 1 ]; return; }
    [ $(( ($1 - 1) % ANALYSIS_STRIDE )) -eq 0 ]
}

run_scf(){
    t="$1"; is_first="$2"; fidx="${4:-1}"; tag=$(step_tag "$t"); d="scf_run/$tag"
    ensure_dir "$d"
    if need_S "$fidx"; then AO_ARG="--ao-ndigits $AO_NDIGITS"; else AO_ARG="--no-ao-overlap"; fi
    if [ "$SPIN" = "on" ]; then
        SPIN_ARG="--lsd --charge $CHARGE"
        [ -n "$MULTIPLICITY" ] && SPIN_ARG="$SPIN_ARG --multiplicity $MULTIPLICITY"
    else
        SPIN_ARG="--no-lsd --charge $CHARGE"
    fi
    if [ "$is_first" -eq 1 ]; then
        frame_guess="atomic"
    else
        _ptag=$(step_tag "$3")
        [ -f "scf_run/$_ptag/${PROJECT}-RESTART.wfn" ] && cp "scf_run/$_ptag/${PROJECT}-RESTART.wfn" "$d/${PROJECT}-RESTART.wfn"
        frame_guess="restart"
    fi
    if scf_ok "$d"; then echo "SCF done frame $t (已完成且通过校验, 跳过)"; return; fi
    if [ -f "$d/point.out" ]; then
        echo "  frame $t: 有残留 point.out 但【没通过校验】(空/未收敛/没收尾/缺wfn) → 重跑这一帧"
        rm -f "$d/point.out"
    fi
    CUBE_ARG=""
    [ "$MO_CUBES" = "on" ] && CUBE_ARG="--mo-cubes --nhomo $NHOMO --nlumo $NLUMO --cube-stride $CUBE_STRIDE"
    case "$SCF_MODE" in
      diag)
        PY vasp2cp2k --vasp "$(frame_file $t)" --template "$TEMPLATE" \
            --out "$d/cp2k.inp" --guess "$frame_guess" --scf-mode diag $SPIN_ARG $CUBE_ARG $AO_ARG
        [ -s "$d/cp2k.inp" ] || scf_fail "$t" "$d" "vasp2cp2k 没写出 $d/cp2k.inp(见上面 PY 那一步是否报错)"
        ( cd "$d" && mpirun cp2k.psmp -i cp2k.inp > point.out 2>err ) || scf_fail "$t" "$d" "mpirun/cp2k 非零退出"
        ;;
      ot+diag)
        # 第一段 OT: 快速稳收敛密度 → 写 ${PROJECT}-RESTART.wfn(仅占据)
        PY vasp2cp2k --vasp "$(frame_file $t)" --template "$TEMPLATE" \
            --out "$d/cp2k_ot.inp" --guess "$frame_guess" --scf-mode ot $SPIN_ARG
        [ -s "$d/cp2k_ot.inp" ] || scf_fail "$t" "$d" "vasp2cp2k 没写出 $d/cp2k_ot.inp(见上面 PY 那一步是否报错)"
        ( cd "$d" && mpirun cp2k.psmp -i cp2k_ot.inp > ot.out 2>err_ot ) || scf_fail "$t" "$d" "OT 段非零退出"
        PY vasp2cp2k --vasp "$(frame_file $t)" --template "$TEMPLATE" \
            --out "$d/cp2k.inp" --guess restart --scf-mode diag $SPIN_ARG $CUBE_ARG $AO_ARG
        [ -s "$d/cp2k.inp" ] || scf_fail "$t" "$d" "vasp2cp2k 没写出 $d/cp2k.inp(见上面 PY 那一步是否报错)"
        ( cd "$d" && mpirun cp2k.psmp -i cp2k.inp > point.out 2>err ) || scf_fail "$t" "$d" "DIAG 段非零退出"
        ;;
      ot)
        PY vasp2cp2k --vasp "$(frame_file $t)" --template "$TEMPLATE" \
            --out "$d/cp2k.inp" --guess "$frame_guess" --scf-mode ot $SPIN_ARG
        [ -s "$d/cp2k.inp" ] || scf_fail "$t" "$d" "vasp2cp2k 没写出 $d/cp2k.inp(见上面 PY 那一步是否报错)"
        ( cd "$d" && mpirun cp2k.psmp -i cp2k.inp > point.out 2>err )
        echo "⚠ SCF_MODE=ot: OT 无虚轨道, 仅供测收敛, 后续 NAC 会因缺虚轨道失败。生产请用 ot+diag。"
        ;;
      *) echo "未知 SCF_MODE=$SCF_MODE (可选 diag/ot+diag/ot)"; exit 1 ;;
    esac
    # ── 无条件立即清理: 这几样从头到尾没有任何下游命令读过, 删了零风险 ──
    rm -f "$d"/*.bak-*
    [ "$SCF_MODE" = "ot+diag" ] && rm -f "$d/ot.out" "$d/err_ot"
    if [ "$SCF_MODE" = "ot" ]; then
        [ -s "$d/point.out" ] || scf_fail "$t" "$d" "point.out 是空的"
    else
        scf_ok "$d" || scf_fail "$t" "$d" "校验未通过(检查: 是否收敛 / 是否正常收尾 / wfn 是否写出)"
    fi
    echo "SCF done frame $t (mode=$SCF_MODE, 校验通过)"
}

# ---------- 每帧体检 + IPR + overlap + 布居 + 能带边成分 ----------
run_analysis(){
    t="$1"; fidx="${2:-1}"; tag=$(step_tag "$t"); d="scf_run/$tag"
    wfn="$d/${PROJECT}-RESTART.wfn"; ovl="$d/point.out"
    [ -f "$wfn" ] || { echo "no wfn frame $t, skip"; return; }
    if [ "$SPIN" = "on" ]; then CHANNELS="1 2"; else CHANNELS="1"; fi
    if ! need_S "$fidx"; then
        PY check --wfn "$wfn" --out "$ovl" --bmin "$BMIN" --bmax "$BMAX"
        run_cube_overlap "$t"
        echo "analysis done frame $t (轻量: ANALYSIS_STRIDE=$ANALYSIS_STRIDE, 本帧跳过 overlap/ipr/character)"
        return
    fi
    PY check --wfn "$wfn" --overlap "$ovl" --out "$ovl" --bmin "$BMIN" --bmax "$BMAX"  # 内部已遍历所有自旋
    for isp in $CHANNELS; do
        sfx=""; [ "$SPIN" = "on" ] && sfx=".spin${isp}"
        PY overlap  --wfn "$wfn" --overlap "$ovl" --bmin "$BMIN" --bmax "$BMAX" --ispin "$isp" --outdir "$d"
        [ -f "$d/overlap.dat" ] && mv "$d/overlap.dat" "Ipr_Overlap/overlap/overlap.${tag}${sfx}.dat"
        PY ipr --mode loewdin --wfn "$wfn" --overlap "$ovl" --bmin "$BMIN" --bmax "$BMAX" --ispin "$isp" --outdir "$d"
        [ -f "$d/ipr.dat" ] && mv "$d/ipr.dat" "Ipr_Overlap/ipr/ipr.${tag}${sfx}.dat"
        PY character --wfn "$wfn" --overlap "$ovl" --bmin "$BMIN" --bmax "$BMAX" --ispin "$isp" > "analysis/character.${tag}${sfx}.txt" 2>&1
    done
    PY popanal --out "$ovl" --outdir "$d" > "analysis/pop.${tag}.txt" 2>&1
    rm -f "$d"/*.__S*.npy                    # S / S^{1/2} 的解析缓存: 本帧 4 条命令共享完就删(各 ~95MB)
    run_cube_overlap "$t"
    echo "analysis done frame $t (自旋通道: $CHANNELS)"
}

# ---------- 每帧 cube → 轨道间实空间重叠(电子-空穴 overlap; 自动区分自旋) ----------
run_cube_overlap(){
    t="$1"; tag=$(step_tag "$t"); d="scf_run/$tag"
    [ "$MO_CUBES" = "on" ] || return
    ls "$d"/*WFN_*.cube >/dev/null 2>&1 || { echo "  cube-overlap: frame $t 没有 cube(已清理 / CP2K 没打印), 跳过"; return; }
    PY cube-overlap --cube-dir "$d" --wfn "$d/${PROJECT}-RESTART.wfn" \
        --tag "$tag" --outdir Cube_Overlap 2>&1 | tee "analysis/cube_overlap.${tag}.txt"
    if ls "Cube_Overlap/eh_dens/eh_dens.${tag}.spin"*.dat >/dev/null 2>&1; then
        [ "$KEEP_CUBE" -eq 1 ] || { rm -f "$d"/*WFN_*.cube; \
            [ -f .nac_notes/_cubeclean ] || { mkdir -p .nac_notes; touch .nac_notes/_cubeclean; \
                echo "  cube 已清理(KEEP_CUBE=0); 结果在 Cube_Overlap/  (后续帧同样处理, 不再提示)"; }; }
    else
        echo "  ⚠ cube-overlap 没产出结果 → 保留 cube 供排查(日志: analysis/cube_overlap.${tag}.txt)"
    fi
}

# ---------- 相邻对 cross-time NAC ----------
merged_diag(){
    echo "  ── merged.out 诊断 ───────────────────────────────────────────────"
    if grep -q "SCF run NOT converged" "$1/merged.out" 2>/dev/null; then
        echo "  ★ 命中已知坑: merged.out 里出现 'SCF run NOT converged' 并 ABORT。"
        echo "    合并体系的 SCF 注定不收敛(两份拷贝叠在一起), 这【不是错】——但 CP2K 会 abort,"
        echo "    导致 post-SCF 的 &AO_MATRICES 打印根本没执行 → 没有 S 可用。"
        echo "  → 解法: merged_overlap.inp 的 &SCF 里必须有  IGNORE_CONVERGENCE_FAILURE T"
        echo "    本脚本生成的 inp 已自动加。若你这里是【上一版留下的旧 m_* 目录】, 删掉重跑:"
        echo "        rm -rf run_nac/nac/m_*"
    fi
    grep -q "OVERLAP MATRIX" "$1/merged.out" 2>/dev/null \
        || echo "  ⚠ merged.out 里没有 'OVERLAP MATRIX' → &DFT/&PRINT/&AO_MATRICES OVERLAP T 没生效。"
    echo "  ── merged.out 末尾 20 行 ──"; tail -20 "$1/merged.out" 2>/dev/null
    [ -s "$1/err" ] && { echo "  ── stderr ──"; tail -10 "$1/err"; }
    echo "  ──────────────────────────────────────────────────────────────────"
}

run_nac_pair(){
    a="$1"; b="$2"; pair="$3"; ta=$(step_tag "$a"); tb=$(step_tag "$b"); ptag=$(step_tag "$pair")
    if [ "$SPIN" = "on" ]; then CHANNELS="1 2"; else CHANNELS="1"; fi

    if [ "$CELL_CHOICE" = "all" ]; then CELLS="average t tdt"; else CELLS="$CELL_CHOICE"; fi

    for cc in $CELLS; do
        case "$cc" in
            average) mdir="run_nac/nac/m_${ta}_${tb}";          oroot="run_nac" ;;
            t)       mdir="run_nac/nac/m_${ta}_${tb}_cell_t";   oroot="run_nac/cell_t" ;;
            tdt)     mdir="run_nac/nac/m_${ta}_${tb}_cell_tdt"; oroot="run_nac/cell_tdt" ;;
            *) echo "✗✗ CELL_CHOICE 非法: '$CELL_CHOICE' (只能是 t / tdt / average / all)"; exit 1 ;;
        esac

        NAC_PAIR_CACHE_DIR="$NAC_PAIR_CACHE_ROOT/$cc"
        export NAC_PAIR_CACHE_DIR
        mkdir -p "$NAC_PAIR_CACHE_DIR"

        if [ ! -s "$mdir/merged.out" ] || ! grep -q "PROGRAM ENDED AT" "$mdir/merged.out"; then
            [ -f "$mdir/merged.out" ] && { echo "  残留的 merged.out 没通过校验 → 重算"; rm -f "$mdir/merged.out"; }
            PY build-merged --out-t "scf_run/$ta/point.out" --out-tdt "scf_run/$tb/point.out" \
                --ref-input "$TEMPLATE" --cell-choice "$cc" --outdir "$mdir" --quiet \
                || { echo "✗✗ build-merged 失败(pair $ta-$tb, cell=$cc) —— 看上面 python 的报错"; exit 1; }
            [ -s "$mdir/merged_overlap.inp" ] \
                || { echo "✗✗ build-merged 没写出 $mdir/merged_overlap.inp"; exit 1; }
            ( cd "$mdir" && mpirun cp2k.psmp -i merged_overlap.inp > merged.out 2>err ) \
                || { echo "✗✗ 合并 overlap 计算失败(pair $ta-$tb, cell=$cc)"; merged_diag "$mdir"; exit 1; }
            grep -q "PROGRAM ENDED AT" "$mdir/merged.out" \
                || { echo "✗✗ $mdir/merged.out 没有正常收尾(ABORT? 被 kill? 磁盘满?)"; merged_diag "$mdir"; exit 1; }
            grep -q "OVERLAP MATRIX" "$mdir/merged.out" \
                || { echo "✗✗ $mdir/merged.out 里没有 OVERLAP MATRIX —— 这一步的唯一产物没拿到"; \
                     merged_diag "$mdir"; exit 1; }
            rm -f "$mdir/MERGED-RESTART.wfn"                  # inp 里已 &RESTART OFF; 保险再删一次
        fi

        for isp in $CHANNELS; do
            if [ "$SPIN" = "on" ]; then sp_sub="/spin${isp}"; else sp_sub=""; fi
            for sub in normal phase_corr; do
                od="${oroot}/${sub}${sp_sub}"; ensure_dir "$od/ham_ry"
                [ -s "$od/ham_ry/0_Ham_${pair}_re" ] && [ -s "$od/ham_ry/0_Ham_${pair}_im" ] \
                    && [ -s "$od/ham_ry/0_Ham_${pair}.meta.npz" ] && continue
                if [ "$sub" = "normal" ]; then phase_flag="--no-phase --no-procrustes"; track_flag="--track none"
                else                           phase_flag="";                         track_flag="--track perm"; fi
                PY wfn-nac --wfn-t "scf_run/$ta/${PROJECT}-RESTART.wfn" \
                    --wfn-tdt "scf_run/$tb/${PROJECT}-RESTART.wfn" --overlap "$mdir/merged.out" \
                    --dt "$DT_PER_USED_FRAME" --bmin "$BMIN" --bmax "$BMAX" --outdir "$od" \
                    --ispin "$isp" --legacy-tag "$ptag" --terse $phase_flag $track_flag \
                    --degen-thresh "$DEGEN_THRESH" \
                    || { echo "✗✗ wfn-nac 失败 (pair $ta-$tb, spin $isp, cell $cc)。中止。"; exit 1; }
                [ -s "$od/ham_ry/0_Ham_${pair}_re" ] && [ -s "$od/ham_ry/0_Ham_${pair}_im" ] \
                    && [ -s "$od/ham_ry/0_Ham_${pair}.meta.npz" ] \
                    || { echo "✗✗ V3 wfn-nac 没写完整 0_Ham_${pair}_re/_im/meta。中止。"; exit 1; }
                cat "$od/energy_by_band" >> "$od/energy"
            done
        done

        if [ "$KEEP_MERGED_OUT" -eq 0 ]; then
            nac_ok=1
            for isp in $CHANNELS; do
                sp_sub=""; [ "$SPIN" = "on" ] && sp_sub="/spin${isp}"
                for sub in normal phase_corr; do
                    [ -s "${oroot}/${sub}${sp_sub}/ham_ry/0_Ham_${pair}_re" ] \
                        && [ -s "${oroot}/${sub}${sp_sub}/ham_ry/0_Ham_${pair}_im" ] \
                        && [ -s "${oroot}/${sub}${sp_sub}/ham_ry/0_Ham_${pair}.meta.npz" ] || nac_ok=0
                done
            done
            [ "$nac_ok" -eq 1 ] && { rm -f "$mdir/merged.out" "$mdir"/*.__S*.npy; \
                [ -f .nac_notes/_mergedclean ] || { mkdir -p .nac_notes; touch .nac_notes/_mergedclean; \
                    echo "  merged.out 已删(KEEP_MERGED_OUT=0; 每对约0.9GB, 后续对同样处理, 不再提示)"; }; }
        fi
    done

    if [ "$CELL_CHOICE" = "all" ]; then
        echo "NAC + Ham(on-the-fly) done pair $ta-$tb (自旋通道: $CHANNELS; 胞: average/t/tdt 三套)"
    else
        echo "NAC + Ham(on-the-fly) done pair $ta-$tb (自旋通道: $CHANNELS)"
    fi
}

# ---------- 主循环 ----------
PAIR=$(( (FRAME_START - GLOBAL_FIRST_FRAME) / FRAME_STRIDE ))
PREV=""; PREV2=""; FIRST=1; PREFLIGHT=0; FIDX=0            # PREV2 必须初始化(set -u)
echo "全局配对编号: 本段从 PAIR=$PAIR 起(GLOBAL_FIRST_FRAME=$GLOBAL_FIRST_FRAME, FRAME_STRIDE=$FRAME_STRIDE)"
for FRM in $FRAME_IDS; do
    FIDX=$((FIDX+1))
    if [ "$FIRST" -eq 1 ]; then
        run_scf "$FRM" 1 "" "$FIDX"
    else
        run_scf "$FRM" 0 "$PREV" "$FIDX"
    fi
    if [ "$PREFLIGHT" -eq 0 ]; then
        PY check --wfn "scf_run/$(step_tag "$FRM")/${PROJECT}-RESTART.wfn" \
            --bmin "$BMIN" --bmax "$BMAX" --strict || exit 1
        PREFLIGHT=1
    fi
    run_analysis "$FRM" "$FIDX"
    [ -n "$PREV" ] && { run_nac_pair "$PREV" "$FRM" "$PAIR"; PAIR=$((PAIR+1)); }
    [ "$KEEP_WFN" -eq 1 ] || { [ -n "$PREV2" ] && [ "$(_dec "$PREV2")" -ne "$(_dec "$FRAME_START")" ] && rm -f "scf_run/$(step_tag "$PREV2")/${PROJECT}-RESTART.wfn"; }
    [ "$KEEP_OUT" -eq 1 ] || { [ -n "$PREV2" ] && [ "$(_dec "$PREV2")" -ne "$(_dec "$FRAME_START")" ] && rm -f "scf_run/$(step_tag "$PREV2")/point.out" "scf_run/$(step_tag "$PREV2")/err"; }
    PREV2="$PREV"; PREV="$FRM"; FIRST=0
done

if [ "$SPIN" = "on" ]; then
    echo "完成(自旋极化)。NAC/Ham 分自旋: run_nac/{normal,phase_corr}/spin1/ 和 .../spin2/ (alpha/beta 各一套独立 Hvib)。"
    echo "  两个自旋通道相互独立(跨自旋 NAC=0), PYXAID 里按两个通道分别做动力学。"
else
    echo "完成(闭壳层)。V3 Ham(Ry) 已由 wfn-nac 直接写入 ham_ry/；phase_corr 保留 full G^T E G 实部非对角。"
fi
echo "对照: normal vs phase_corr 的 real/eig 摆一起看; |d_ij| 需相位校正后才是规范无关可比量。"
if [ "$MO_CUBES" = "on" ]; then
    echo "电子-空穴实空间 overlap → Cube_Overlap/{eh_dens,eh_amp,eh_cos}/*.<帧>.spin<n>.dat"
    echo "  逐帧汇总(直接拿去画随相变的趋势): Cube_Overlap/summary.csv"
    echo "  每个 cube 到底是哪条轨道(真 HOMO/LUMO 按占据数定, 不按下标猜): Cube_Overlap/index/"
    echo "  ★变胞轨迹比趋势用 eh_cos(无量纲); 要和文献绝对值对表用 eh_dens(注意文献是 |ψ|² 还是 |ψ| 约定)。"
fi
