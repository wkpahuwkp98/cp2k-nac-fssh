#!/usr/bin/env python3
# Copyright 2026 wkpahuwkp98
# 本文件包含改编自 nano-qmflows 的代码
# Copyright 2022 Netherlands eScience Center, Vrije Universiteit Amsterdam
# https://github.com/SCM-NV/nano-qmflows
# Licensed under the Apache License, Version 2.0
# 许可全文见同目录 LICENSE，涉及范围与修改说明见同目录 NOTICE

"""
cp2k_gaussian_nac_parallel.py

用法:
    python cp2k_gaussian_nac_parallel.py selftest --xyz <frame.extxyz> --frame 0 \
        --basis gth-dzvp-molopt-sr --pseudo gth-pbe --xc pbe
    python cp2k_gaussian_nac_parallel.py nac --xyz <frame.extxyz> --frame-t 0 --frame-tdt 1 \
        --dt 1.0 --bmin 160 --bmax 200 --basis gth-dzvp-molopt-sr --pseudo gth-pbe --xc pbe

参数说明见仓库 README。
"""
import argparse, os, sys, re, glob, mmap
import numpy as np

_NOTES_MODE = os.environ.get("NAC_NOTES", "once").lower()
_NOTES_DIR = os.environ.get("NAC_NOTES_DIR", ".nac_notes")

def note_once(key, msg):
    if _NOTES_MODE == "off":
        return
    if _NOTES_MODE == "full":
        print(msg); return
    try:
        os.makedirs(_NOTES_DIR, exist_ok=True)
        mk = os.path.join(_NOTES_DIR, re.sub(r'[^A-Za-z0-9_.-]', '_', key)[:80])
        if os.path.exists(mk):
            return
        open(mk, 'w').close()
    except OSError:
        pass
    print(msg)

def mo_overlap(C_a, S_cross, C_b):
    return C_a.conj().T @ S_cross @ C_b

def phase_factors(C_0, S_0x, C_x):
    diag = np.einsum('mi,mn,ni->i', C_0.conj(), S_0x, C_x)
    mag = np.abs(diag)
    mag[mag < 1e-12] = 1e-12
    return diag / mag

def nac_gaussian(C_t, C_tdt, S_cross, dt,
                 C_0=None, S_0t=None, S_0tdt=None, gamma=True):
    pij = mo_overlap(C_t, S_cross, C_tdt)
    pji = mo_overlap(C_tdt, S_cross.T, C_t)
    if C_0 is not None and S_0t is not None and S_0tdt is not None:
        cc1 = phase_factors(C_0, S_0t, C_t)
        cc2 = phase_factors(C_0, S_0tdt, C_tdt)
        pij = pij * cc1[:, None] * cc2.conj()[None, :]
        pji = pji * cc2[:, None] * cc1.conj()[None, :]
    tmp = (pij - pji) / (2.0 * dt)
    nac = tmp.real if gamma else tmp

    if gamma:
        nac = 0.5 * (nac - nac.T)
    else:
        nac = 0.5 * (nac - nac.conj().T)
    return nac

def read_extxyz_frame(fn, frame_idx):
    lines = open(fn).read().splitlines()
    i = 0; cur = 0
    while i < len(lines):
        if not lines[i].strip() or not lines[i].split()[0].isdigit():
            i += 1; continue
        nat = int(lines[i].split()[0])
        if cur == frame_idx:
            m = re.search(r'Lattice="([^"]+)"', lines[i + 1])
            lat = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3) if m else None
            els = []; pos = []
            for k in range(nat):
                p = lines[i + 2 + k].split()
                els.append(p[0]); pos.append([float(x) for x in p[1:4]])
            return lat, els, np.array(pos)
        cur += 1; i += 2 + nat
    raise IndexError(f"帧 {frame_idx} 超出范围(共 {cur} 帧)")

def pyscf_scf(lat, els, pos, basis, pseudo, xc, verbose=0):
    from pyscf.pbc import gto, dft
    cell = gto.Cell()
    cell.atom = [[e, tuple(p)] for e, p in zip(els, pos)]
    cell.a = lat
    cell.basis = basis
    cell.pseudo = pseudo
    cell.verbose = verbose
    cell.build()
    mf = dft.RKS(cell)
    mf.xc = xc
    mf.kernel()
    return mf.mo_coeff, mf.get_ovlp(), mf.mo_energy, cell

def pyscf_cross_overlap(cell_t, cell_tdt):
    from pyscf.pbc import gto

    from pyscf import gto as molgto
    def to_mol(cell):
        m = molgto.Mole()
        m.atom = cell.atom
        m.basis = cell.basis
        m.pseudo = cell.pseudo
        m.spin = None
        m.build(parse_arg=False)
        return m
    m_t = to_mol(cell_t); m_tdt = to_mol(cell_tdt)
    return molgto.intor_cross('int1e_ovlp', m_t, m_tdt)

def cmd_selftest(a):
    try:
        import pyscf
    except ImportError:
        sys.exit("✗ 需要 PySCF: pip install pyscf")
    lat, els, pos = read_extxyz_frame(a.xyz, a.frame)
    print(f"帧 {a.frame}: {len(els)} 原子, 做 PBC-DFT({a.xc}/{a.basis})...")
    C, S, E, cell = pyscf_scf(lat, els, pos, a.basis, a.pseudo, a.xc, verbose=a.verbose)
    nao, nmo = C.shape
    print(f"  nao={nao}, nmo={nmo}")

    CtSC = mo_overlap(C, S, C)
    off = CtSC - np.eye(nmo)
    print(f"  ‖C†SC − I‖_max = {np.abs(off).max():.2e}  (应 ~1e-10, 证明夹 S 的 MO overlap 正确)")
    print(f"  对比: 若【不】夹 S(照抄平面波点乘) ‖C†C − I‖_max = {np.abs(C.conj().T@C - np.eye(nmo)).max():.2e}"
          f"  (高斯基非正交, 这个【不】是单位阵→证明必须夹 S)")

def cmd_nac(a):
    try:
        import pyscf
    except ImportError:
        sys.exit("✗ 需要 PySCF: pip install pyscf")
    lat_t, els_t, pos_t = read_extxyz_frame(a.xyz, a.frame_t)
    lat_d, els_d, pos_d = read_extxyz_frame(a.xyz, a.frame_tdt)
    print(f"帧 t={a.frame_t}, t+dt={a.frame_tdt}, dt={a.dt} fs")
    print("  SCF(t)...");   C_t, S_t, E_t, cell_t = pyscf_scf(lat_t, els_t, pos_t, a.basis, a.pseudo, a.xc, a.verbose)
    print("  SCF(t+dt)..."); C_d, S_d, E_d, cell_d = pyscf_scf(lat_d, els_d, pos_d, a.basis, a.pseudo, a.xc, a.verbose)
    print("  cross-time AO overlap S^{t,t+dt}...")
    S_cross = pyscf_cross_overlap(cell_t, cell_d)

    bmin, bmax = a.bmin, a.bmax
    Ct = C_t[:, bmin:bmax + 1]
    Cd = C_d[:, bmin:bmax + 1]
    nac = nac_gaussian(Ct, Cd, S_cross, a.dt, gamma=True)
    ent = 0.5 * (E_t[bmin:bmax + 1] + E_d[bmin:bmax + 1])

    print(f"\n  活性空间 band {bmin}..{bmax} ({bmax-bmin+1} 个)")
    print(f"  NAC 最大 |off-diag| = {np.abs(nac - np.diag(np.diag(nac))).max():.4e} (1/fs)")
    print(f"  反对称性 ‖nac+nacᵀ‖_max = {np.abs(nac + nac.T).max():.2e} (应~0)")

    os.makedirs(a.outdir, exist_ok=True)

    np.savetxt(os.path.join(a.outdir, "eig.txt"), ent[np.newaxis, :], fmt="%.8e")
    np.savetxt(os.path.join(a.outdir, "nac2_re.txt"), nac.flatten()[np.newaxis, :], fmt="%.6e")
    np.savetxt(os.path.join(a.outdir, "nac2_im.txt"), np.zeros_like(nac).flatten()[np.newaxis, :], fmt="%.6e")
    print(f"  已写 {a.outdir}/eig.txt, nac2_re.txt, nac2_im.txt (格式同 nac.py, 可接 PYXAID)")
    print("\n  ★ 小测试要看的: (1) selftest 里 C†SC=I 成立; (2) 这里 NAC 反对称; "
          "(3) NAC 量级合理(过渡态附近 ~0.01-0.1/fs)。都成立 → 高斯基 time-overlap 方法在你体系 work。")

CP2K_NOTE = r"""
────────────────────────────────────────────────────────────────────────
从你的两个 .wfn 到 NAC —— 缺的那块是 AO overlap 矩阵 S:

  MO overlap = C(t)† · S · C(t+dt)     (高斯基非正交, 不能像平面波直接点乘)
  C(t),C(t+dt) ← 已能从 .wfn 读出(read_cp2k_wfn); S ← .wfn 里没有, 要另外让 CP2K 给。

拿 S 的两个方案(取决于你两帧原子笛卡尔坐标是否相同):

【方案1: 只改晶格盒子、原子笛卡尔坐标不变】→ 单帧 S 就够
  基函数中心没动, cross-time overlap ≈ 单帧 AO overlap S(Γ点忽略镜像差)。
  在任一帧 CP2K 输入里加:
      &DFT
        &PRINT
          &AO_MATRICES
            OVERLAP .TRUE.
            OMIT_HEADERS .TRUE.
            NDIGITS 16
            FILENAME =overlap
          &END
        &END
      &END
  解析成 (nao,nao) 的 .npy(可用本文件 parse_cp2k_ao_overlap, 务必核对 S 对称、对角≈1), 然后:
    python cp2k_gaussian_nac.py wfn-nac --wfn-t R1.wfn --wfn-tdt R2.wfn \
        --overlap overlap.npy --dt <fs> --bmin <..> --bmax <..>
  脚本自检 ‖C1†S C1 − I‖≈0 —— 不为0说明 S 与 wfn 的 AO 排序没对上(基组/原子顺序不一致)。

【方案2: 原子也动了(真实相变轨迹)】→ 必须 cross-time S(严格, 含 Pulay 项)
  构造合并体系: 帧t 全部原子 + 帧t+dt 全部原子(顺序: 帧t那份在前), 放同一盒子。对这 2N 原子
  跑一次 CP2K(MAX_SCF 1 即可, 只要 overlap), 同样 &AO_MATRICES OVERLAP 打印整块 S_full(2N×2N)。
  传 --overlap 时脚本自动取非对角块 S_full[:N,N:] = <χ(帧t)|χ(帧t+dt)> 作 cross-time S。
  (这与 Libra 的 CP2K+libint2 接口做的是同一件事。)

变胞镜像求和: Γ点周期 overlap=Σ_L<χ(r_t)|χ(r_t+dt+L)>, 晶格 L 两帧不同, 用 t/t+dt/平均是近似;
  盒子大、局域基衰减快时影响有限, 相变大幅变 cell 建议对比看差异。

⚠ 生产前用 `check` 子命令确认 wfn 是 SCF 收敛的(本征值不能全≈0, 那是未收敛 atomic guess 的
  特征): `python cp2k_gaussian_nac.py check --wfn xxx.wfn --out xxx.out`。overlap 部分和本征值
  是否合理是两件独立的事——S 算得对不代表 Hvib 对角(能量)就有意义。

生产整条 201 帧轨迹时建议直接用 Libra 的 CP2K+libint2 接口出 C 和 cross-time S, 喂 nac_gaussian()。
────────────────────────────────────────────────────────────────────────
"""

def read_cp2k_wfn(fn, meta_only=False):
    from scipy.io import FortranFile
    f = FortranFile(fn, 'r')
    natom, nspin, nao, nset_max, nshell_max = f.read_ints(np.int32)
    f.read_ints(np.int32); f.read_ints(np.int32); f.read_ints(np.int32)
    spins = []
    for _ in range(nspin):
        nmo, homo, lfomo, nelectron = f.read_ints(np.int32)
        eo = f.read_reals(np.float64)
        eig = eo[:nmo].copy(); occ = eo[nmo:2 * nmo].copy()
        if meta_only:
            C = None
            for imo in range(nmo):
                f.read_reals(np.float64)
        else:
            C = np.zeros((nao, nmo))
            for imo in range(nmo):
                C[:, imo] = f.read_reals(np.float64)
        spins.append(dict(C=C, eig=eig, occ=occ, nmo=nmo, homo=homo, nelectron=nelectron))
    f.close()
    return dict(natom=natom, nao=nao, nspin=nspin, spins=spins)

def _rowchars(buf, st, nrb, voff, ncol, fw, llen, rows, mcache):
    dst = np.diff(st)
    if dst.size == 0 or bool((dst == dst[0]).all()):
        Wl = int(dst[0]) if dst.size else int(llen[rows[0]]) + 1
        return buf[st[0]:st[0] + nrb * Wl].reshape(nrb, Wl)[:, voff:voff + ncol * fw]
    rel = st - st[0]
    L = int(rel[-1]) + voff + ncol * fw
    key = (nrb, ncol, L)
    mk = mcache.get(key)
    if mk is None or not np.array_equal(mk[1], rel):
        d = np.zeros(L + 1, np.int8)
        d[rel + voff] += 1
        d[rel + voff + ncol * fw] -= 1
        mk = (np.cumsum(d[:-1]).astype(bool), rel)
        mcache[key] = mk
    return buf[st[0]:st[0] + L][mk[0]].reshape(nrb, ncol * fw)

_FW_W = {}

def _fw_atof(ch, fw, dp, ndec):
    if dp + ndec > 15:
        return np.fromstring(ch.tobytes().decode('ascii', 'replace'), sep=' ')
    key = (fw, dp, ndec)
    if key not in _FW_W:
        w = np.zeros(fw)
        for j in range(fw):
            if j < dp:
                w[j] = 10.0 ** (dp - 1 - j)
            elif j > dp:
                w[j] = 10.0 ** (dp - j)
        _FW_W[key] = ((w * 10.0 ** ndec).round().astype(np.int64), 10.0 ** ndec)
    wint, scale = _FW_W[key]
    f = ch.reshape(-1, fw)
    d = f.astype(np.int16) - 48
    dig = np.where((d >= 0) & (d <= 9), d, 0).astype(np.int64)
    v = (dig @ wint) / scale
    return np.where((f == 45).any(axis=1), -v, v)

def _parse_worker(args):
    (fn, pos, hard_end, todo, p_dot, voff, fw, dp, ndec, dim, m, n_half, shm_name) = args
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    S = np.ndarray((m, m), dtype=np.float64, buffer=shm.buf)
    filled = 0
    with open(fn, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        buf = np.frombuffer(mm, dtype=np.uint8, count=hard_end - pos, offset=pos)
        ends = np.flatnonzero(buf == 10)
        lstart = np.concatenate(([0], ends[:-1] + 1))
        llen = ends - lstart
        probe = buf[np.minimum(lstart + p_dot, len(buf) - 1)]
        is_data = (llen > p_dot) & (probe == 46)
        dat = np.flatnonzero(is_data)
        mcache = {}
        for (h0, h1, cb) in todo:
            lo = int(np.searchsorted(dat, h0, 'right'))
            up = int(np.searchsorted(dat, h1, 'left'))
            rows = dat[lo:up][:dim]
            if n_half:
                rows = rows[:n_half]
            if rows.size == 0:
                continue
            nrb = int(rows.size); ncol = len(cb)
            st = lstart[rows]
            ch = _rowchars(buf, st, nrb, voff, ncol, fw, llen, rows, mcache)
            vals = _fw_atof(ch, fw, dp, ndec).reshape(nrb, ncol)

            r0 = int(bytes(buf[st[0]:st[0] + voff]).split(None, 1)[0]) - 1
            rN = int(bytes(buf[st[-1]:st[-1] + voff]).split(None, 1)[0]) - 1
            if r0 != 0 or rN != nrb - 1:
                raise RuntimeError(f"列块 {cb[0]+1} 的行号不连续({r0+1}..{rN+1}, 期望 1..{nrb})")
            rr = np.arange(nrb, dtype=np.int64)
            cba = np.array(cb)
            if n_half:
                csel = cba >= n_half
                if not csel.any():
                    continue
                S[np.ix_(rr, cba[csel] - n_half)] = vals[:, csel]
                filled += nrb * int(csel.sum())
            else:
                S[np.ix_(rr, cba)] = vals
                filled += nrb * ncol

    shm.close()
    return filled

def _cache_path(fn, tag):
    st = os.stat(fn)
    return f"{fn}.__S{tag}_{st.st_size}_{int(st.st_mtime)}.npy"

def parse_cp2k_ao_overlap(fn, nao=None, occurrence=-1, header="OVERLAP MATRIX",
                          allow_incomplete=False, cross_of=None, cross_hint=None):
    use_cache = os.environ.get("NAC_CACHE", "on").lower() != "off"

    tag = f"x{cross_of or cross_hint}" if (cross_of or cross_hint) else "f"
    cp = None
    if use_cache:
        try:
            cp = _cache_path(fn, tag)
            if os.path.exists(cp):
                return np.load(cp)
        except OSError:
            cp = None

    hb = header.encode()
    with open(fn, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        starts = []
        i = mm.find(hb)
        while i >= 0:
            starts.append(i); i = mm.find(hb, i + 1)
        if not starts:
            mm.close()
            sys.exit(f"✗ {fn} 里没找到 '{header}'。确认 CP2K &AO_MATRICES OVERLAP 打印是否打开。")
        pos = starts[occurrence]

        hard_end = min([x for x in starts if x > pos], default=len(mm))
        seg = mm[pos:hard_end]
        mm.close()

    buf = np.frombuffer(seg, dtype=np.uint8)
    ends = np.flatnonzero(buf == 10)
    if ends.size == 0:
        sys.exit(f"✗ {fn}: OVERLAP MATRIX 段里一行都没有?")
    lstart = np.concatenate(([0], ends[:-1] + 1))
    llen = ends - lstart

    p_dot = -1
    for k in range(min(3000, len(ends))):
        ln0 = seg[lstart[k]:ends[k]]
        if b'.' in ln0:
            p_dot = ln0.find(b'.'); break
    if p_dot < 0:
        sys.exit(f"✗ {fn}: OVERLAP MATRIX 段里找不到任何数值行(没有小数点)。")
    probe = buf[np.minimum(lstart + p_dot, len(buf) - 1)]
    is_data = (llen > p_dot) & (probe == 46)
    is_blank = llen == 0

    def _chain(cand_idx):
        hs = []; nx = 0
        for hi in cand_idx:
            t = seg[lstart[hi]:ends[hi]].split()
            if not t or not all(x.isdigit() for x in t):
                continue
            cb = [int(x) - 1 for x in t]
            if cb[0] != nx or any(cb[k] != cb[0] + k for k in range(len(cb))):
                if hs:
                    break
                continue
            hs.append((int(hi), np.array(cb)))
            nx = cb[-1] + 1
        return hs, nx

    lastc = buf[np.maximum(ends - 1, 0)]
    strict = np.flatnonzero((~is_data) & (~is_blank) & (lastc >= 48) & (lastc <= 57))
    heads, dim = _chain(strict)
    if not heads:
        heads, dim = _chain(np.flatnonzero((~is_data) & (~is_blank)))
    if dim == 0:
        sys.exit(f"✗ {fn}: OVERLAP MATRIX 段里没找到列头行, 格式可能不符。")

    if cross_hint and dim == 2 * cross_hint:
        cross_of = cross_hint
    if nao is None:
        nao = dim
    n_half = cross_of
    m = cross_of if cross_of else nao

    d0 = int(np.flatnonzero(is_data)[0])
    fl = seg[lstart[d0]:ends[d0]]
    dots = [ii for ii, c in enumerate(fl) if c == 46]
    ncol0 = len(heads[0][1])
    fw = dots[1] - dots[0] if len(dots) >= 2 else -1
    voff = len(fl) - ncol0 * fw if fw > 0 else -1
    fixed = fw > 0 and voff > 0 and len(fl[voff:].split()) == ncol0
    dp = dots[0] - voff if fixed else 0
    ndec = fw - dp - 1 if fixed else 0

    dat = np.flatnonzero(is_data)
    S = np.zeros((m, m)); rowcnt = np.zeros(m, np.int64)
    mcache = {}

    def _rows_of(i):
        h0 = heads[i][0]
        h1 = heads[i + 1][0] if i + 1 < len(heads) else len(ends)
        lo = int(np.searchsorted(dat, h0, 'right'))
        up = int(np.searchsorted(dat, h1, 'left'))
        return dat[lo:up][:dim]

    njobs = int(os.environ.get("NAC_JOBS", "1"))
    if fixed and njobs > 1 and len(heads) >= 4 * njobs:
        try:
            from multiprocessing import shared_memory, get_context
            todo = [(heads[i][0],
                     heads[i + 1][0] if i + 1 < len(heads) else len(ends),
                     [int(x) for x in heads[i][1]])
                    for i in range(len(heads))
                    if not (cross_of and int(heads[i][1].max()) < n_half)]
            shm = shared_memory.SharedMemory(create=True, size=m * m * 8)
            try:
                Sp = np.ndarray((m, m), dtype=np.float64, buffer=shm.buf); Sp[:] = 0.0
                args = [(fn, pos, hard_end, todo[k::njobs], p_dot, voff, fw, dp, ndec,
                         dim, m, n_half, shm.name) for k in range(njobs)]
                args = [a for a in args if a[3]]
                with get_context("fork").Pool(len(args)) as pool:
                    got = sum(pool.map(_parse_worker, args))
                if got != m * m:
                    raise RuntimeError(f"并行解析填了 {got}/{m*m} 个元素")
                S = Sp.copy()
            finally:
                shm.close(); shm.unlink()
            if cp:
                try:
                    np.save(cp, S)
                except OSError:
                    pass
            return S
        except Exception as e:
            print(f"  ⚠ 并行解析失败({e}) → 退回单进程(结果不受影响)")
            S = np.zeros((m, m)); rowcnt[:] = 0

    for i, (hi, col_block) in enumerate(heads):
        if cross_of and int(col_block.max()) < n_half:
            continue
        rows = _rows_of(i)
        if cross_of:
            rows = rows[:n_half]
        if rows.size == 0:
            continue
        nrb = int(rows.size); ncol = len(col_block)
        st = lstart[rows]
        if fixed:
            ch = _rowchars(buf, st, nrb, voff, ncol, fw, llen, rows, mcache)
            vals = _fw_atof(ch, fw, dp, ndec).reshape(nrb, ncol)

            r0 = int(seg[st[0]:st[0] + voff].split(None, 1)[0]) - 1
            rN = int(seg[st[-1]:st[-1] + voff].split(None, 1)[0]) - 1
            if r0 == 0 and rN == nrb - 1:
                rr = np.arange(nrb, dtype=np.int64)
            else:
                lab = buf[st[:, None] + np.arange(voff)[None, :]]
                rr = np.fromiter((int(bytes(x).split(None, 1)[0]) - 1 for x in lab), np.int64, nrb)
        else:
            rr = np.empty(nrb, np.int64); vv = []
            for j in range(nrb):
                ln_ = seg[lstart[rows[j]]:ends[rows[j]]]
                t2 = ln_.rsplit(None, ncol)
                rr[j] = int(t2[0].split()[0]) - 1; vv.append(b" ".join(t2[1:]))
            vals = np.fromstring(b" ".join(vv).decode('ascii', 'replace'),
                                 sep=' ').reshape(nrb, ncol)
        if n_half:

            csel = col_block >= n_half
            keep = rr < n_half
            if csel.any() and keep.any():
                S[np.ix_(rr[keep], col_block[csel] - n_half)] = vals[np.ix_(keep, csel)]
                np.add.at(rowcnt, rr[keep], int(csel.sum()))
        else:
            ok_ = (rr >= 0) & (rr < m)
            S[np.ix_(rr[ok_], col_block)] = vals[ok_]
            np.add.at(rowcnt, rr[ok_], ncol)

    need = m * m
    got = int(rowcnt.sum())
    if got != need:
        bad = np.flatnonzero(rowcnt != m)
        msg = (f"✗ {fn}: OVERLAP MATRIX 只解析到 {got}/{need} 个元素, 缺 {need-got} 个"
               f"({bad.size} 行不完整, 首个缺行 AO#{bad[0]+1 if bad.size else '?'})。\n"
               f"  最常见原因: 这个 out 被写了一半就中断了(作业被 kill / 磁盘满 / SCF 崩了)。\n"
               f"  ★ 决不能带着缺元素的 S 往下算——它会静默产出一份'看起来正常'的垃圾 NAC。\n"
               f"  排查: grep -c 'OVERLAP MATRIX' {fn}; tail -5 {fn}(应有正常收尾)。\n"
               f"  确认无误仍要强行继续: 加 --allow-incomplete。")
        if not allow_incomplete:
            sys.exit(msg)
        print("  ⚠ " + msg.replace("✗ ", "").splitlines()[0] + "  (--allow-incomplete 已放行, 缺的按 0 算)")

    if cp:
        try:
            np.save(cp, S)
        except OSError:
            pass
    return S

def read_cp2k_coords(fn, header="ATOMIC COORDINATES IN ANGSTROM"):
    text = open(fn).read().splitlines()
    i = 0
    while i < len(text) and header not in text[i]:
        i += 1
    if i >= len(text):
        sys.exit(f"✗ {fn} 里没找到 '{header}'(需 PRINT_LEVEL MEDIUM 或更高)")
    i += 1

    while i < len(text) and ('Atom' in text[i] or not text[i].split()):
        i += 1
    els = []; xyz = []
    while i < len(text):
        t = text[i].split()
        if len(t) >= 9 and re.fullmatch(r'\d+', t[0]):
            els.append(t[2]); xyz.append([float(t[4]), float(t[5]), float(t[6])])
            i += 1
        else:
            break
    return els, np.array(xyz)

def read_cp2k_cell(fn):
    a = b = c = None
    for ln in open(fn):
        t = ln.split()
        if "CELL| Vector a" in ln and len(t) >= 7:
            a = [float(x) for x in t[4:7]]
        elif "CELL| Vector b" in ln and len(t) >= 7:
            b = [float(x) for x in t[4:7]]
        elif "CELL| Vector c" in ln and len(t) >= 7:
            c = [float(x) for x in t[4:7]]
    if a and b and c:
        return np.array([a, b, c])
    return None

def diag_overlap_health(pij, eig_t=None, bmin=0, label="", thresh=0.9, show_worst=6, verbose=True):
    d = np.abs(np.diag(pij))
    bad = np.where(d < thresh)[0]
    if not verbose:
        return d
    print(f"  [{label}] diag|<i(t)|i(t+dt)>| : min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f}"
          f"  低于{thresh}的比例={len(bad)}/{len(d)}")
    if len(bad):
        order = bad[np.argsort(d[bad])][:show_worst]
        print(f"    最差 {len(order)} 个(局部idx→全局MO, |diag|" +
              (", 该态能量eV, 到上/下态gap eV)" if eig_t is not None else ")") + ":")
        for k in order:
            gi = k + bmin
            if eig_t is not None:
                HARTREE2EV = 27.211386245988
                e = eig_t[gi] * HARTREE2EV
                gdn = (eig_t[gi] - eig_t[gi - 1]) * HARTREE2EV if gi > 0 else float('nan')
                gup = (eig_t[gi + 1] - eig_t[gi]) * HARTREE2EV if gi + 1 < len(eig_t) else float('nan')
                print(f"      局部{k:3d}→MO{gi:4d}  |diag|={d[k]:.3f}  E={e:8.4f}eV  gap↓{gdn:7.4f} gap↑{gup:7.4f}")
            else:
                print(f"      局部{k:3d}→MO{gi:4d}  |diag|={d[k]:.3f}")
        print("    ⚠ 上表 gap 小(密集/近简并manifold)的态, 低|diag|大概率是子空间规范自由度, 不是"
              "算错; --tracking 对此无效(见函数说明)。若这些态在你的活性窗口边缘, 考虑收窄 bmin/bmax"
              "到 gap 更大的孤立态。")
    return d

def manifold_slices(nwin, nocc_active):
    if not (0 < int(nocc_active) < int(nwin)):
        raise ValueError(f"invalid nocc_active={nocc_active} for nwin={nwin}")
    return (slice(0, int(nocc_active)), slice(int(nocc_active), int(nwin)))

def _perm_respects_manifolds(P, nocc_active):
    P = np.asarray(P, dtype=int)
    n = len(P); nv = int(nocc_active)
    return (np.all(P[:nv] < nv) and np.all(P[nv:] >= nv)
            and np.array_equal(np.sort(P), np.arange(n)))

def block_hungarian(O, nocc_active):
    from scipy.optimize import linear_sum_assignment
    O = np.asarray(O)
    n = O.shape[0]
    if O.shape != (n, n):
        raise ValueError("block_hungarian requires square O")
    Q = np.arange(n, dtype=int)
    for sl in manifold_slices(n, nocc_active):
        rows = np.arange(sl.start, sl.stop)
        sub = O[np.ix_(rows, rows)]
        qloc = linear_sum_assignment(-(np.abs(sub) ** 2))[1]
        Q[rows] = rows[qloc]
    return Q

def degen_clusters_by_manifold(e_slot_ev, nocc_active, threshold_ev):
    e = np.asarray(e_slot_ev, dtype=float)
    n = len(e)
    out = []
    for sl in manifold_slices(n, nocc_active):
        ids = np.arange(sl.start, sl.stop)
        if len(ids) == 0:
            continue
        sube = e[ids]
        adj = np.abs(sube[:, None] - sube[None, :]) < float(threshold_ev)
        seen = np.zeros(len(ids), bool)
        for kk in range(len(ids)):
            if seen[kk]:
                continue
            stack = [kk]; comp = []
            while stack:
                u = stack.pop()
                if seen[u]:
                    continue
                seen[u] = True
                comp.append(int(ids[u]))
                stack.extend(np.where(adj[u] & ~seen)[0].tolist())
            out.append(sorted(comp))
    return out

def offdiag_max(A):
    A = np.asarray(A)
    if A.size == 0:
        return 0.0
    B = A.copy()
    np.fill_diagonal(B, 0)
    return float(np.max(np.abs(B)))

def vc_block_metrics(nac, Hel_eV, nocc_active):
    HBAR_EVFS = 0.6582119569
    n = nac.shape[0]; nv = int(nocc_active)
    V = slice(0, nv); C = slice(nv, n)
    hd = HBAR_EVFS * np.asarray(nac)
    vc = hd[V, C]
    vv = hd[V, V]
    cc = hd[C, C]
    return {
        "nvalence": nv,
        "nconduction": n - nv,
        "VC_hbar_d_frobenius_eV": float(np.linalg.norm(vc)),
        "VC_hbar_d_rms_eV": float(np.sqrt(np.mean(np.abs(vc)**2))) if vc.size else 0.0,
        "VC_hbar_d_max_eV": float(np.max(np.abs(vc))) if vc.size else 0.0,
        "VV_hbar_d_frobenius_eV": float(np.linalg.norm(vv)),
        "CC_hbar_d_frobenius_eV": float(np.linalg.norm(cc)),
        "Hel_offdiag_max_eV": offdiag_max(Hel_eV),
        "Hel_VC_max_eV": float(np.max(np.abs(np.asarray(Hel_eV)[V, C]))) if vc.size else 0.0,
    }

def write_pyxaid_ham_v3(outdir, pair_idx, Hel_eV, nac, metadata=None):
    RY = 13.60569253
    HBAR_EVFS = 0.6582119569
    outdir = os.path.abspath(outdir)
    hdir = os.path.join(outdir, "ham_ry")
    os.makedirs(hdir, exist_ok=True)
    Hel_eV = np.asarray(Hel_eV, dtype=float)
    nac = np.asarray(nac, dtype=float)
    X = -HBAR_EVFS * nac
    ham_re = Hel_eV / RY
    ham_im = (-1.0 / RY) * X
    np.fill_diagonal(ham_im, 0.0)
    np.savetxt(os.path.join(hdir, f"0_Ham_{pair_idx}_re"), ham_re, fmt="%15.10f")
    np.savetxt(os.path.join(hdir, f"0_Ham_{pair_idx}_im"), ham_im, fmt="%15.10f")

    md = {} if metadata is None else dict(metadata)
    md.update({
        "format_version": np.array("V3_neardeg_full_Hel"),
        "pair": np.int64(pair_idx),
        "Hel_eV": Hel_eV,
        "nac_1_per_fs": nac,
        "X_minus_hbar_d_eV": X,
    })
    np.savez(os.path.join(hdir, f"0_Ham_{pair_idx}.meta.npz"), **md)
    return ham_re, ham_im, X

def cmd_wfn_nac(a):
    w1 = read_cp2k_wfn(a.wfn_t)
    w2 = read_cp2k_wfn(a.wfn_tdt)
    if a.terse:
        note_once("wfnnac_dims",
                  f"wfn: natom={w1['natom']} nao={w1['nao']} nmo={w1['spins'][0]['nmo']} (t & t+dt 一致, 全程不变)")
    else:
        print(f"wfn_t : natom={w1['natom']} nao={w1['nao']} nmo={w1['spins'][0]['nmo']}")
        print(f"wfn_tdt: natom={w2['natom']} nao={w2['nao']} nmo={w2['spins'][0]['nmo']}")
    assert w1['nao'] == w2['nao'], "两帧 nao 不一致(基组/原子数必须相同)"
    isp = a.ispin - 1
    if isp >= w1['nspin'] or isp >= w2['nspin']:
        sys.exit(f"✗ 请求 --ispin {a.ispin}, 但 wfn 只有 {w1['nspin']} 个自旋通道。"
                 f"{' 这是闭壳层(未开自旋)计算, 只有 --ispin 1。' if w1['nspin']==1 else ''}"
                 f" 若要 alpha/beta 两通道, CP2K 输入需开 LSD(奇电子/缺陷体系: SPIN=on + MULTIPLICITY=2)。")
    C1 = w1['spins'][isp]['C']; C2 = w2['spins'][isp]['C']
    nao = w1['nao']

    if a.overlap.endswith('.npy'):
        S = np.load(a.overlap)
    else:

        S = parse_cp2k_ao_overlap(a.overlap, nao=None, cross_hint=nao,
                                  allow_incomplete=getattr(a, 'allow_incomplete', False))
    merged_s = (S.shape[0] == nao) and not a.overlap.endswith('.npy')
    if S.shape[0] == 2 * nao:
        S = S[:nao, nao:]; merged_s = True
    if merged_s:
        note_once("wfnnac_cross", f"  overlap: 合并体系 → 只解析 cross-time 块 [:{nao},{nao}:](跳过另外 3/4)")
        if np.abs(S).max() < 1e-10:
            sys.exit(f"✗ {a.overlap}: 合并 S 的【跨时非对角块全是 0】 → 这不可能(相邻帧的基函数必然有交叠)。"
                     f" 多半是 out 被截断, 或 OVERLAP MATRIX 段没打全。")
    assert S.shape == (nao, nao), f"overlap 形状 {S.shape} 应为 ({nao},{nao}) 或 ({2*nao},{2*nao})"

    sym = np.abs(S - S.T).max()
    CtSC = mo_overlap(C1, S if sym < 1e-6 else 0.5 * (S + S.T), C1)
    selfcheck = np.abs(CtSC - np.eye(C1.shape[1])).max() if sym < 1e-6 else None
    if a.terse:
        tag = f"合并{2*nao}×{2*nao}" if merged_s else "单帧"
        note_once("wfnnac_ovl_kind", f"  overlap: {tag} (每对同构, 下面只报数值)")
        line = f"  ‖S−Sᵀ‖={sym:.1e}"
        if selfcheck is not None:
            line += f"  单帧S自检‖C†SC−I‖={selfcheck:.1e}"
        print(line)
    else:
        print(f"  overlap 对称性 ‖S−Sᵀ‖={sym:.2e} (单帧S应~0; cross-time非对角块可非0)")
        if selfcheck is not None:
            print(f"  单帧S自检: ‖C1†S C1 − I‖={selfcheck:.2e} (应~0, 否则S与wfn不匹配!)")

    bmin, bmax = a.bmin, a.bmax
    nwin = bmax - bmin + 1
    e1 = w1['spins'][isp]['eig']; e2 = w2['spins'][isp]['eig']

    ifr1, _ = occ_frontier(w1['spins'][isp])
    ifr2, _ = occ_frontier(w2['spins'][isp])
    if not (bmin <= ifr1 < bmax):
        where = "完全在 HOMO 之下(深价带)" if bmax <= ifr1 else "完全在 LUMO 之上(高激发态)"
        msg = (f"✗ 活性窗 [{bmin}..{bmax}](0基, 共{nwin}个态) 【没有骑在占据前沿上】!\n"
               f"  本帧整数占据 HOMO = band {ifr1}(0基) = MO{ifr1+1}(1基); LUMO = band {ifr1+1}。\n"
               f"  你的窗口{where} → 窗内没有带边, NAC/Hvib 与带边复合无关。\n"
               f"  ★ 建议 BMIN={max(ifr1-5,0)}  BMAX={ifr1+5}  (HOMO±5)\n"
               f"  (确实研究非带边窗口才用 --allow-window-off 放行。)")
        if not getattr(a, 'allow_window_off', False):
            sys.exit(msg)
        print(msg.replace("✗", "⚠"))
    if ifr1 != ifr2:
        sys.exit(f"✗ 相邻两帧整数占据前沿不一致: HOMO(t)={ifr1}, HOMO(t+dt)={ifr2}。"
                 " V/C manifold 边界不能安全定义; 请先检查 SCF/电子数/自旋。")
    nocc_active = ifr1 - bmin + 1
    if not (0 < nocc_active < nwin):
        sys.exit(f"✗ active-space V/C split invalid: nocc_active={nocc_active}, nwin={nwin}")
    note_once("vc_split",
              f"  V3 manifold split: active slots 0..{nocc_active-1}=valence, "
              f"{nocc_active}..{nwin-1}=conduction; Hungarian/Procrustes 永不跨 gap。")

    C1w = C1[:, bmin:bmax + 1]
    C2w = C2[:, bmin:bmax + 1]
    Ct = C1w.copy()
    Cd = C2w.copy()

    O_raw = mo_overlap(Ct, S, Cd)
    d_before = np.abs(np.diag(O_raw))
    diag_overlap_health(O_raw, eig_t=e1, bmin=bmin, label="wfn-nac", verbose=not a.terse)

    P = np.arange(nwin)
    Q = np.arange(nwin)
    G_prev = np.eye(nwin)
    G_new = np.eye(nwin)
    nproc = 0
    track = getattr(a, 'track', 'perm')
    perm_file = os.path.join(a.outdir, f"track_perm.spin{a.ispin}.txt")
    pair_idx = int(a.legacy_tag) if (a.legacy_tag and str(a.legacy_tag).isdigit()) else None

    cache_dir = os.environ.get("NAC_PAIR_CACHE_DIR", "").strip()
    if cache_dir and pair_idx is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"pair_{pair_idx:06d}.spin{a.ispin}.npz")
        tmp_path = cache_path + f".tmp.{os.getpid()}.npz"
        np.savez(
            tmp_path,
            format_version=np.array("V3_neardeg_raw_pair"),
            pair=np.int64(pair_idx),
            ispin=np.int64(a.ispin),
            bmin=np.int64(bmin),
            bmax=np.int64(bmax),
            nocc_active=np.int64(nocc_active),
            dt_fs=np.float64(a.dt),
            O_raw=np.asarray(O_raw, dtype=np.float64),
            e_t_h=np.asarray(e1[bmin:bmax + 1], dtype=np.float64),
            e_tdt_h=np.asarray(e2[bmin:bmax + 1], dtype=np.float64),
        )
        os.replace(tmp_path, cache_path)
        note_once("raw_pair_cache_v3",
                  f"  V3 raw-pair cache: {cache_dir}/pair_XXXXXX.spin{a.ispin}.npz "
                  f"(含 nocc_active={nocc_active}, 用于全局 segment stitch)")

    if track == 'perm':
        os.makedirs(a.outdir, exist_ok=True)
        prev_pair = None
        if os.path.isfile(perm_file):
            try:
                raw = open(perm_file).read().split()
                prev_pair = int(raw[0])
                cand = np.array(raw[1:1 + nwin], dtype=int)
                if cand.size == nwin:
                    P = cand
                gtok = raw[1 + nwin:]
                if len(gtok) == nwin * nwin:
                    G_prev = np.array(gtok, dtype=float).reshape(nwin, nwin)
                elif len(gtok) == 0:
                    note_once("gauge_legacy",
                              "  提示: track_perm 是老格式(只有置换) → 本对按单位规范起算。")
            except Exception:
                pass

        if pair_idx is not None and prev_pair is not None and pair_idx != prev_pair + 1:
            if pair_idx == 1:
                P = np.arange(nwin); G_prev = np.eye(nwin)
            else:
                print(f"  ⚠ 态跟踪链不连续(上次第 {prev_pair} 对, 这次第 {pair_idx} 对)。"
                      f" 分段并行生产请最终使用 stitch_nac_segments.py 的全局结果。")

        if not _perm_respects_manifolds(P, nocc_active):
            sys.exit(f"✗ 继承的 permutation 跨越了 V/C gap: P={P.tolist()}。"
                     f" 删除旧 {perm_file} 后重跑本段，或使用 V3 global stitch。")
        cross_g = max(np.max(np.abs(G_prev[:nocc_active, nocc_active:])),
                      np.max(np.abs(G_prev[nocc_active:, :nocc_active])))
        if cross_g > 1e-8:
            sys.exit(f"✗ 继承的 gauge matrix 含 V/C 跨块旋转(max={cross_g:.3e})。"
                     f" 这是旧版 tracking 产物，不能用于 V3；删除 {perm_file} 后重跑。")

        Ct = C1w[:, P] @ G_prev
        O = mo_overlap(Ct, S, C2w)

        Q = block_hungarian(O, nocc_active)
        Cd = C2w[:, Q]
        d_hung = np.abs(np.diag(mo_overlap(Ct, S, Cd)))
        nswap = int((Q != P).sum())
        gtag = "单位" if np.allclose(G_prev, np.eye(nwin)) else "继承"
        print(f"  V3 block-Hungarian: {nswap}/{nwin} slots 重排 (t规范:{gtag}); "
              f"min|Oii| {d_before.min():.3f} → {d_hung.min():.3f}; "
              f"V={nocc_active}, C={nwin-nocc_active}")

        if not getattr(a, 'no_procrustes', False):
            e_slot_ev = e1[bmin:bmax + 1][P] * 27.211386245988
            thr = getattr(a, 'degen_thresh', 0.050)
            clusters = degen_clusters_by_manifold(e_slot_ev, nocc_active, thr)
            dnow = np.abs(np.diag(mo_overlap(Ct, S, Cd)))
            done = []
            for cl in clusters:
                if len(cl) < 2:
                    continue
                if dnow[cl].min() > 0.95:
                    continue
                O_sub = mo_overlap(Ct[:, cl], S, Cd[:, cl])
                U_, _, Vt_ = np.linalg.svd(O_sub)
                R = Vt_.T @ U_.T
                G_new[np.ix_(cl, cl)] = R
                nproc += 1
                done.append(cl)
            if nproc:
                Cd = Cd @ G_new
                d_pro = np.abs(np.diag(mo_overlap(Ct, S, Cd)))
                print(f"  V3 same-manifold Procrustes: {nproc} clusters "
                      f"(ΔE<{thr*1000:.0f} meV, slots={done}); "
                      f"min|Oii| {dnow.min():.3f} → {d_pro.min():.3f}")
    else:
        print("  态跟踪: 关闭(--track none)，按原始 KS index；仅建议作为 raw baseline。")

    if a.no_phase:
        ph = np.ones(nwin)
        print("  相位: 未校正(raw baseline)")
    else:
        diag = np.diag(mo_overlap(Ct, S, Cd))
        ph = np.sign(diag.real)
        ph[ph == 0] = 1.0
        Cd = Cd * ph[None, :]
        print(f"  相位: 已校正，翻转 {(ph < 0).sum()}/{nwin} 个态")

    if track == 'perm':
        G_out = G_new * ph[None, :]
        with open(perm_file, 'w') as fh:
            fh.write(f"{pair_idx if pair_idx is not None else 0}\n")
            fh.write(" ".join(map(str, Q)) + "\n")
            fh.write(" ".join(f"{v:.16e}" for v in G_out.ravel()) + "\n")
    else:

        G_out = np.eye(nwin) * ph[None, :]

    O_final = mo_overlap(Ct, S, Cd)
    d_fin = np.abs(np.diag(O_final))
    if np.sum(d_fin < 0.8):
        worst = np.argsort(d_fin)[:3]
        print(f"  ⚠ V3 对齐后仍有 {int(np.sum(d_fin<0.8))}/{nwin} 个态 |Oii|<0.8; "
              f"worst slots={worst.tolist()} values={[float(d_fin[k]) for k in worst]}。"
              " 建议检查 dt / degen threshold / active window。")

    nac = nac_gaussian(Ct, Cd, S, a.dt, gamma=True)

    E1raw = np.diag(e1[bmin:bmax + 1][P])
    E2raw = np.diag(e2[bmin:bmax + 1][Q])
    if track == 'perm':
        Ht_h = G_prev.T @ E1raw @ G_prev
        Htdt_h = G_out.T @ E2raw @ G_out
    else:
        Ht_h = E1raw
        Htdt_h = G_out.T @ E2raw @ G_out
    Hel_h = 0.5 * (Ht_h + Htdt_h)
    Hel_h = 0.5 * (Hel_h + Hel_h.T)
    Hel_eV = Hel_h * 27.211386245988
    ent_eV = np.diag(Hel_eV).copy()

    if np.abs(Hel_h).max() < 1e-8:
        print("  ⚠ 本征值≈0: WFN 可能不是收敛后的物理解；NAC可算但 Hamiltonian 能量无意义。")

    print(f"  active {bmin}..{bmax}: max|NAC offdiag|="
          f"{offdiag_max(nac):.4e}/fs, anti-sym err={np.abs(nac+nac.T).max():.2e}; "
          f"max|Hel offdiag|={offdiag_max(Hel_eV)*1000:.3f} meV")

    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, "eig.txt"), ent_eV[np.newaxis, :], fmt="%.8e")
    np.savetxt(os.path.join(a.outdir, "nac2_re.txt"), nac.flatten()[np.newaxis, :], fmt="%.6e")
    np.savetxt(os.path.join(a.outdir, "nac2_im.txt"), np.zeros_like(nac).flatten()[np.newaxis, :], fmt="%.6e")

    HBAR_EVFS = 0.6582119569
    X = -HBAR_EVFS * nac

    dE_diag = np.abs(ent_eV[:, None] - ent_eV[None, :]) + np.eye(nwin) * 1e9
    ratio = np.abs(X) / dE_diag
    iu = np.triu_indices(nwin, 1)
    if iu[0].size and np.max(ratio[iu]) > 1.0:
        k = np.argmax(ratio[iu]); ii, jj = iu[0][k], iu[1][k]
        print(f"  ⚠ strong-coupling/tracking diagnostic: max |ħd|/|ΔE_diag|="
              f"{ratio[ii,jj]:.2f} at slots {ii}-{jj} "
              f"(|ħd|={abs(X[ii,jj])*1000:.1f} meV, ΔE_diag={dE_diag[ii,jj]*1000:.1f} meV). "
              "这可能是真实强非绝热区，也可能是 tracking/dt 问题；不要据此自动删除。")

    tag = a.legacy_tag if a.legacy_tag else "0001"
    real_iv = np.zeros((nwin, 2*nwin))
    real_iv[:, 0::2] = X
    np.savetxt(os.path.join(a.outdir, f"real{tag}"), real_iv, fmt="%.6e")
    with open(os.path.join(a.outdir, "energy_by_band"), "w") as fh:
        fh.write("  ".join(f"{e:12.6f}" for e in ent_eV) + "\n")
    np.savetxt(os.path.join(a.outdir, f"electronic_ham_eV.{tag}.txt"),
               Hel_eV, fmt="%15.10f")

    with open(os.path.join(a.outdir, f"track_index.spin{a.ispin}.txt"), "w") as fh:
        fh.write("# slot MO_t(0base) MO_tdt(0base) diag_Hel_eV |Oii| manifold\n")
        for kk in range(nwin):
            man = "V" if kk < nocc_active else "C"
            fh.write(f"{kk:5d} {bmin+P[kk]:9d} {bmin+Q[kk]:11d} "
                     f"{ent_eV[kk]:14.7f} {d_fin[kk]:12.6f} {man}\n")

    metrics = vc_block_metrics(nac, Hel_eV, nocc_active)
    with open(os.path.join(a.outdir, f"block_metrics.{tag}.txt"), "w") as fh:
        for key, val in metrics.items():
            fh.write(f"{key} {val}\n")
    print(f"  gauge-invariant V↔C ||ħd||F = {metrics['VC_hbar_d_frobenius_eV']*1000:.3f} meV; "
          f"RMS={metrics['VC_hbar_d_rms_eV']*1000:.3f} meV")

    out_pair = pair_idx if pair_idx is not None else int(tag)
    write_pyxaid_ham_v3(
        a.outdir, out_pair, Hel_eV, nac,
        metadata={
            "ispin": np.int64(a.ispin),
            "bmin": np.int64(bmin),
            "bmax": np.int64(bmax),
            "nocc_active": np.int64(nocc_active),
            "dt_fs": np.float64(a.dt),
            "P": np.asarray(P, dtype=np.int64),
            "Q": np.asarray(Q, dtype=np.int64),
            "G_prev": np.asarray(G_prev, dtype=np.float64),
            "G_out": np.asarray(G_out, dtype=np.float64),
            "O_final": np.asarray(O_final, dtype=np.float64),
        }
    )
    note_once("wfnnac_v3_files",
              f"  V3 输出: {a.outdir}/ham_ry/0_Ham_*_re/im (完整 transported Hel + NAC, Ry); "
              "realNNNN/energy_by_band 仅保留给旧分析，不可再 combine 覆盖 V3 Ham。")

def cmd_build_merged(a):
    els_t, xyz_t = read_cp2k_coords(a.out_t)
    els_d, xyz_d = read_cp2k_coords(a.out_tdt)
    assert els_t == els_d, "两帧元素序不一致(原子顺序必须相同)"
    n = len(els_t)
    cell_t = read_cp2k_cell(a.out_t); cell_d = read_cp2k_cell(a.out_tdt)
    if a.cell_t:
        cell_t = np.diag(a.cell_t)
    if a.cell_tdt:
        cell_d = np.diag(a.cell_tdt)
    if cell_t is None or cell_d is None:
        sys.exit("✗ out 里没解析到 CELL, 请用 --cell-t / --cell-tdt 手动给 ABC(如 --cell-t 8.716 8.7365 13.098)")
    cell = {'t': cell_t, 'tdt': cell_d, 'average': 0.5 * (cell_t + cell_d)}[a.cell_choice]

    os.makedirs(a.outdir, exist_ok=True)

    cinv = np.linalg.inv(cell)
    delta = xyz_d - xyz_t
    frac = delta @ cinv
    n0 = np.rint(frac).astype(int)

    _sh27 = np.array(
        [(i, j, k)
         for i in (-1, 0, 1)
         for j in (-1, 0, 1)
         for k in (-1, 0, 1)],
        dtype=int,
    )
    cand_n = n0[:, None, :] + _sh27[None, :, :]
    cand_shift = np.einsum("nsi,ij->nsj", cand_n, cell)
    cand_dvec = delta[:, None, :] - cand_shift
    cand_d2 = np.einsum("nsj,nsj->ns", cand_dvec, cand_dvec)

    ibest = np.argmin(cand_d2, axis=1)
    ii = np.arange(n)
    best_n = cand_n[ii, ibest]
    dvec = cand_dvec[ii, ibest]

    nshift = int(np.any(best_n != 0, axis=1).sum())

    n_round_wrong = int(np.any(best_n != n0, axis=1).sum())
    if n_round_wrong:
        print(f"  ★ triclinic 27-image: old round() would choose the wrong nearest image "
              f"for {n_round_wrong}/{n} atoms in this pair; strict Cartesian search used.")

    xyz_d_mi = xyz_t + dvec

    dfrac = dvec @ cinv

    xyz_path = os.path.join(a.outdir, "merged.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{2*n}\nmerged: frame_t (atoms 1-{n}) + frame_t+dt (atoms {n+1}-{2*n}, 已做最小镜像)\n")
        for e, p in zip(els_t, xyz_t):
            f.write(f"{e:2s} {p[0]:15.8f} {p[1]:15.8f} {p[2]:15.8f}\n")
        for e, p in zip(els_d, xyz_d_mi):
            f.write(f"{e:2s} {p[0]:15.8f} {p[1]:15.8f} {p[2]:15.8f}\n")

    ref_raw = open(a.ref_input).read()
    ref = "\n".join(ln for ln in ref_raw.splitlines() if not ln.lstrip().startswith('#'))
    def grab(pat, default=""):
        m = re.search(pat, ref)
        return m.group(1).strip() if m else default
    basis_file = grab(r'BASIS_SET_FILE_NAME\s+(\S+)')
    pot_file = grab(r'POTENTIAL_FILE_NAME\s+(\S+)')
    cutoff = grab(r'CUTOFF\s+(\d+)', "500")
    ndig = getattr(a, 'ndigits', 8)
    kinds = re.findall(r'(&KIND\b.*?&END KIND)', ref, re.DOTALL)
    kinds_txt = "\n".join("  " + k.replace("\n", "\n  ") for k in kinds)

    inp_path = os.path.join(a.outdir, "merged_overlap.inp")
    with open(inp_path, "w") as f:
        f.write(f"""&GLOBAL
  PROJECT MERGED
  RUN_TYPE ENERGY
  PRINT_LEVEL LOW
&END GLOBAL
&FORCE_EVAL
  &DFT
    BASIS_SET_FILE_NAME  {basis_file}
    POTENTIAL_FILE_NAME  {pot_file}
    &PRINT
      &AO_MATRICES
        OVERLAP T
        NDIGITS {ndig}
      &END
    &END
    &MGRID
      CUTOFF {cutoff}
      COMMENSURATE
    &END MGRID
    &QS
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      # ★★ 这个 SCF 【注定不收敛, 而且不需要收敛】★★
      #   合并体系 = 同一个体系的两份拷贝(t 和 t+dt), 对应原子只差 ~0.1 Å → 每个原子都压着一个
      #   孪生原子。核-核重叠能会飙到 ~1e6 Hartree, SCF 残差 ~1e7, 永远收敛不了。这是【设计如此】。
      #   我们要的只有 S_uv = <chi_u(t) | chi_v(t+dt)>, 它【只依赖几何+基组, 与密度/SCF 无关】。
      #   但 CP2K 的 &AO_MATRICES 打印在 post-SCF 阶段 → 一 CPABORT 就走不到那儿, out 里没有
      #   OVERLAP MATRIX。所以必须显式放行(CP2K 的报错信息里点名的就是这个关键字):
      IGNORE_CONVERGENCE_FAILURE T
      MAX_SCF 1
      SCF_GUESS ATOMIC
      EPS_SCF 1.0
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END
      &PRINT
        &RESTART OFF          # 别写 MERGED-RESTART.wfn(6912×1280≈70MB/对, 没有任何下游读它)
        &END
      &END
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      A  {cell[0,0]:.6f} {cell[0,1]:.6f} {cell[0,2]:.6f}
      B  {cell[1,0]:.6f} {cell[1,1]:.6f} {cell[1,2]:.6f}
      C  {cell[2,0]:.6f} {cell[2,1]:.6f} {cell[2,2]:.6f}
    &END CELL
    &TOPOLOGY
      COORD_FILE_FORMAT XYZ
      COORD_FILE_NAME merged.xyz
      CONNECTIVITY OFF
    &END TOPOLOGY
{kinds_txt}
  &END SUBSYS
&END FORCE_EVAL
""")
    disp_raw = np.linalg.norm(xyz_t - xyz_d, axis=1)
    disp = np.linalg.norm(dfrac @ cell, axis=1)
    dcell = np.abs(cell_t - cell_d).max()
    print(f"合并体系: {2*n} 原子 (帧t {n} + 帧t+dt {n}), cell={a.cell_choice}")
    print(f"  两帧【真实】原子位移(最小镜像后): max={disp.max():.3f} avg={disp.mean():.3f} Å")
    if nshift:
        print(f"  最小镜像: {nshift}/{n} 个原子跨了周期边界(裸笛卡尔差会把它们误报成 max={disp_raw.max():.2f} Å)。"
              f" 注: 周期体系的 AO overlap 本来就对 wrap 不变, 这一步修的是诊断和变胞残差, 不改物理。")
    if disp.max() > 2.0:
        print(f"  ⚠ 真实位移仍 >2Å: 帧间隔可能太大(dt 过长), cross-time overlap 会明显偏离单位阵 → NAC 不可信")

    print(f"  两帧胞差异 max|Δcell| = {dcell:.4f} Å  (cell_choice={a.cell_choice} 引入的镜像求和近似即此量级)")
    if dcell > 0.05:
        print(f"    ⚠ Δcell 不算小 → 建议做敏感性测试: CELL_CHOICE 分别取 t / tdt / average 各跑一遍,"
              f" 比较 NAC。若差异达几十%, 说明这个近似还没收敛, 不能直接用。")
    print(f"  写出: {xyz_path}")
    print(f"        {inp_path}")
    note_once("merged_kind", f"        (提取了 {len(kinds)} 个 &KIND; ⚠ 首次请核对 basis/potential 路径和 KIND 是否齐全)")
    if not a.quiet:
        print(f"\n下一步:")
        print(f"  1) cd {a.outdir} && <你的cp2k命令> -i merged_overlap.inp -o merged.out")
        print(f"  2) python {sys.argv[0]} wfn-nac --wfn-t <帧t.wfn> --wfn-tdt <帧t+dt.wfn> \\")
        print(f"         --overlap {a.outdir}/merged.out --dt <fs> --bmin <..> --bmax <..>")
        print(f"     (脚本自动从 {2*n}x{2*n} 的 S 取非对角块 [:{n_ao_hint()},{n_ao_hint()}:] 作 cross-time S)")

def n_ao_hint():
    return "N"

NQ_FS2AU = 1e-15 / 2.4188843265857e-17
NQ_H2EV = 27.211386245988
NQ_HBAR_EVFS = 0.6582119569

def nq_couplings_3points(dt_au, sji_t0, sij_t0, sji_t1, sij_t1):
    return (3.0 * (sji_t1 - sij_t1) + (sij_t0 - sji_t0)) / (4.0 * dt_au)

def nq_couplings_levine(dt_au, w_jk_in, w_kj_in):
    w_jk = w_jk_in.copy(); w_kj = w_kj_in.copy()
    w_jj = np.diag(np.diag(w_jk)); w_kk = np.diag(np.diag(w_kj))
    np.fill_diagonal(w_jk, 0); np.fill_diagonal(w_kj, 0)
    acos_w_jj = np.arccos(np.clip(w_jj, -1, 1)); asin_w_jk = np.arcsin(np.clip(w_jk, -1, 1))
    a = acos_w_jj - asin_w_jk; b = acos_w_jj + asin_w_jk
    A = -np.sinc(a / np.pi); B = np.sinc(b / np.pi)
    acos_w_kk = np.arccos(np.clip(w_kk, -1, 1)); asin_w_kj = np.arcsin(np.clip(w_kj, -1, 1))
    c = acos_w_kk - asin_w_kj; d = acos_w_kk + asin_w_kj
    C = np.sinc(c / np.pi); D = np.sinc(d / np.pi)
    w_lj = np.sqrt(np.clip(1 - w_jj ** 2 - w_kj ** 2, 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        w_lk = np.where(w_lj > 1e-12, -(w_jk * w_jj + w_kk * w_kj) / w_lj, 0.0)
    asin_w_lj = np.arcsin(np.clip(w_lj, -1, 1)); asin_w_lk = np.arcsin(np.clip(w_lk, -1, 1))
    asin_w_lj2 = asin_w_lj ** 2; asin_w_lk2 = asin_w_lk ** 2
    t1 = w_lj * w_lk * asin_w_lj
    x1 = np.sqrt(np.clip((1 - w_lj ** 2) * (1 - w_lk ** 2), 0, None)) - 1
    t2 = x1 * asin_w_lk; t = t1 + t2
    with np.errstate(divide='ignore', invalid='ignore'):
        E_nonzero = np.where(np.abs(asin_w_lj2 - asin_w_lk2) > 1e-30,
                             2 * asin_w_lj * t / (asin_w_lj2 - asin_w_lk2), 0.0)
    E1 = np.where(np.abs(w_lj) > 1e-8, E_nonzero, 0.0)
    E = np.where(np.isclose(asin_w_lj2, asin_w_lk2), w_lj ** 2, E1)
    return (np.arccos(np.clip(w_jj, -1, 1)) * (A + B) + np.arcsin(np.clip(w_kj, -1, 1)) * (C + D) + E) / (2 * dt_au)

def nq_compute_phases(overlaps):
    n, dim, _ = overlaps.shape
    mtx = np.empty((n + 1, dim)); mtx[0] = 1.0
    ref = np.ones(dim)
    for i in range(n):
        ref = np.sign(np.diag(overlaps[i])) * ref
        mtx[i + 1] = ref
    return mtx

def nq_correct_phases(overlaps, mtx_phases):
    n, dim, _ = overlaps.shape
    for k in range(n):
        p0 = mtx_phases[k].reshape(dim, 1); p1 = mtx_phases[k + 1].reshape(1, dim)
        overlaps[k] *= p0 @ p1
    return overlaps

def nq_track_unavoided_crossings(overlaps, nHOMO):
    from scipy.optimize import linear_sum_assignment
    nOv, nOrb, _ = overlaps.shape
    indexes = np.empty((nOv + 1, nOrb), dtype=np.int64)
    indexes[0] = np.arange(nOrb)
    for k in range(nOv):
        cost_h = -(overlaps[k, :nHOMO, :nHOMO] ** 2)
        cost_l = -(overlaps[k, nHOMO:, nHOMO:] ** 2)
        sw_h = linear_sum_assignment(cost_h)[1]
        sw_l = linear_sum_assignment(cost_l)[1]
        total = np.concatenate((sw_h, sw_l + nHOMO))
        indexes[k + 1] = total
        if k != nOv - 1:
            overlaps[k] = np.transpose(np.transpose(overlaps[k])[total])
            for i in range(k + 1, nOv):
                overlaps[i] = overlaps[i][:, total][total]
    acc = indexes[0]
    arr = np.empty(indexes.shape, dtype=np.int64); arr[0] = acc
    for i in range(nOv):
        acc = acc[indexes[i + 1]]; arr[i + 1] = acc
    return overlaps, arr

def cmd_hvib(a):
    wfns = a.wfns
    n = len(wfns)
    if n < 2:
        sys.exit("✗ 至少 2 个 .wfn")
    if len(a.overlaps) != n - 1:
        sys.exit(f"✗ {n} 帧需要 {n-1} 个逐对 overlap, 给了 {len(a.overlaps)}")

    Cs = []; Es = []; ifr = None; nao = None
    for fn in wfns:
        w = read_cp2k_wfn(fn); sp = w['spins'][a.ispin - 1]
        if nao is None:
            nao = w['nao']
            ifr, _ = occ_frontier(sp)
            homo_field = sp['homo']
        assert w['nao'] == nao, f"{fn}: nao 不一致"
        Cs.append(sp['C']); Es.append(sp['eig'])
    bmin, bmax = a.bmin, a.bmax
    nwin = bmax - bmin + 1
    nhomo_win = ifr - bmin + 1
    if not (0 < nhomo_win < nwin):
        sys.exit(f"✗ 活性窗 [{bmin}..{bmax}] 没有骑在占据前沿上(整数占据 HOMO = band {ifr}, 0基)。\n"
                 f"  窗内占据数算出来是 {nhomo_win}(应 0<n<{nwin}) → tracking 的占据/虚分块没有意义。\n"
                 f"  ★ 请把 BMIN/BMAX 设成包住 band {ifr}(HOMO)和 {ifr+1}(LUMO), 例如 "
                 f"BMIN={max(ifr-5,0)} BMAX={ifr+5}。")
    if homo_field != ifr + 1:
        print(f"  注: wfn 的 homo 字段={homo_field} ≠ 整数占据前沿={ifr+1}(1基) → SMEAR 尾巴撑大了它;"
              f" 这里用的是占据数定的前沿(正确的那个)。")
    print(f"{n} 帧, nao={nao}, 活性窗 MO[{bmin}..{bmax}]({nwin}个), 窗内占据 {nhomo_win}(HOMO=band {ifr})")

    S_mo = np.empty((n - 1, nwin, nwin))
    diag_min = np.empty(n - 1)
    for k, ov in enumerate(a.overlaps):
        S = np.load(ov) if ov.endswith('.npy') else parse_cp2k_ao_overlap(
            ov, nao=None, allow_incomplete=getattr(a, 'allow_incomplete', False))
        if S.shape[0] == 2 * nao:
            S = S[:nao, nao:]
        assert S.shape == (nao, nao), f"{ov}: 形状 {S.shape}"
        S_mo[k] = Cs[k][:, bmin:bmax + 1].T @ S @ Cs[k + 1][:, bmin:bmax + 1]
        diag_min[k] = diag_overlap_health(S_mo[k], verbose=False).min()
    print(f"逐对 MO overlap 完成: {n-1} 个 {nwin}×{nwin}")
    nflagged = int((diag_min < 0.9).sum())
    if nflagged:
        worst_k = int(np.argmin(diag_min))
        print(f"  ⚠ 对角健康检查: {nflagged}/{n-1} 对的 min|diag(pij)|<0.9(最差第{worst_k}对={diag_min[worst_k]:.3f})。"
              f" 大概率是密集/近简并子空间的规范自由度(不是算错, tracking 也修不了, 见 diag_overlap_health"
              f" 的说明); 若这些对恰好落在你的活性窗口边缘, 对单独一对跑"
              f" `wfn-nac --wfn-t {wfns[worst_k]} --wfn-tdt {wfns[worst_k+1]}` 看详细表, 评估要不要收窄窗口。")
    else:
        print(f"  对角健康检查: 全部 {n-1} 对 min|diag(pij)|≥0.9, 活性窗口内没有密集简并子空间的信号。")

    if a.tracking:
        S_mo, swaps = nq_track_unavoided_crossings(S_mo, nhomo_win)
        nsw = sum(int((swaps[i] != np.arange(nwin)).sum() > 0) for i in range(1, n))
        print(f"态跟踪: {nsw}/{n-1} 个时刻检测到 crossing 重排")
    else:
        swaps = np.tile(np.arange(nwin), (n, 1))
    phases = nq_compute_phases(S_mo)
    S_mo = nq_correct_phases(S_mo, phases)
    nflip = int((phases[1:] < 0).sum())
    print(f"相位链: 修正了 {nflip} 个 MO×帧 的符号翻转")

    dt_au = a.dt * NQ_FS2AU
    if a.algorithm == "3points":
        ncoup = n - 2
        coup = [nq_couplings_3points(dt_au, S_mo[i], S_mo[i].T, S_mo[i + 1], S_mo[i + 1].T)
                for i in range(ncoup)]
    else:
        ncoup = n - 1
        coup = [nq_couplings_levine(dt_au, S_mo[i], S_mo[i].T) for i in range(ncoup)]

    os.makedirs(a.outdir, exist_ok=True)
    for i in range(ncoup):
        ham_im = coup[i] * NQ_FS2AU * NQ_HBAR_EVFS
        np.fill_diagonal(ham_im, 0)
        e = 0.5 * (Es[i][bmin:bmax + 1] + Es[i + 1][bmin:bmax + 1])
        e = e[swaps[i + 1]]
        ham_re = np.diag(NQ_H2EV * e)
        np.savetxt(os.path.join(a.outdir, f"Ham_{i}_re"), ham_re, fmt='%10.5e', delimiter='  ')
        np.savetxt(os.path.join(a.outdir, f"Ham_{i}_im"), ham_im, fmt='%10.5e', delimiter='  ')
    mx = max(np.abs(c - np.diag(np.diag(c))).max() for c in coup) * NQ_FS2AU
    print(f"已写 {ncoup} 对 Ham_*_re/im → {a.outdir}/  (最大|NAC|≈{mx:.3e}/fs)")
    print("Ham_re: 对角=两帧平均轨道能[eV]; Ham_im: ħ·d_ij[eV], 对角=0。喂 PYXAID step3。")

def occ_frontier(sp):
    occ = sp['occ']
    maxocc = 2.0 if (occ.size and occ.max() > 1.5) else 1.0
    n_int = int(round(sp['nelectron'] / maxocc))
    return max(n_int - 1, 0), maxocc

def cmd_check(a):
    HARTREE2EV = 27.211386245988
    w = read_cp2k_wfn(a.wfn)
    note_once("chk_dims", f"wfn: natom={w['natom']} nao={w['nao']} nspin={w['nspin']}  (全程不变)")
    for isp, sp in enumerate(w['spins']):
        eig, occ, homo, nmo = sp['eig'], sp['occ'], sp['homo'], sp['nmo']
        note_once(f"chk_nmo{isp}", f"  spin{isp+1}: nmo={nmo} homo(wfn字段)={homo} nelectron={sp['nelectron']}")
        print(f"    eig range: [{eig.min():.4f}, {eig.max():.4f}] Hartree")
        if np.abs(eig).max() < 1e-8:
            print("    ⚠ 本征值全≈0: 像是未收敛的 atomic guess, 不要拿去做生产 NAC/Hvib。")
        if nmo == w['nao']:
            note_once("chk_virt", f"    虚轨道: nmo({nmo})=nao, DIAG+ADDED_MOS 补满了虚轨道。")
        else:
            print(f"    ⚠ nmo({nmo})<nao({w['nao']}), 虚轨道没补满——检查是不是 OT(OT 不产生空轨道,"
                  f" NAMD 需要 DIAGONALIZATION+ADDED_MOS)。")

        ifr, maxocc = occ_frontier(sp)
        if 0 <= ifr < nmo - 1:
            gap = (eig[ifr + 1] - eig[ifr]) * HARTREE2EV
            print(f"    整数占据前沿 idx={ifr}(0基, occ={occ[ifr]:.3f}) E={eig[ifr]*HARTREE2EV:.4f}eV;"
                  f" 下一条 idx={ifr+1}(occ={occ[ifr+1]:.3f}) E={eig[ifr+1]*HARTREE2EV:.4f}eV;"
                  f" gap={gap:.4f}eV")
            if homo != ifr + 1:
                print(f"    ⚠ wfn homo 字段({homo})≠整数占据前沿({ifr+1}, 1基): SMEAR 尾巴把 homo 撑大了,"
                      f" 别拿 wfn homo 当带边/算 gap(会拿到展宽尾巴虚轨道之间的假 gap)。")

        mocc = occ.max() if occ.size else 0.0
        frac = occ[(occ > 1e-6) & (occ < mocc - 1e-6)]
        if len(frac):
            print(f"    smearing: {len(frac)} 个态占据介于 0~{mocc:.1f} 之间(费米面展宽); "
                  f"若这些态落在你的 bmin/bmax 活性窗边缘, nHOMO/占据数窗口可能不再是整数分界,"
                  f" 注意 --tracking 的 nHOMO 参数假设了一个整数占据/非占据分界。")
        else:
            note_once("chk_smear_clean", f"    smearing: HOMO/LUMO 附近占据仍是干净的 0/{mocc:.1f} 分界(该体系 gap 远大于 kT,"
                  f" 光有 SMEAR 设置不代表占据真的展宽)。")
    if w['nspin'] == 2:
        mag = w['spins'][0]['nelectron'] - w['spins'][1]['nelectron']
        print(f"  磁化 nα−nβ = {mag}" + ("  (=1: 单缺陷双重态 ✓)" if mag == 1 else
              ("  (=0: 无净磁矩——若这是奇电子缺陷体系, 说明 SMEAR 把缺陷态摊成两半, 不是物理双重态!)" if mag == 0 else "")))

    if a.overlap:
        S = _load_single_frame_S(a.overlap, w['nao'])
        for isp, sp in enumerate(w['spins']):
            C = sp['C']
            bmin = 0 if a.bmin is None else a.bmin
            bmax = (sp['nmo'] - 1) if a.bmax is None else a.bmax
            Ct = C[:, bmin:bmax + 1]
            dev = np.abs(mo_overlap(Ct, S, Ct) - np.eye(Ct.shape[1])).max()
            print(f"  spin{isp+1} 自检 ‖C†SC−I‖[{bmin}..{bmax}] = {dev:.2e}"
                  )
            note_once("chk_csc_expl", "     (★ 量级由 NDIGITS 决定: 16位→~1e-15, 8位→~1e-8(打印舍入, 正常)。"
                      "到 1e-3 以上才说明 S 与该 wfn 不是同一次计算 / AO 排序不匹配)")

    if a.strict:

        bad = []
        for isp, sp in enumerate(w['spins']):
            ifr, _ = occ_frontier(sp)
            if a.bmin is None or a.bmax is None:
                sys.exit("✗ --strict 需要同时给 --bmin/--bmax")
            if not (a.bmin <= ifr < a.bmax):
                bad.append((isp + 1, ifr))
        if bad:
            isp, ifr = bad[0]
            sys.exit(f"\n✗✗ 活性窗 [{a.bmin}..{a.bmax}] 没有骑在占据前沿上(spin{isp}: HOMO=band {ifr}, 0基)!\n"
                     f"   ★ 请把 cp2k_nac.sh 里改成:  BMIN={max(ifr-5,0)}   BMAX={ifr+5}\n"
                     f"   (现在这个窗口里没有带边, 跑出来的 NAC/Hvib 与载流子复合无关。)\n")
        print(f"  --strict: 活性窗 [{a.bmin}..{a.bmax}] 包住了占据前沿 ✓")

    if a.out:
        text = open(a.out).read()
        if "SCF run converged in" in text:
            m_ = re.search(r'SCF run converged in\s+(\d+)\s+steps', text)
            print(f"  out: SCF 已收敛" + (f"(用了{m_.group(1)}步)" if m_ else ""))
        elif "SCF run NOT converged" in text:
            print("  out: ⚠ SCF 未收敛! 不要拿这个 wfn 做生产。")
        else:
            print("  out: 没找到 SCF 收敛状态标记(检查 print level 是否够高, 或这不是 SCF 计算的 out)。")
        cell = read_cp2k_cell(a.out)
        if cell is not None:
            abc = np.linalg.norm(cell, axis=1)
            print(f"  out: CELL |a|={abc[0]:.4f} |b|={abc[1]:.4f} |c|={abc[2]:.4f} Å")
        kind_counts = [int(x) for x in re.findall(r'Number of atoms:\s+(\d+)', text)]
        if kind_counts:
            total = sum(kind_counts)
            flag = "✓" if total == w['natom'] else "⚠ 不一致!"
            if flag == "✓":
                note_once("chk_kind_ok", f"  out: 各 kind 原子数合计 {total} vs wfn natom {w['natom']}  ✓")
            else:
                print(f"  out: 各 kind 原子数合计 {total} vs wfn natom {w['natom']}  {flag}")

def cmd_ot_diag(a):
    def _load(arg, header):
        if arg.endswith('.npy'):
            return np.load(arg)
        return parse_cp2k_ao_overlap(arg, nao=None, header=header)
    F = _load(a.ks, a.ks_header)
    S = _load(a.overlap, "OVERLAP MATRIX")
    assert F.shape == S.shape and F.shape[0] == F.shape[1], f"KS {F.shape} / S {S.shape} 形状不符"
    F = 0.5 * (F + F.T); S = 0.5 * (S + S.T)
    nao = F.shape[0]
    eig, C = generalized_eigh(F, S)
    dev = np.abs(C.T @ S @ C - np.eye(nao)).max()
    print(f"ot-diag: nao={nao}, 广义对角化完成, ‖CᵀSC−I‖={dev:.2e} (应≈0, 否则 KS/overlap 不匹配或未收敛)")
    print(f"  本征值范围: [{eig.min():.4f}, {eig.max():.4f}] Hartree")
    if a.homo:
        HARTREE2EV = 27.211386245988
        if 0 < a.homo < nao:
            gap = (eig[a.homo] - eig[a.homo - 1]) * HARTREE2EV
            print(f"  HOMO={eig[a.homo-1]*HARTREE2EV:.4f}eV  LUMO={eig[a.homo]*HARTREE2EV:.4f}eV"
                  f"  gap={gap:.4f}eV  虚轨道数={nao - a.homo}")
    os.makedirs(a.outdir, exist_ok=True)
    np.save(os.path.join(a.outdir, "C.npy"), C)
    np.save(os.path.join(a.outdir, "eig.npy"), eig)
    print(f"  已存 {a.outdir}/C.npy (nao×nao, 含虚轨道) 和 eig.npy (Hartree, 升序)。")
    print("  说明: 这是把 OT 收敛的 KS 矩阵事后对角化, 与 DIAG+ADDED_MOS 得到的正则轨道等价"
          "(同一个收敛密度 → 同一个 F → 同一套本征态)。相变段 NAC 仍需 cross-time S, 这步只解决"
          "\"OT 没有虚轨道\"的问题, 不改变 Pulay 那件事。")

def parse_ao_labels(fn, nao, header="OVERLAP MATRIX"):
    text = open(fn).read()
    starts = [x.start() for x in re.finditer(re.escape(header), text)]
    if not starts:
        return None
    seg = text[starts[-1]:]
    lab = [None] * nao
    for ln in seg.splitlines():
        t = ln.split()
        if len(t) >= 4 and re.fullmatch(r'\d+', t[0]) and re.fullmatch(r'\d+', t[1]) and re.fullmatch(r'[A-Za-z]+', t[2]):
            r = int(t[0]) - 1
            if 0 <= r < nao and lab[r] is None:
                mm = re.search(r'[spdfgSPDFG]', t[3])
                lab[r] = (int(t[1]), t[2], t[3][mm.start()].lower() if mm else '?')
    return lab

def parse_cp2k_populations(fn):
    lines = open(fn).read().splitlines()
    out = {'mulliken': None, 'hirshfeld': None, 'energy': None}

    def grab_block(start_kw, elem_col, charge_col):
        rows = []
        i = 0
        while i < len(lines) and start_kw not in lines[i]:
            i += 1
        if i >= len(lines):
            return None
        for j in range(i + 1, len(lines)):
            t = lines[j].split()
            if len(t) > max(elem_col, charge_col) and re.fullmatch(r'\d+', t[0]) and re.fullmatch(r'[A-Za-z]+', t[elem_col]):
                try:
                    rows.append((t[elem_col], float(t[charge_col])))
                except ValueError:
                    pass
            elif rows and ('Total' in lines[j] or not lines[j].strip()):
                if 'Total' in lines[j]:
                    break
        return rows or None

    out['mulliken'] = grab_block("Mulliken Population Analysis", 1, 4)
    out['hirshfeld'] = grab_block("Hirshfeld Charges", 1, 5)
    for ln in lines:
        if "ENERGY| Total FORCE_EVAL" in ln:
            try:
                out['energy'] = float(ln.split()[-1])
            except ValueError:
                pass
    return out

def cmd_popanal(a):
    p = parse_cp2k_populations(a.out)
    if p['energy'] is not None:
        print(f"Total energy = {p['energy']:.8f} Hartree = {p['energy']*27.211386245988:.4f} eV")
    for scheme in ('mulliken', 'hirshfeld'):
        rows = p[scheme]
        if not rows:
            print(f"{scheme}: 未在 out 里找到(需 PRINT_LEVEL MEDIUM; Hirshfeld 需 &DFT&PRINT&HIRSHFELD)")
            continue
        by_el = {}
        for el, q in rows:
            by_el.setdefault(el, []).append(q)
        print(f"\n{scheme} 每元素净电荷(个数, 平均, 范围):")
        for el in sorted(by_el, key=lambda e: -abs(np.mean(by_el[e]))):
            v = np.array(by_el[el])
            print(f"   {el:2s}  n={len(v):3d}  <q>={v.mean():+.3f}  [{v.min():+.3f}, {v.max():+.3f}]")
    if a.outdir and p['mulliken']:
        os.makedirs(a.outdir, exist_ok=True)
        np.savetxt(os.path.join(a.outdir, "mulliken_charges.dat"),
                   np.array([q for _, q in p['mulliken']])[:, None], fmt='%.6f')
        print(f"\n已存 {a.outdir}/mulliken_charges.dat (每原子净电荷, 顺序同 out)")

def cmd_character(a):
    w = read_cp2k_wfn(a.wfn); sp = w['spins'][a.ispin - 1]
    C = sp['C']; eig = sp['eig']; nao = w['nao']
    ifr, _ = occ_frontier(sp)
    homo = ifr + 1
    S = _load_single_frame_S(a.overlap, nao)
    lab = parse_ao_labels(a.overlap, nao) if not a.overlap.endswith('.npy') else None
    if lab is None or sum(x is None for x in lab):
        sys.exit("✗ 无法从 overlap 文本解析 AO 标签(需 CP2K &AO_MATRICES OVERLAP 文本, 非 .npy)。")
    P = np.abs(_sqrtm_psd_cached(S, a.overlap) @ C) ** 2
    H2EV = 27.211386245988

    def by(i, key):
        d = {}
        for mu in range(nao):
            k = key(lab[mu])
            d[k] = d.get(k, 0.0) + P[mu, i]
        return d

    def show_mo(i, name):
        el = by(i, lambda L: L[1])
        ell = by(i, lambda L: (L[1], L[2]))
        atom = by(i, lambda L: (L[0], L[1]))
        top_el = sorted(el.items(), key=lambda kv: -kv[1])[:4]
        top_ell = sorted(ell.items(), key=lambda kv: -kv[1])[:5]
        top_at = sorted(atom.items(), key=lambda kv: -kv[1])[:5]
        occ = sp['occ'][i]
        print(f"\n{name}  MO#{i}(0基)  E={eig[i]*H2EV:.3f} eV  occ={occ:.2f}")
        print("   元素: " + "  ".join(f"{e}:{100*v:.0f}%" for e, v in top_el))
        print("   通道: " + "  ".join(f"{e}-{l}:{100*v:.0f}%" for (e, l), v in top_ell))
        print("   最局域5原子: " + "  ".join(f"#{at}({e}):{100*v:.0f}%" for (at, e), v in top_at)
              + f"   (top5合计{100*sum(v for _,v in top_at):.0f}% → 越集中越局域)")

    show_mo(homo - 1, "VBM")
    show_mo(homo, "CBM")
    if a.bmin is not None and a.bmax is not None:
        print(f"\n活性窗 MO[{a.bmin}..{a.bmax}] 主导成分:")
        print("  MO   E(eV)   occ   主元素   主通道")
        for i in range(a.bmin, a.bmax + 1):
            el = by(i, lambda L: L[1]); ell = by(i, lambda L: (L[1], L[2]))
            e1 = max(el, key=el.get); (pe, pl) = max(ell, key=ell.get)
            print(f"  {i:4d} {eig[i]*H2EV:7.3f} {sp['occ'][i]:5.2f}   {e1}:{100*el[e1]:.0f}%   {pe}-{pl}:{100*ell[(pe,pl)]:.0f}%")

def _load_single_frame_S(overlap_arg, nao):
    S = np.load(overlap_arg) if overlap_arg.endswith('.npy') else parse_cp2k_ao_overlap(overlap_arg, nao=None)
    if S.shape[0] == 2 * nao:
        S = S[:nao, :nao]
    assert S.shape == (nao, nao), f"overlap 形状 {S.shape} 应为 ({nao},{nao}) 或 ({2*nao},{2*nao})"
    return 0.5 * (S + S.T)

def _sqrtm_psd_cached(S, src):
    if os.environ.get("NAC_CACHE", "on").lower() == "off" or not src or src.endswith('.npy'):
        return _sqrtm_psd(S)
    try:
        cp = _cache_path(src, "sqrt")
        if os.path.exists(cp):
            return np.load(cp)
        R = _sqrtm_psd(S)
        np.save(cp, R)
        return R
    except OSError:
        return _sqrtm_psd(S)

def _sqrtm_psd(S):
    w, U = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (U * np.sqrt(w)) @ U.T

def generalized_eigh(F, S):
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    Ft = Linv @ F @ Linv.T
    Ft = 0.5 * (Ft + Ft.T)
    w, U = np.linalg.eigh(Ft)
    C = Linv.T @ U
    return w, C

def read_poscar(fn):
    L = open(fn).read().splitlines()
    scale = float(L[1].split()[0])
    cell = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)])
    if scale > 0:
        cell = cell * scale
    else:
        vol = abs(np.linalg.det(cell)); cell = cell * (abs(scale) / vol) ** (1.0 / 3.0)
    tok6 = L[5].split()
    if all(re.fullmatch(r'[A-Za-z]{1,2}', t) for t in tok6):
        species = tok6; counts = [int(x) for x in L[6].split()]; k = 7
    else:
        sys.exit("✗ 该 POSCAR 第6行不是元素符号(疑似 VASP4 格式)。请补上元素符号行(如 'C N H Pb I')。")
    elements = []
    for el, c in zip(species, counts):
        elements += [el] * c
    mode = L[k].strip().lower()
    if mode.startswith('s'):
        k += 1; mode = L[k].strip().lower()
    k += 1
    n = sum(counts)
    raw = np.array([[float(x) for x in L[k + i].split()[:3]] for i in range(n)])
    if mode.startswith('d'):
        cart = raw @ cell
    else:
        cart = raw * (scale if scale > 0 else 1.0)
    return elements, cart, cell

def cmd_combine(a):
    RY = 13.60569253
    E_all = np.loadtxt(a.energy, ndmin=2)
    outdir = a.outdir or a.real_dir
    os.makedirs(outdir, exist_ok=True)
    nsteps = E_all.shape[0]
    for i in range(nsteps):
        idx = a.imin + i
        marker = os.path.join(outdir, f"0_Ham_{idx}.meta.npz")
        if os.path.isfile(marker) and not getattr(a, "force_legacy", False):
            sys.exit(
                f"✗ REFUSE legacy combine: {marker} indicates a V3 full-Hamiltonian result. "
                "Legacy combine would discard Procrustes-induced Re(H)_offdiag. "
                "Use the existing V3 0_Ham files.  Only use --force-legacy for an intentional old-format test."
            )
        E = E_all[i]
        sz = len(E)
        R = np.loadtxt(os.path.join(a.real_dir, f"real{idx:04d}"), ndmin=2)
        assert R.shape == (sz, 2 * sz), f"real{idx:04d} shape {R.shape} != ({sz},{2*sz})"
        re_col = R[:, 0::2]; im_col = R[:, 1::2]
        ham_re = (-1.0 / RY) * im_col
        np.fill_diagonal(ham_re, E / RY)
        ham_im = (-1.0 / RY) * re_col
        np.fill_diagonal(ham_im, 0.0)
        np.savetxt(os.path.join(outdir, f"0_Ham_{idx}_re"), ham_re, fmt="%15.10f")
        np.savetxt(os.path.join(outdir, f"0_Ham_{idx}_im"), ham_im, fmt="%15.10f")
    print(f"LEGACY combine: {nsteps} steps -> {outdir}.  V3 production should not use this path.")

def _comment_block(txt, header, endname):
    m = re.search(r'(?ism)^[ \t]*&' + header + r'\b.*?^[ \t]*&END[ \t]+' + endname + r'\b[^\n]*\n', txt)
    if not m:
        return txt, False
    commented = ''.join(
        (('#' + ln) if (ln.strip() and not ln.lstrip().startswith('#')) else ln) + '\n'
        for ln in m.group(0).rstrip('\n').split('\n'))
    return txt[:m.start()] + commented + txt[m.end():], True

def _uncomment_block(txt, header, endname):
    m = re.search(r'(?ism)^[ \t]*#[ \t]*&' + header + r'\b.*?^[ \t]*#[ \t]*&END[ \t]+' + endname + r'\b[^\n]*\n', txt)
    if not m:
        return txt, False
    uncommented = ''.join(re.sub(r'^([ \t]*)#[ \t]?', r'\1', ln) + '\n'
                          for ln in m.group(0).rstrip('\n').split('\n'))
    return txt[:m.start()] + uncommented + txt[m.end():], True

def set_ao_ndigits(txt, nd):
    m = re.search(r'(?ims)^([ \t]*)&AO_MATRICES\b.*?^[ \t]*&END\b[^\n]*$', txt)
    if not m:
        return txt, False
    blk = m.group(0)
    if re.search(r'(?im)^[ \t]*NDIGITS\b', blk):
        nb = re.sub(r'(?im)^([ \t]*)NDIGITS[ \t]+\d+', lambda mm: f"{mm.group(1)}NDIGITS {nd}", blk)
    else:
        ls = blk.splitlines()
        ind = re.match(r'[ \t]*', ls[0]).group(0)
        ls.insert(1, f"{ind}  NDIGITS {nd}")
        nb = "\n".join(ls)
    return txt[:m.start()] + nb + txt[m.end():], True

def _cp2k_walk(lines):
    ev = []; stack = []
    for i, ln in enumerate(lines):
        s = ln.split('#', 1)[0].split('!', 1)[0].strip()
        if not s.startswith('&'):
            continue
        body = s[1:].strip()
        if body.upper().startswith('END'):
            name = stack.pop() if stack else None
            ev.append((i, 'end', name, tuple(stack)))
        else:
            name = body.split()[0].upper()
            ev.append((i, 'begin', name, tuple(stack)))
            stack.append(name)
    return ev

def _find_section(ev, path):
    parent, name = tuple(path[:-1]), path[-1]
    beg = next((i for i, k, nm, st in ev if k == 'begin' and nm == name and st == parent), None)
    if beg is None:
        return None, None
    end = next((i for i, k, nm, st in ev if k == 'end' and nm == name and st == parent and i > beg), None)
    return beg, end

def inject_mo_cubes(txt, nhomo, nlumo, stride):
    lines = txt.splitlines()
    ev = _cp2k_walk(lines)
    d0, d1 = _find_section(ev, ('FORCE_EVAL', 'DFT'))
    if d0 is None:
        d0, d1 = _find_section(ev, ('DFT',))
    if d0 is None or d1 is None:
        sys.exit("✗ 注入 &MO_CUBES 失败: 模板里没找到完整的 &DFT ... &END(DFT)")
    p0, p1 = _find_section(ev, ('FORCE_EVAL', 'DFT', 'PRINT'))
    ind = re.match(r'[ \t]*', lines[p0 if p0 is not None else d1]).group(0) + '  '
    blk = [f"{ind}&MO_CUBES",
           f"{ind}  NHOMO {nhomo}",
           f"{ind}  NLUMO {nlumo}",
           f"{ind}  WRITE_CUBE T",
           f"{ind}  STRIDE {stride[0]} {stride[1]} {stride[2]}",
           f"{ind}&END MO_CUBES"]
    if p0 is not None:
        m0, m1 = _find_section(ev, ('FORCE_EVAL', 'DFT', 'PRINT', 'MO_CUBES'))
        if m0 is not None:
            lines[m0:m1 + 1] = blk
        else:
            lines[p1:p1] = blk
    else:
        ind = re.match(r'[ \t]*', lines[d1]).group(0) + '  '
        blk = [f"{ind}&PRINT"] + [f"  {b}" for b in blk] + [f"{ind}&END PRINT"]
        lines[d1:d1] = blk
    return "\n".join(lines) + "\n"

def cmd_vasp2cp2k(a):
    elements, cart, cell = read_poscar(a.vasp)
    cell_block = ("  &CELL\n"
                  f"    A  {cell[0,0]:.8f} {cell[0,1]:.8f} {cell[0,2]:.8f}\n"
                  f"    B  {cell[1,0]:.8f} {cell[1,1]:.8f} {cell[1,2]:.8f}\n"
                  f"    C  {cell[2,0]:.8f} {cell[2,1]:.8f} {cell[2,2]:.8f}\n"
                  "    PERIODIC XYZ\n"
                  "  &END CELL")
    coord_block = "  &COORD\n" + "\n".join(
        f"    {el}  {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}" for el, r in zip(elements, cart)) + "\n  &END COORD"

    if not a.template:
        print(cell_block); print(coord_block); return

    txt = open(a.template).read()
    n_cell = len(re.findall(r'(?is)&CELL\b', txt))
    txt, c1 = re.subn(r'(?ism)^[ \t]*&CELL\b.*?&END[^\n]*\n', cell_block + "\n", txt, count=1)
    if c1 == 0:
        sys.exit("✗ 模板里没找到 &CELL ... &END CELL 块")

    txt2, c2 = re.subn(r'(?ism)^[ \t]*&COORD\b.*?&END[^\n]*\n', coord_block + "\n", txt, count=1)
    if c2:
        txt = txt2
    else:
        txt = re.sub(r'(?im)^([ \t]*)(COORD_FILE_FORMAT|COORD_FILE_NAME)\b', r'\1# \2', txt)
        txt, ci = re.subn(r'(?im)^([ \t]*&END[ \t]+CELL[^\n]*\n)', r'\1' + coord_block + "\n", txt, count=1)
        if ci == 0:
            sys.exit("✗ 模板既无 &COORD 块, 也无法在 &END CELL 后插入 &COORD(检查 &CELL/&END CELL)")
        print("  模板原用 COORD_FILE 读坐标 → 已注释 COORD_FILE_*, 改注入 &COORD 内联坐标")
    guess = "   SCF_GUESS ATOMIC" if a.guess == "atomic" else "   SCF_GUESS RESTART"
    txt, cg = re.subn(r'(?im)^\s*SCF_GUESS\b.*$', guess, txt, count=1)
    if cg == 0:
        print("  ⚠ 模板 &SCF 里没有 SCF_GUESS 行, 未设置 guess(第一帧应 ATOMIC, 其余 RESTART)")
    out = a.out or (os.path.splitext(a.vasp)[0] + ".inp")

    if a.lsd is not None:
        if a.lsd:
            if not re.search(r'(?im)^\s*(LSD|UKS)\b', txt):
                txt = re.sub(r'(?im)^([ \t]*&DFT\b[^\n]*\n)', r'\1   LSD\n', txt, count=1)
            if a.multiplicity is not None:
                txt = re.sub(r'(?im)^[ \t]*#?\s*MULTIPLICITY\b[^\n]*\n?', '', txt)
                txt = re.sub(r'(?im)^([ \t]*(?:LSD|UKS)\b[^\n]*)$',
                             r'\1\n   MULTIPLICITY %d' % a.multiplicity, txt, count=1)

        else:
            txt = re.sub(r'(?im)^[ \t]*(LSD|UKS)\b[^\n]*\n?', '', txt)
            txt = re.sub(r'(?im)^[ \t]*#?\s*MULTIPLICITY\b[^\n]*\n?', '', txt)

    if a.scf_mode == "ot":
        txt, ot_ok = _uncomment_block(txt, "OT", "OT")
        if not ot_ok and not re.search(r'(?im)^[ \t]*&OT\b', txt):
            sys.exit("✗ 切 OT 模式失败: 模板 &SCF 里没有 &OT 块(连注释的都没有)。请先在 &SCF 里放一个"
                     "(可注释的)&OT ... &END OT 块, 例如 MINIMIZER CG / PRECONDITIONER FULL_ALL / ENERGY_GAP。")
        txt, _ = _comment_block(txt, "DIAGONALIZATION", "DIAGONALIZATION")
        txt, sm = _comment_block(txt, "SMEAR", "SMEAR")
        txt = re.sub(r'(?im)^([ \t]*)(ADDED_MOS\b[^\n]*)$', r'\1# \2', txt)

        n_ovl, ovl_ok = re.subn(r'(?im)^([ \t]*)OVERLAP[ \t]+T\b', r'\1OVERLAP F', txt)
        if ovl_ok:
            txt = n_ovl
        note_once("scfmode_ot",
                  "  SCF 模式: 【OT】(启用 &OT; 注释 &DIAGONALIZATION"
                  + ("/&SMEAR" if sm else "") + "/ADDED_MOS"
                  + ("; 关闭 AO overlap 打印(OT阶段没人读, 省一份几百MB的废文件)" if ovl_ok else "") + ")。\n"
                  "  ★ OT 的 .wfn 里【只有占据轨道】(nmo=homo) → 这个 wfn 不能直接喂 wfn-nac。\n"
                  "    (注: OT 仍【能】打 LUMO 的 cube/PDOS —— CP2K 会用 ot_eigensolver 在 SCF 后事后算出来;\n"
                  "     只是不写进 wfn。两条路径不是一回事。)\n"
                  "    用途: 快速稳收敛密度, 之后 diag 段(SCF_GUESS RESTART 读此 wfn)对角化出虚轨道。")
    else:

        if a.lsd:
            m1 = re.search(r'(?im)^([ \t]*)ADDED_MOS[ \t]+(-?\d+)[ \t]*$', txt)
            if m1:
                txt = txt[:m1.start()] + f"{m1.group(1)}ADDED_MOS {m1.group(2)} {m1.group(2)}" + txt[m1.end():]
                print(f"  ADDED_MOS 单值({m1.group(2)}) → 双值({m1.group(2)} {m1.group(2)}): LSD 两个自旋通道各需一个,"
                      f" 单值时 beta 通道会拿到原始 -1 → n_mo<电子数 → qs_environment CPASSERT。")
        elif a.lsd is False:
            m2 = re.search(r'(?im)^([ \t]*)ADDED_MOS[ \t]+(-?\d+)[ \t]+(-?\d+)[ \t]*$', txt)
            if m2:
                txt = txt[:m2.start()] + f"{m2.group(1)}ADDED_MOS {m2.group(2)}" + txt[m2.end():]
                print(f"  ADDED_MOS 双值 → 单值({m2.group(2)}): 闭壳层只有一个通道。")

        if a.no_smear:
            txt, sm = _comment_block(txt, "SMEAR", "SMEAR")
            if sm:
                print("  已注释 &SMEAR(按 --no-smear 要求; 展宽本身与 LSD 不冲突, 只是大 gap 体系用不到)")

    if a.lsd is not None:

        qmap = {}
        for km in re.finditer(r'(?is)&KIND\s+(\S+).*?POTENTIAL\s+\S*?q(\d+)', txt):
            qmap[km.group(1)] = int(km.group(2))
        if qmap and all(e in qmap for e in set(elements)):
            nelec = sum(qmap[e] for e in elements) - a.charge
            if a.lsd:
                need = 2 if nelec % 2 else 1
                if a.multiplicity is None:
                    print(f"  自旋: LSD on, 未显式给 MULTIPLICITY → CP2K 用默认值 {need}"
                          f"(价电子≈{nelec}, {'奇' if nelec%2 else '偶'}); 注释掉≠没有多重度, 只是默认。")
                else:
                    ok = (nelec + a.multiplicity - 1) % 2 == 0
                    if ok:
                        print(f"  自旋: LSD on, MULTIPLICITY={a.multiplicity}; 价电子≈{nelec}"
                              f"({'奇' if nelec%2 else '偶'}), 奇偶匹配 ✓")
                    else:
                        print(f"  ⚠ 自旋奇偶【不匹配】: 价电子≈{nelec}({'奇' if nelec%2 else '偶'}) 与 "
                              f"MULTIPLICITY={a.multiplicity} 冲突 → CP2K 报 'LSD: try to use a different"
                              f" multiplicity'。应设 MULTIPLICITY={need}(或 {need}+2, {need}+4…)。")
            elif nelec % 2:
                print(f"  ⚠ 你关了自旋(闭壳层)但价电子≈{nelec} 是【奇数】→ CP2K 必崩(奇电子必须开 LSD)。"
                      f" 请设 SPIN=on。(这就是 VASP 能而 CP2K 不能的那个差异。)")

    if getattr(a, 'no_ao_overlap', False):
        txt = re.sub(r'(?im)^([ \t]*)OVERLAP[ \t]+T\b', r'\1OVERLAP F', txt)
        note_once("no_ao_ovl", "  AO overlap 打印: 【关闭】(这一帧不做要 S 的分析 → 省下几百 MB 的 out"
                  " 和几十秒的 python 解析; wfn-nac 用的是合并体系的 S, 不受影响)")
    elif getattr(a, 'ao_ndigits', 0):
        txt, ok_ = set_ao_ndigits(txt, a.ao_ndigits)
        if ok_:
            note_once("ao_ndig", f"  AO overlap NDIGITS → {a.ao_ndigits} (模板里是16; 每个数 22→14 字符,"
                      f" out 小 36%, 解析同比例加快。实测对 NAC 的偏差比最小耦合小 3 个量级, 安全)")

    if getattr(a, 'mo_cubes', False):
        if a.scf_mode == "ot" and not getattr(a, 'mo_cubes_in_ot', False):

            print("  MO_CUBES: OT 段【默认跳过】(不是打不出——CP2K 会用 ot_eigensolver 事后算 LUMO 再打印,"
                  " 但 ot+diag 的第二段 diag 已经会打一遍, 在这里打是白跑一次)。纯 OT 流程要 cube: --mo-cubes-in-ot")
        else:
            st = getattr(a, 'cube_stride', [2, 2, 2]) or [2, 2, 2]
            txt = inject_mo_cubes(txt, a.nhomo, a.nlumo, st)
            nsp = 2 if a.lsd else 1
            note_once("mocubes", f"  MO_CUBES: NHOMO {a.nhomo} + NLUMO {a.nlumo}, STRIDE {st[0]} {st[1]} {st[2]}"
                      f" → 每帧 {(a.nhomo + a.nlumo) * nsp} 个 cube(自旋通道各一套; 实测单个≈34MB@stride2)。")
            if a.scf_mode == "ot":
                print("  ⚠ OT + MO_CUBES: CP2K 会在 SCF 后用 ot_eigensolver 现算这几条虚轨道(额外开销)。"
                      "另外 OT 的 .wfn 里没有虚轨道 → cube-overlap 用它贴标签时, LUMO 侧拿不到 E/occ(标签仍对)。")

            if re.search(r'(?im)^[ \t]*&SMEAR\b', txt):
                note_once("smear_mocubes",
                          "  ⚠ 模板里 &SMEAR 是【打开】的: CP2K 数 NHOMO/NLUMO 是从 mo_set%homo 往下/往上数,"
                          " 而 SMEAR 会把 homo 撑大(展宽尾巴) → cube 窗口可能整体上移几条。\n"
                          "    cube-overlap 会用 occ_frontier 按 nelectron 贴真实标签(不按下标瞎猜), 并在标签"
                          "错位时报警; 但 NHOMO/NLUMO 建议各多给 2~3 条留余量。")

    open(out, 'w').write(txt)
    print(f"写出 {out}")
    note_once("v2c_out", f"  ({len(elements)} 原子, 胞已注入 A/B/C 三矢量; 后续帧同构, 只报路径)")
    if n_cell > 1:
        print("  ⚠ 模板里出现多处 &CELL, 只替换了第一处——确认没替错")

BOHR2ANG = 0.52917720859
INV_BOHR3_TO_INV_ANG3 = 1.0 / BOHR2ANG ** 3

def _fast_floats(text):
    try:
        v = np.fromstring(text, dtype=np.float64, sep=' ')
    except Exception:
        v = np.array([], dtype=np.float64)
    if v.size == 0 and text.strip():
        v = np.array(text.split(), dtype=np.float64)
    return v

def read_cube_full(fn, dtype=np.float64):
    with open(fn, 'r') as fh:
        head = [fh.readline() for _ in range(6)]
        rest = fh.read()
    t2 = head[2].split()
    natom_raw = int(t2[0]); origin = np.array([float(x) for x in t2[1:4]])
    nval_hdr = int(t2[4]) if len(t2) >= 5 else 1
    natom = abs(natom_raw)
    n = np.zeros(3, dtype=int); h = np.zeros((3, 3))
    for k in range(3):
        tk = head[3 + k].split()
        n[k] = int(tk[0]); h[k] = [float(x) for x in tk[1:4]]
    units = 'bohr'
    if np.any(n < 0):
        units = 'angstrom'
        n = np.abs(n); h = h / BOHR2ANG
    nval = nval_hdr
    npts = int(n[0]) * int(n[1]) * int(n[2])

    vals = _fast_floats(rest)
    need = npts * nval
    if vals.size < need:
        raise ValueError(f"{fn}: 体素数据不够({vals.size} < {need}={n[0]}×{n[1]}×{n[2]}×{nval})。"
                         f"文件可能被截断, 或第4-6行网格数解析错了。前6行:\n" + "".join(head))
    n_extra = vals.size - need
    if n_extra == 5 * natom:
        conv = f"正原子数({natom_raw}), 无轨道行"
    elif n_extra == 5 * natom + 1 + nval:
        conv = f"负原子数({natom_raw})约定, 原子行后多一行轨道编号(已跳过)"
    elif natom_raw < 0 and n_extra > 5 * natom:
        m = n_extra - 5 * natom - 1
        if m >= 1 and (vals.size - (5 * natom + 1 + m)) % npts == 0:
            nval = (vals.size - (5 * natom + 1 + m)) // npts
            need = npts * nval; n_extra = vals.size - need
            conv = f"负原子数({natom_raw})约定, 单文件 {nval} 条轨道"
        else:
            conv = f"未知(前缀 {n_extra} 个数, natom={natom})"
    else:
        conv = f"未知(前缀 {n_extra} 个数, 期望 {5*natom} 或 {5*natom+1+nval})"
    data = vals[-need:].astype(dtype).reshape(int(n[0]), int(n[1]), int(n[2]), nval) if nval > 1 \
        else vals[-need:].astype(dtype).reshape(int(n[0]), int(n[1]), int(n[2]))
    dv = abs(np.linalg.det(h))

    ttl = head[1]
    mh = re.search(r'HOMO\s*-\s*(\d+)', ttl); mm = re.search(r'WAVEFUNCTION\s+(\d+)', ttl)
    homo_hint = (int(mm.group(1)) + int(mh.group(1))) if (mh and mm) else None
    return dict(data=data, n=n, h=h, dv=dv, natom=natom, natom_raw=natom_raw, origin=origin,
                nval=nval, units=units, title=ttl.strip(), conv=conv, homo_hint=homo_hint,
                cell=np.array([n[k] * h[k] for k in range(3)]))

def read_cube(fn):
    d = read_cube_full(fn)
    return d['data'] if d['nval'] == 1 else d['data'][..., 0]

def cmd_overlap(a):
    w = read_cp2k_wfn(a.wfn); sp = w['spins'][a.ispin - 1]
    C = sp['C']; nao = w['nao']
    S = _load_single_frame_S(a.overlap, nao)
    Ct = C[:, a.bmin:a.bmax + 1]
    O = Ct.conj().T @ S @ Ct
    dev = np.abs(O - np.eye(O.shape[0])).max()
    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, 'overlap.dat'), O, fmt='%06e')
    print(f"overlap.dat: {O.shape[0]}×{O.shape[0]} 同帧 C†SC → {a.outdir}/")
    print(f"  ‖C†SC − I‖_max = {dev:.2e}")
    note_once("ovl_expl", "    (这就是 VASP overlap.dat 的高斯基版, 只是中间夹了 S。★ 量级看 NDIGITS:"
              " 16位→~1e-15, 8位→~1e-8(打印舍入, 正常); >1e-3 才是真出问题)")
    if dev > 1e-3:
        print("  ⚠ 偏离单位阵较大: 检查 S 与 wfn 是否同一次计算、AO 排序是否一致、SCF 是否收敛。")

def cmd_ipr(a):
    if a.mode == 'cube':
        if not a.cubes:
            sys.exit("✗ --mode cube 需要 --cubes 给出按 band 顺序的 .cube 文件列表")
        iprs = []
        for fn in a.cubes:
            p = read_cube(fn) ** 2
            iprs.append(np.sum(p ** 2) / (np.sum(p) ** 2))
        iprs = np.array(iprs)
        os.makedirs(a.outdir, exist_ok=True)
        np.savetxt(os.path.join(a.outdir, 'ipr.dat'), iprs[:, None], fmt='%06e')
        print(f"ipr.dat: {len(iprs)} 条(来自 {len(a.cubes)} 个 cube), 实空间, 表象无关 → {a.outdir}/")
        print(f"  IPR∈[{iprs.min():.4f},{iprs.max():.4f}]  N_eff=1/IPR∈[{(1/iprs).min():.1f},{(1/iprs).max():.1f}]")
        return
    if not a.wfn or not a.overlap:
        sys.exit("✗ --mode loewdin 需要 --wfn 和 --overlap")
    w = read_cp2k_wfn(a.wfn); sp = w['spins'][a.ispin - 1]
    C = sp['C']; nao = w['nao']
    bmax = (C.shape[1] - 1) if a.bmax < 0 else a.bmax
    S = _load_single_frame_S(a.overlap, nao)
    Ctil = _sqrtm_psd_cached(S, a.overlap) @ C[:, a.bmin:bmax + 1]
    p = np.abs(Ctil) ** 2
    ipr = np.sum(p ** 2, axis=0) / (np.sum(p, axis=0) ** 2)
    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, 'ipr.dat'), ipr[:, None], fmt='%06e')
    print(f"ipr.dat: {len(ipr)} 条(band {a.bmin}..{bmax}), Löwdin-AO 系数空间 → {a.outdir}/")
    print(f"  IPR∈[{ipr.min():.4f},{ipr.max():.4f}]  有效参与AO数 N_eff=1/IPR∈[{(1/ipr).min():.1f},{(1/ipr).max():.1f}]")
    note_once("ipr_repr", "  ⚠ IPR 依赖表象/基组: CP2K(Löwdin-AO) 不能直接和 VASP(平面波G空间) 比数值,"
              " 只比【沿轨迹的相对趋势】。要表象无关的绝对值请用 --mode cube。")

ALPHA_FS = 1.0 / 137.035999084
H2EV = 27.211386245988
T_AU_S = 2.4188843265857e-17
AU_DIP_TO_DEBYE = 2.5417464519

def cube_grad(psi3d, hmat):
    n = psi3d.shape
    F = np.fft.fftn(psi3d)
    gu = np.empty((3,) + n)
    for q in range(3):
        kk = 2j * np.pi * np.fft.fftfreq(n[q])
        sh = [1, 1, 1]; sh[q] = n[q]
        gu[q] = np.fft.ifftn(F * kk.reshape(sh)).real
    return np.einsum('qp,p...->q...', np.linalg.inv(hmat), gu)

def cmd_cube_tdm(a):
    files = sorted(glob.glob(os.path.join(a.cube_dir, a.glob)))
    if not files:
        sys.exit(f"✗ {a.cube_dir}/{a.glob} 没有 cube 文件。")
    groups = {}
    for f in files:
        mm_ = re.search(r'WFN_(\d+)_(\d+)', os.path.basename(f))
        if not mm_:
            continue
        groups.setdefault(int(mm_.group(2)), []).append((int(mm_.group(1)), int(mm_.group(2)), f))
    if not groups:
        sys.exit("✗ 没有文件名形如 *WFN_<MO>_<spin>-*.cube 的 cube")

    meta = None
    if a.wfn and os.path.isfile(a.wfn):
        try:
            meta = read_cp2k_wfn(a.wfn, meta_only=True)
        except Exception as e:
            print(f"  ⚠ 读 .wfn 失败({e}) → 没有本征值就【算不了】速度规范(要 ΔE)。")
    if meta is None:
        sys.exit("✗ cube-tdm 必须给 --wfn: 速度规范 μ=<i|∇|j>/ΔE 需要本征值 E_i。")

    tag = a.tag or os.path.basename(os.path.normpath(a.cube_dir))
    for sub in ('tdm', 'fosc', 'rad'):
        os.makedirs(os.path.join(a.outdir, sub), exist_ok=True)
    note_once("tdm_gauge",
              "  ★ 周期体系用【速度规范】 μ=<i|∇|j>/(E_j−E_i): 位置算符 r 在 PBC 下没有定义,\n"
              "    直接用 r 算的 <i|r|j> 依赖原点位置, 是个没意义的数。∇ψ 是周期的 → 良定义。")
    rows = []
    for spin in sorted(groups):
        items = sorted(groups[spin], key=lambda x: (x[1], x[0])); m = len(items)
        sname = "all" if spin == 0 else str(spin)
        isp = 0 if spin == 0 else spin - 1
        sp = meta['spins'][min(isp, len(meta['spins']) - 1)]
        eig = sp['eig']; occ = sp['occ']
        ifr, _ = occ_frontier(sp)
        homo_true = ifr + 1
        print(f"\n── spin{sname}: {m} 个 cube, MO {items[0][0]}..{items[-1][0]} ──")

        P = None; E = np.zeros(m); ref = None
        for k, (mo, _, path) in enumerate(items):
            d = read_cube_full(path, dtype=np.float64)
            if ref is None:
                ref = d; n3 = tuple(int(x) for x in d['n']); npts = int(np.prod(d['n']))
                P = np.empty((m, npts))
            elif not np.array_equal(d['n'], ref['n']) or not np.allclose(d['h'], ref['h'], rtol=1e-8, atol=1e-12):
                sys.exit(f"✗ {os.path.basename(path)} 网格与第一个 cube 不一致。")
            psi = d['data'].ravel()
            nrm = float(np.dot(psi, psi) * d['dv'])
            P[k] = psi / np.sqrt(nrm)
            if mo - 1 < len(eig):
                E[k] = eig[mo - 1]
            del d, psi
        dv = ref['dv']; h = ref['h']

        O = (P @ P.T) * dv
        offd = np.abs(O - np.diag(np.diag(O))).max()
        print(f"  自检① <ψ_i|ψ_j>: 对角 ∈[{np.diag(O).min():.6f},{np.diag(O).max():.6f}](应=1); "
              f"非对角最大 {offd:.2e}(应=0 —— 正交性)")
        note_once("tdm_ortho",
                  "    ★ 非对角≈0 正是 '∫ψ_iψ_j dr 恒为 0' 的直接证据: 同一次 SCF 的本征态两两正交。\n"
                  "      所以拿它当'轨道重叠'是没有意义的; 空间共存度请用 cube-overlap(eh_amp/eh_cos)。")
        if offd > 1e-3:
            print(f"    ⚠ 正交性偏差 {offd:.1e} 偏大: STRIDE 抽太狠(网格太粗)? cube 混了不同帧?")

        V = np.zeros((m, m, 3))
        for j in range(m):
            g = cube_grad(P[j].reshape(n3), h)
            V[:, j, :] = (P @ g.reshape(3, -1).T) * dv
            del g

        asym = np.abs(V + np.transpose(V, (1, 0, 2))).max()
        diag0 = np.abs(V[np.arange(m), np.arange(m), :]).max()
        print(f"  自检② <i|∇|j>+<j|∇|i> 最大 {asym:.2e}(应=0, 反对称);  "
              f"自检③ <i|∇|i> 最大 {diag0:.2e}(应=0)")
        if asym > 1e-3 * max(np.abs(V).max(), 1e-30):
            print("    ⚠ 反对称性破得厉害 → 网格太粗(FFT 看到了不连续)或 cube 不是同一套网格。")

        dE = E[None, :] - E[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            MU = V / dE[:, :, None]
        near = np.abs(dE) < a.degen_eps
        MU[near] = np.nan
        mu2 = (MU ** 2).sum(-1)
        F = (2.0 / 3.0) * np.abs(dE) * mu2
        A = (4.0 / 3.0) * ALPHA_FS ** 3 * np.abs(dE) ** 3 * mu2 / T_AU_S
        nd = int(near.sum() - m)
        if nd > 0:
            print(f"  ⚠ {nd} 对态的 |ΔE| < {a.degen_eps*H2EV*1000:.1f} meV(近简并) → μ 发散, 已置 NaN。")

        labels = [("HOMO" if mo == homo_true else f"HOMO-{homo_true-mo}") if mo <= homo_true
                  else ("LUMO" if mo == homo_true + 1 else f"LUMO+{mo-homo_true-1}")
                  for (mo, _, _) in items]
        hdr = (f"frame={tag} spin={sname}  行/列 = MO " +
               " ".join(f"{mo}({lb})" for (mo, _, _), lb in zip(items, labels)))
        np.savetxt(os.path.join(a.outdir, 'tdm', f'tdm.{tag}.spin{sname}.dat'),
                   np.sqrt(mu2), fmt='%.6e',
                   header=hdr + "\n|μ_ij| = |<i|∇|j>|/|E_j-E_i|, 速度规范, 单位 e·bohr"
                          f"  (×{AU_DIP_TO_DEBYE:.6f} = Debye)"
                          "\n★ 对角无意义(ΔE=0); 近简并态对为 NaN(μ 真的发散, 不是 bug)")
        np.savetxt(os.path.join(a.outdir, 'fosc', f'fosc.{tag}.spin{sname}.dat'), F, fmt='%.6e',
                   header=hdr + "\n振子强度 f_ij = (2/3)·ΔE·|μ|² (原子单位, 无量纲)")
        with open(os.path.join(a.outdir, 'rad', f'rad.{tag}.spin{sname}.txt'), 'w') as fh:
            fh.write(f"# {hdr}\n# i j  ΔE[eV]  |μ|[e·bohr]  |μ|[Debye]  f_osc  A[1/s]  τ_rad[s]\n")
            for i in range(m):
                for j in range(i + 1, m):
                    if not np.isfinite(mu2[i, j]):
                        continue
                    tau = 1.0 / A[i, j] if A[i, j] > 0 else float('inf')
                    fh.write(f"{labels[i]:>8s} {labels[j]:>8s} {abs(dE[i,j])*H2EV:9.4f} "
                             f"{np.sqrt(mu2[i,j]):11.4e} {np.sqrt(mu2[i,j])*AU_DIP_TO_DEBYE:11.4e} "
                             f"{F[i,j]:11.4e} {A[i,j]:11.4e} {tau:11.4e}\n")

        ih = [k for k, (mo, _, _) in enumerate(items) if mo == homo_true]
        il = [k for k, (mo, _, _) in enumerate(items) if mo == homo_true + 1]
        row = dict(frame=tag, spin=sname, n_cube=m)
        if ih and il:
            i0, j0 = ih[0], il[0]
            g = abs(dE[i0, j0]) * H2EV
            mu = np.sqrt(mu2[i0, j0]); f_ = F[i0, j0]; A_ = A[i0, j0]
            tau = 1.0 / A_ if A_ > 0 else float('inf')
            print(f"  ★ HOMO→LUMO: ΔE={g:.4f} eV  |μ|={mu:.4e} e·bohr ({mu*AU_DIP_TO_DEBYE:.4e} D)")
            print(f"     振子强度 f={f_:.4e}   辐射速率 A={A_:.4e} /s   辐射寿命 τ={tau:.4e} s"
                  f" ({tau*1e9:.2f} ns)")
            row.update(dE_eV=g, mu_au=mu, mu_debye=mu * AU_DIP_TO_DEBYE, f_osc=f_,
                       A_rate_s=A_, tau_rad_s=tau)
        rows.append(row)

    if rows:
        import csv
        cf = os.path.join(a.outdir, 'summary.csv')
        new = not os.path.exists(cf)
        keys = sorted({k for r in rows for k in r})
        with open(cf, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)
    note_once("tdm_write", f"写出 → {a.outdir}/{{tdm,fosc,rad}}/  和 {a.outdir}/summary.csv")

def cmd_cube_overlap(a):
    files = sorted(glob.glob(os.path.join(a.cube_dir, a.glob)))
    if not files:
        sys.exit(f"✗ {a.cube_dir}/{a.glob} 没有 cube 文件。(CP2K 没打印? 已被清理? 文件名不是 *WFN_*.cube?)")
    groups = {}
    for f in files:
        m = re.search(r'WFN_(\d+)_(\d+)', os.path.basename(f))
        if not m:
            print(f"  跳过(文件名认不出 MO/自旋): {os.path.basename(f)}"); continue
        groups.setdefault(int(m.group(2)), []).append((int(m.group(1)), int(m.group(2)), f))
    if not groups:
        sys.exit("✗ 没有文件名形如 *WFN_<MO>_<spin>-*.cube 的 cube")

    meta = None
    if a.wfn and os.path.isfile(a.wfn):
        try:
            meta = read_cp2k_wfn(a.wfn, meta_only=True)
        except Exception as e:
            print(f"  ⚠ 读 wfn 失败({e}) → 不贴 HOMO/LUMO 标签, 只按 MO 编号出矩阵")
    elif a.wfn:
        print(f"  ⚠ 没有 {a.wfn} → 不贴 HOMO/LUMO 标签(强烈建议给 --wfn: SMEAR 下 cube 窗口会整体偏移)")

    tag = a.tag or os.path.basename(os.path.normpath(a.cube_dir))
    for sub in ('eh_dens', 'eh_amp', 'eh_cos', 'index'):
        os.makedirs(os.path.join(a.outdir, sub), exist_ok=True)
    if a.cross_spin and len(groups) > 1:
        groups[0] = sorted(sum(groups.values(), []), key=lambda x: (x[1], x[0]))
    summary_rows = []

    def frontier_of(isp):
        if meta is None or not (1 <= isp <= meta['nspin']):
            return None, None
        sp = meta['spins'][isp - 1]
        ifr, maxocc = occ_frontier(sp)
        return ifr + 1, sp

    for spin in sorted(groups):
        items = sorted(groups[spin], key=lambda x: (x[1], x[0])); m = len(items)
        sname = "all" if spin == 0 else str(spin)
        print(f"\n── spin{sname}: {m} 个 cube, MO {items[0][0]}..{items[-1][0]} ──")
        A = None; norms = []; ref = None; homo_cp2k = None
        for k, (mo, isp, path) in enumerate(items):
            d = read_cube_full(path, dtype=np.float64)
            if ref is None:
                ref = d
                npts = int(np.prod(d['n']))
                A = np.empty((m, npts), dtype=(np.float32 if a.dtype == 'float32' else np.float64))
                cellA = d['cell'] * BOHR2ANG
                note_once("cube_grid",
                          f"  网格 {d['n'][0]}×{d['n'][1]}×{d['n'][2]} = {npts:,} 点; 体元 dV={d['dv']:.6e} bohr³;"
                          f" 胞 |a|,|b|,|c| = {np.linalg.norm(cellA,axis=1)[0]:.3f},"
                          f" {np.linalg.norm(cellA,axis=1)[1]:.3f}, {np.linalg.norm(cellA,axis=1)[2]:.3f} Å"
                          + f"  (变胞时逐帧会变, 精确值见 summary.csv 的 V_ang3)")
                note_once("cube_conv", f"  cube 头约定: {d['conv']}; 单位 {d['units']}; 内存 A ≈ "
                      f"{m*npts*(4 if a.dtype=='float32' else 8)/2**20:.0f} MB")
            else:
                if not np.array_equal(d['n'], ref['n']) or not np.allclose(d['h'], ref['h'], rtol=1e-8, atol=1e-12):
                    sys.exit(f"✗ {os.path.basename(path)} 的网格/体素矢量和第一个 cube 不一致 → 不能同格点积分。"
                             f"(同一帧的 cube 必须同网格; 是不是混进了别的帧的 cube?)")
            if d.get('homo_hint'):
                homo_cp2k = d['homo_hint']
            psi = d['data'].ravel()
            nrm = float(np.dot(psi, psi) * d['dv'])
            norms.append(nrm)
            A[k] = (np.abs(psi) / np.sqrt(nrm)).astype(A.dtype)
            del d, psi
        dv = ref['dv']; norms = np.array(norms)

        S_amp = np.zeros((m, m)); S_dens = np.zeros((m, m))
        chunk = max(1, int(8e6 // max(m, 1)))
        for s0 in range(0, npts, chunk):
            Ac = A[:, s0:s0 + chunk].astype(np.float64)
            S_amp += Ac @ Ac.T
            Bc = Ac * Ac
            S_dens += Bc @ Bc.T
        S_amp *= dv; S_dens *= dv
        dg = np.sqrt(np.diag(S_dens)); S_cos = S_dens / np.outer(dg, dg)
        del A

        print(f"  自检 ∫|ψ|²dV(归一化前) ∈ [{norms.min():.4f}, {norms.max():.4f}]  (应≈1.0)")
        if np.abs(norms - 1).max() > 0.05:
            print(f"    ⚠ 偏离 1 超过 5%: STRIDE 抽稀太狠(网格太粗)→ 减小 STRIDE; 若偏离极大则是 cube 解析出错。")

        if np.abs(np.diag(S_amp) - 1).max() > 1e-6:
            print(f"    ⚠ eh_amp 对角偏离 1: {np.abs(np.diag(S_amp)-1).max():.2e}(不该发生, 报我)")

        SP = "αβ"
        labels = []; eig_l = []; occ_l = []
        for (mo, isp, _) in items:
            ht, sp = frontier_of(isp)
            pre = (SP[isp - 1] + ":") if spin == 0 else ""
            if ht is None:
                labels.append(f"{pre}MO{mo}"); eig_l.append(None); occ_l.append(None)
            else:
                lb = ("HOMO" if mo == ht else f"HOMO-{ht-mo}") if mo <= ht else \
                     ("LUMO" if mo == ht + 1 else f"LUMO+{mo-ht-1}")
                labels.append(pre + lb)
                if mo - 1 < sp['nmo']:
                    eig_l.append(sp['eig'][mo - 1] * 27.211386245988); occ_l.append(sp['occ'][mo - 1])
                else:
                    eig_l.append(None); occ_l.append(None)
        homo_true = None; homo_src = ""
        if spin != 0:
            homo_true, sp = frontier_of(spin)
            if homo_true is None and homo_cp2k is not None:

                homo_true = homo_cp2k; homo_src = "cube_title"
                labels = [("HOMO" if mo == homo_true else f"HOMO-{homo_true-mo}") if mo <= homo_true else
                          ("LUMO" if mo == homo_true + 1 else f"LUMO+{mo-homo_true-1}") for mo, _, _ in items]
                note_once("cube_frontier_title",
                          f"  前沿(来自 cube 标题, CP2K 自报 homo={homo_cp2k}): HOMO=MO{homo_true}, LUMO=MO{homo_true+1}\n"
                          f"    ⚠ 没给 --wfn → 这个 homo 就是 SMEAR 会撑大的那个字段, 无法交叉验证。"
                          f"给上 --wfn 才能用占据数定真前沿。")
            elif homo_true is not None:
                homo_src = "occ"
                _, maxocc = occ_frontier(sp)
                note_once("cube_frontier", f"  真前沿(按占据数): HOMO=MO{homo_true}, LUMO=MO{homo_true+1}"
                          f"  [wfn homo 字段={sp['homo']}, nelectron={sp['nelectron']}, maxocc={maxocc:.0f}"
                          + (f", cube 标题自报 homo={homo_cp2k}" if homo_cp2k else "") + "]")
                if homo_cp2k is not None and homo_cp2k == homo_true:
                    note_once("cube_homo_ok",
                              "    ✓ CP2K 数 cube 窗口用的 homo 与真前沿一致 → 这批 cube 就是你要的那几条轨道。")
                if sp['homo'] != homo_true:
                    print(f"    ⚠★ SMEAR 把 wfn 的 homo 字段撑大了 {sp['homo']-homo_true} 条 → CP2K 打 cube 时"
                          f" NHOMO/NLUMO 是围着 MO{sp['homo']} 数的, 【整个 cube 窗口向上偏了"
                          f" {sp['homo']-homo_true} 条】。本命令已按真前沿贴标签, 结果是对的; 但下次请把 NHOMO"
                          f" 多给 {sp['homo']-homo_true}+2 条, 或在模板里关掉 &SMEAR。")
                mos = [mo for mo, _, _ in items]
                if homo_true not in mos or homo_true + 1 not in mos:
                    print(f"    ⚠★ 真 HOMO(MO{homo_true}) 或真 LUMO(MO{homo_true+1}) 【不在这批 cube 里】"
                          f"(窗口 MO{mos[0]}..{mos[-1]})→ 这一帧算不出真正的 e-h overlap! 加大 NHOMO/NLUMO 重跑。")
                    homo_true = None

        sfx = f".spin{sname}"
        hdr_common = (f"frame={tag} spin={sname}  行/列 = MO " +
                      " ".join(f"{mo}({lb})" for (mo, _, _), lb in zip(items, labels)))
        np.savetxt(os.path.join(a.outdir, 'eh_dens', f'eh_dens.{tag}{sfx}.dat'), S_dens, fmt='%.6e',
                   header=hdr_common + "\n∫ρ_i(r)ρ_j(r)dr, 各自归一(∫ρdV=1), 单位 bohr^-3"
                          f"  (×{INV_BOHR3_TO_INV_ANG3:.6f} = Å^-3)"
                          "\n★ 对角元【不是 1】! ρ 已归一 → [ρ]=bohr^-3 → ∫ρ²dV 的量纲是 bohr^-3,"
                          " 无量纲的 1 根本装不下它。"
                          "\n  对角元 = 1/V_eff (有效体积的倒数) = 离域程度: 数越小 = 密度铺得越开。"
                          "\n  要那个'相同的东西=1'的无量纲量, 请看 eh_cos/ 或 eh_amp/ (它们对角严格=1)。"
                          "\n  另: 两个对角元不等时, 非对角元【可以】大于其中较小的那个"
                          " —— 只要不破 Cauchy-Schwarz |D_ij|≤√(D_ii·D_jj) 就没问题。")
        np.savetxt(os.path.join(a.outdir, 'eh_amp', f'eh_amp.{tag}{sfx}.dat'), S_amp, fmt='%.6e',
                   header=hdr_common + "\n∫|ψ_i(r)||ψ_j(r)|dr, 无量纲, ∈[0,1], 对角=1")
        np.savetxt(os.path.join(a.outdir, 'eh_cos', f'eh_cos.{tag}{sfx}.dat'), S_cos, fmt='%.6e',
                   header=hdr_common + "\neh_dens 的归一(余弦)版, 无量纲, 对角=1, 不随胞体积漂移(变胞首选)")
        with open(os.path.join(a.outdir, 'index', f'index.{tag}{sfx}.txt'), 'w') as fh:
            fh.write(f"# frame {tag}  spin {sname}  {m} 个 cube\n")
            fh.write("# row  MO(1基,=cube文件名) band(0基,对齐BMIN/BMAX) spin   label      E[eV]      occ    ∫|ψ|²dV\n")
            for k, ((mo, isp, path), lb) in enumerate(zip(items, labels)):
                e = f"{eig_l[k]:11.5f}" if eig_l[k] is not None else "         --"
                o = f"{occ_l[k]:8.4f}" if occ_l[k] is not None else "      --"
                fh.write(f"{k:5d} {mo:8d} {mo-1:12d} {isp:6d}  {lb:>11s} {e} {o}  {norms[k]:9.5f}\n")

        Vang = abs(np.linalg.det(ref['cell'])) * BOHR2ANG ** 3
        veff = 1.0 / (np.diag(S_dens) * INV_BOHR3_TO_INV_ANG3)
        note_once("cube_ruler", f"  标尺: 胞 V={Vang:.0f} Å³; 均匀极限 1/V={1/Vang:.3e} Å⁻³。"
                  f"下面 V_eff=1/∫ρ²dV 是各态的有效体积(反参与体积), 括号是占全胞的比例。")
        print("  V_eff: " + "  ".join(f"{lb}:{v:.0f}Å³({v/Vang*100:.1f}%)" for lb, v in zip(labels, veff)))

        iu = np.triu_indices(m, 1)
        kmax = np.argmax(S_cos[iu]); i0, j0 = iu[0][kmax], iu[1][kmax]
        print(f"  最强共存态对(eh_cos): {labels[i0]}–{labels[j0]}  cos={S_cos[i0,j0]:.4f}"
              f"  dens={S_dens[i0,j0]*INV_BOHR3_TO_INV_ANG3:.4e} Å⁻³  amp={S_amp[i0,j0]:.4f}")
        row = dict(frame=tag, spin=sname, n_cube=m, mo_min=items[0][0], mo_max=items[-1][0],
                   homo_src=homo_src, norm_min=norms.min(), norm_max=norms.max())
        if homo_true is not None:
            ih = [k for k, (mo, _, _) in enumerate(items) if mo == homo_true][0]
            il = [k for k, (mo, _, _) in enumerate(items) if mo == homo_true + 1][0]
            dHL = S_dens[ih, il] * INV_BOHR3_TO_INV_ANG3
            print(f"  ★ HOMO–LUMO(电子-空穴): eh_dens = {S_dens[ih,il]:.6e} bohr⁻³ = {dHL:.6e} Å⁻³"
                  f" = {dHL*Vang:.2f}×(1/V) ;  eh_amp = {S_amp[ih,il]:.6f} ;  eh_cos = {S_cos[ih,il]:.6f}")
            row.update(homo_mo=homo_true, lumo_mo=homo_true + 1,
                       eh_dens_HL_bohr=S_dens[ih, il], eh_dens_HL_ang=dHL,
                       eh_amp_HL=S_amp[ih, il], eh_cos_HL=S_cos[ih, il],
                       enh_HL=dHL * Vang, V_ang3=Vang, veff_homo=veff[ih], veff_lumo=veff[il])
        summary_rows.append(row)

    cols = ["frame", "spin", "n_cube", "mo_min", "mo_max", "homo_mo", "lumo_mo", "homo_src",
            "eh_dens_HL_bohr", "eh_dens_HL_ang", "eh_amp_HL", "eh_cos_HL",
            "enh_HL", "V_ang3", "veff_homo", "veff_lumo", "norm_min", "norm_max"]
    csv = os.path.join(a.outdir, "summary.csv")
    old = {}
    if os.path.isfile(csv):
        for ln in open(csv).read().splitlines()[1:]:
            p = ln.split(',')
            if len(p) == len(cols):
                old[(p[0], p[1])] = ln
    for r in summary_rows:
        vals = [str(r.get(c, '')) if not isinstance(r.get(c), float) else f"{r[c]:.6e}" for c in cols]
        old[(str(r['frame']), str(r['spin']))] = ",".join(vals)
    with open(csv, 'w') as fh:
        fh.write(",".join(cols) + "\n")
        for k in sorted(old):
            fh.write(old[k] + "\n")
    note_once("cube_out", f"\n写出 → {a.outdir}/{{eh_dens,eh_amp,eh_cos,index}}/*.<frame>.spin*.{{dat,txt}}"
              f" 和 {csv}   (每帧追加, 后续不再逐帧提示)")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--xyz", required=True)
        p.add_argument("--basis", default="gth-dzvp-molopt-sr")
        p.add_argument("--pseudo", default="gth-pbe")
        p.add_argument("--xc", default="pbe")
        p.add_argument("--verbose", type=int, default=0)

    ps = sub.add_parser("selftest", help="单帧验证 C†SC=I(证明必须夹 S)")
    add_common(ps); ps.add_argument("--frame", type=int, default=0)

    pn = sub.add_parser("nac", help="两帧 time-overlap NAC")
    add_common(pn)
    pn.add_argument("--frame-t", type=int, required=True)
    pn.add_argument("--frame-tdt", type=int, required=True)
    pn.add_argument("--dt", type=float, default=1.0, help="离子步 [fs]")
    pn.add_argument("--bmin", type=int, required=True, help="活性空间起始band(0基)")
    pn.add_argument("--bmax", type=int, required=True, help="活性空间结束band(0基,含)")
    pn.add_argument("--outdir", default="nac_out")

    pw = sub.add_parser("wfn-nac", help="从两个 CP2K .wfn + AO overlap 算 NAC")
    pw.add_argument("--wfn-t", required=True, help="帧 t 的 PROJECT-RESTART.wfn")
    pw.add_argument("--wfn-tdt", required=True, help="帧 t+dt 的 .wfn")
    pw.add_argument("--overlap", required=True,
                    help="AO overlap: .npy(自己转好) 或 CP2K AO_MATRICES 打印文本。"
                         "单帧S(N×N, 原子只挪盒子) 或 合并体系S(2N×2N, 取非对角块作cross-time)")
    pw.add_argument("--dt", type=float, default=1.0, help="离子步 [fs]")
    pw.add_argument("--bmin", type=int, required=True, help="活性空间起始 MO(0基)")
    pw.add_argument("--bmax", type=int, required=True, help="活性空间结束 MO(0基,含)")
    pw.add_argument("--ispin", type=int, default=1)
    pw.add_argument("--track", choices=["none", "perm"], default="perm",
                    help="态跟踪: perm(默认)=Hungarian 最大匹配定置换(跨对累计, 修换序/交叉); none=按下标硬配(旧行为)")
    pw.add_argument("--allow-window-off", action="store_true",
                    help="放行'活性窗没骑在占据前沿上'(只有研究非带边窗口时才该用, 如导带内热载流子弛豫)")
    pw.add_argument("--allow-incomplete", action="store_true",
                    help="放行不完整的 AO overlap 矩阵(默认硬报错; 截断的 out 会静默产垃圾 NAC)")
    pw.add_argument("--outdir", default="nac_out")
    pw.add_argument("--legacy-tag", default=None, help="旧链兼容输出 realNNNN 的编号(如 0003); 默认 0001")
    pw.add_argument("--terse", action="store_true", help="只打逐步必看的关键行(帧信息/自检数值/相位/NAC量级/写出文件); "
                     "省略对角健康检查的逐态详表(仍会算, 有问题时用不加--terse重跑单对看细节)")
    pw.add_argument("--degen-thresh", type=float, default=0.050,
                    help="V3近简并判据[eV]: 仅在同一 V 或 C manifold 内聚类并做 Procrustes；永不跨 band gap。")
    pw.add_argument("--no-procrustes", action="store_true",
                    help="关掉簇内 Procrustes(退回只有 Hungarian 置换 + 符号), 做对照用")
    pw.add_argument("--no-phase", action="store_true", help="跳过相位校正(raw, =Fortran normal); 不给则默认做相位校正(=nac.py phase_corr)")

    pcb = sub.add_parser("combine", help="py3 版 combine_old_VASP: energy(eV,每行一步)+real%%04d(−ħd,eV) → Ham_*_re/im(Ry, 老PYXAID)")
    pcb.add_argument("--energy", required=True, help="energy 文件(每行一个时刻, 每列一个态, eV)")
    pcb.add_argument("--real-dir", required=True, help="realNNNN 所在目录")
    pcb.add_argument("--imin", type=int, required=True, help="起始编号(real%%04d 与 Ham 编号从此起)")
    pcb.add_argument("--outdir", default=None, help="Ham 输出目录(默认= real-dir)")
    pcb.add_argument("--force-legacy", action="store_true",
                     help="允许 legacy combine 覆盖 V3 Ham（通常不要用；会丢 G^T E G 实部非对角）")

    pm = sub.add_parser("build-merged", help="读两帧 out 坐标, 构造 96 原子合并体系算 cross-time S")
    pm.add_argument("--out-t", required=True, help="帧 t 的 CP2K out(需 MEDIUM 打印坐标)")
    pm.add_argument("--out-tdt", required=True, help="帧 t+dt 的 CP2K out")
    pm.add_argument("--ref-input", required=True, help="你的 input.inp(提取 basis/potential/KIND)")
    pm.add_argument("--cell-choice", default="average", choices=["t", "tdt", "average"],
                    help="合并体系用哪帧晶格(镜像求和用; 默认平均)")
    pm.add_argument("--cell-t", type=float, nargs=3, default=None, help="手动给帧t的ABC(out无CELL时)")
    pm.add_argument("--cell-tdt", type=float, nargs=3, default=None, help="手动给帧t+dt的ABC")
    pm.add_argument("--outdir", default="merged")
    pm.add_argument("--ndigits", type=int, default=8,
                    help="合并 out 里 OVERLAP 的打印位数(默认8)。6912² 个数: 16位≈1.3GB, 8位≈0.85GB; "
                         "实测8位引入的 NAC 偏差比最小耦合还小3个量级")
    pm.add_argument("--quiet", action="store_true", help="跳过手动'下一步'提示(给自动化流水线用, 因为脚本自己就会接着跑那两步)")

    ph = sub.add_parser("hvib", help="整条轨迹: 逐帧wfn+逐对overlap → PYXAID Ham_*_re/im(抽取自nano-qmflows)")
    ph.add_argument("--wfns", nargs="+", required=True, help="按时间顺序的 .wfn 列表(N 帧)")
    ph.add_argument("--overlaps", nargs="+", required=True,
                    help="按时间顺序的 N-1 个逐对 cross-time overlap(.npy 或合并体系 CP2K out)")
    ph.add_argument("--dt", type=float, required=True, help="离子步 [fs]")
    ph.add_argument("--bmin", type=int, required=True, help="活性窗起始 MO(0基)")
    ph.add_argument("--bmax", type=int, required=True, help="活性窗结束 MO(0基,含)")
    ph.add_argument("--algorithm", choices=["3points", "levine"], default="3points")
    ph.add_argument("--tracking", action="store_true", help="开态跟踪(min-cost 追 unavoided crossing)")
    ph.add_argument("--ispin", type=int, default=1)
    ph.add_argument("--outdir", default="hamiltonians")

    po = sub.add_parser("overlap", help="单帧 band-band overlap C†SC → overlap.dat(对齐 VASP Ipr_Overlap.py; 体检)")
    po.add_argument("--wfn", required=True, help="该帧的 .wfn")
    po.add_argument("--overlap", required=True, help="该帧单帧 AO overlap(.npy 或 CP2K AO_MATRICES 文本)")
    po.add_argument("--bmin", type=int, required=True, help="起始 MO(0基)")
    po.add_argument("--bmax", type=int, required=True, help="结束 MO(0基,含)")
    po.add_argument("--ispin", type=int, default=1)
    po.add_argument("--outdir", default=".")

    pi = sub.add_parser("ipr", help="KS 态局域化 IPR → ipr.dat(loewdin: 系数空间夹S; cube: 实空间表象无关)")
    pi.add_argument("--mode", choices=["loewdin", "cube"], default="loewdin")
    pi.add_argument("--wfn", help="[loewdin] 该帧 .wfn")
    pi.add_argument("--overlap", help="[loewdin] 该帧单帧 AO overlap(.npy 或 CP2K AO_MATRICES 文本)")
    pi.add_argument("--cubes", nargs="+", help="[cube] 按 band 顺序的 .cube 文件列表")
    pi.add_argument("--bmin", type=int, default=0, help="[loewdin] 起始 MO(0基)")
    pi.add_argument("--bmax", type=int, default=-1, help="[loewdin] 结束 MO(0基,含)")
    pi.add_argument("--ispin", type=int, default=1)
    pi.add_argument("--outdir", default=".")

    pc = sub.add_parser("check", help="单帧体检: nao/nmo/homo/gap/smearing crisp度; 可选--overlap做C†SC=I自检; 可选--out核对SCF收敛/原子数")
    pc.add_argument("--wfn", required=True, help="该帧 .wfn")
    pc.add_argument("--overlap", default=None, help="[可选] 该帧单帧 AO overlap(.npy 或 CP2K AO_MATRICES 文本)")
    pc.add_argument("--out", default=None, help="[可选] 该帧 CP2K out(核对 SCF 收敛状态/CELL/原子数)")
    pc.add_argument("--bmin", type=int, default=None, help="[可选,配--overlap] 自检起始 MO(0基); 默认全部")
    pc.add_argument("--bmax", type=int, default=None, help="[可选,配--overlap] 自检结束 MO(0基,含); 默认全部")
    pc.add_argument("--strict", action="store_true",
                    help="活性窗[--bmin..--bmax]若没骑在占据前沿上就【非零退出】(driver 开跑前预检用)")

    pod = sub.add_parser("ot-diag", help="OT收敛后对KS矩阵事后广义对角化→含虚轨道的正则MO(移植 nac-release 的 OT→虚轨道)")
    pod.add_argument("--ks", required=True, help="KS 矩阵: CP2K &AO_MATRICES KOHN_SHAM 打印文本 或 .npy")
    pod.add_argument("--overlap", required=True, help="overlap: CP2K &AO_MATRICES OVERLAP 打印文本 或 .npy")
    pod.add_argument("--ks-header", default="KOHN-SHAM MATRIX", help="KS 矩阵在 out 里的块头名(默认 'KOHN-SHAM MATRIX')")
    pod.add_argument("--homo", type=int, default=None, help="[可选] 占据轨道数(报 gap/虚轨道数用)")
    pod.add_argument("--outdir", default=".")

    pv = sub.add_parser("vasp2cp2k", help="VASP .vasp/POSCAR → CP2K 输入(注入每帧三斜胞 A/B/C 矢量+坐标); 变胞必需")
    pv.add_argument("--vasp", required=True, help="输入 .vasp/POSCAR")
    pv.add_argument("--template", default=None, help="CP2K 模板(含 &CELL...&END CELL 和 &COORD...&END COORD); 不给则只打印片段")
    pv.add_argument("--out", default=None, help="输出 .inp(默认 <vasp基名>.inp)")
    pv.add_argument("--guess", choices=["restart", "atomic"], default="restart", help="SCF_GUESS: 第一帧用 atomic, 其余 restart")
    pv.add_argument("--lsd", dest="lsd", action="store_true", default=None, help="强制开自旋极化(LSD)")
    pv.add_argument("--no-lsd", dest="lsd", action="store_false", help="强制闭壳层(去掉 LSD/MULTIPLICITY)")
    pv.add_argument("--multiplicity", type=int, default=None, help="[可选]显式多重度; 不给则模板原样(CP2K默认: 奇电子2/偶电子1, 真实out实测)")
    pv.add_argument("--charge", type=int, default=0, help="体系净电荷(用于电子数奇偶自检; 默认0)")
    pv.add_argument("--no-smear", action="store_true", help="注释掉模板里的 &SMEAR(固定 MULTIPLICITY 时必须, 否则 CP2K 在 qs_environment CPASSERT)")
    pv.add_argument("--scf-mode", choices=["diag", "ot"], default="diag",
                    help="diag(默认)=对角化(产虚轨道, 供 NAC); ot=启用&OT/关&DIAGONALIZATION&SMEAR&ADDED_MOS(只收敛密度, 无虚轨道)")
    pv.add_argument("--mo-cubes", action="store_true",
                    help="往 &DFT/&PRINT 注入 &MO_CUBES(打 HOMO/LUMO 的 cube, 供 cube-overlap 算电子-空穴重叠); ot 段自动跳过")
    pv.add_argument("--nhomo", type=int, default=5, help="[配--mo-cubes] 打几条 HOMO 侧轨道(HOMO, HOMO-1, ...)")
    pv.add_argument("--nlumo", type=int, default=5, help="[配--mo-cubes] 打几条 LUMO 侧轨道(LUMO, LUMO+1, ...)")
    pv.add_argument("--cube-stride", type=int, nargs=3, default=[2, 2, 2], metavar=("SX", "SY", "SZ"),
                    help="[配--mo-cubes] cube 网格抽稀(CP2K 默认 2 2 2; 单个 cube 实测≈34MB@stride2, 1 1 1 会×8)")
    pv.add_argument("--ao-ndigits", type=int, default=8,
                    help="&AO_MATRICES 的 NDIGITS(默认8)。模板常写16 → out 大36%%、解析慢36%%; 实测8位对NAC的影响比最小耦合小3个量级。0=不改")
    pv.add_argument("--no-ao-overlap", action="store_true",
                    help="这一帧【不打印】AO overlap(OVERLAP F)。用于不做单帧S分析的帧: out 从几百MB降到几MB")
    pv.add_argument("--mo-cubes-in-ot", action="store_true",
                    help="OT 段也注入 &MO_CUBES(CP2K 会用 ot_eigensolver 事后算虚轨道; ot+diag 流程不需要, 纯OT流程才用)")

    pp = sub.add_parser("popanal", help="从 out 读 Mulliken/Hirshfeld 每原子净电荷(MEDIUM 已有)+总能量, 按元素汇总")
    pp.add_argument("--out", required=True, help="CP2K out(PRINT_LEVEL MEDIUM)")
    pp.add_argument("--outdir", default=None, help="[可选] 存 mulliken_charges.dat 的目录")

    pch = sub.add_parser("character", help="每个 MO 的 Löwdin 轨道成分(元素/spd)+局域; 报 VBM/CBM 能带边成分(相变电子指纹)")
    pch.add_argument("--wfn", required=True, help="该帧 .wfn")
    pch.add_argument("--overlap", required=True, help="该帧 CP2K &AO_MATRICES OVERLAP 文本(需标签, 不能用 .npy)")
    pch.add_argument("--bmin", type=int, default=None, help="[可选] 额外打印的活性窗起始 MO(0基)")
    pch.add_argument("--bmax", type=int, default=None, help="[可选] 活性窗结束 MO(0基,含)")
    pch.add_argument("--ispin", type=int, default=1)

    pcu = sub.add_parser("cube-overlap", help="读一帧的 *WFN_*.cube(按自旋分组), 算轨道间实空间重叠矩阵(电子-空穴 overlap)")
    pcu.add_argument("--cube-dir", required=True, help="放 cube 的目录(= 该帧的 scf_run/<tag>/)")
    pcu.add_argument("--glob", default="*WFN_*.cube", help="cube 文件通配(默认 *WFN_*.cube, 匹配 PROJECT-WFN_00636_1-1_0.cube)")
    pcu.add_argument("--wfn", default=None, help="[强烈建议] 该帧 .wfn: 用占据数定真 HOMO/LUMO 并贴标签(SMEAR 下窗口会偏移)")
    pcu.add_argument("--tag", default=None, help="帧标签(默认取 cube-dir 的目录名), 用于输出文件名")
    pcu.add_argument("--outdir", default="Cube_Overlap", help="输出根目录(默认 Cube_Overlap)")
    pcu.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                     help="cube 数据在内存里的精度(float32 省一半内存; 求和一律用 float64)")
    pcu.add_argument("--cross-spin", action="store_true",
                     help="额外算一个不分自旋的全矩阵(alpha/beta 混在一起; 密度共存跨自旋也有意义, 但跨自旋 NAC 恒为0)")

    ptd = sub.add_parser("cube-tdm",
                         help="从 cube 算跃迁偶极矩/振子强度/辐射寿命(速度规范, 周期体系正确)。"
                              "NAC 管非辐射复合, 这个管辐射复合。")
    ptd.add_argument("--cube-dir", required=True, help="放 cube 的目录(= 该帧的 scf_run/<tag>/)")
    ptd.add_argument("--glob", default="*WFN_*.cube", help="cube 文件通配")
    ptd.add_argument("--wfn", required=True,
                     help="【必需】该帧 .wfn: 速度规范 μ=<i|∇|j>/(E_j−E_i) 要本征值, 没它算不了")
    ptd.add_argument("--tag", default=None, help="帧标签(默认取 cube-dir 的目录名)")
    ptd.add_argument("--outdir", default="Cube_TDM", help="输出根目录(默认 Cube_TDM)")
    ptd.add_argument("--degen-eps", type=float, default=1e-4,
                     help="近简并阈值[Hartree, 默认1e-4≈2.7meV]: |ΔE| 小于它时 μ=<i|∇|j>/ΔE 真的发散 → 置 NaN")

    a = ap.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "nac":
        cmd_nac(a)
    elif a.cmd == "build-merged":
        cmd_build_merged(a)
    elif a.cmd == "hvib":
        cmd_hvib(a)
    elif a.cmd == "overlap":
        cmd_overlap(a)
    elif a.cmd == "ipr":
        cmd_ipr(a)
    elif a.cmd == "check":
        cmd_check(a)
    elif a.cmd == "ot-diag":
        cmd_ot_diag(a)
    elif a.cmd == "popanal":
        cmd_popanal(a)
    elif a.cmd == "vasp2cp2k":
        cmd_vasp2cp2k(a)
    elif a.cmd == "combine":
        cmd_combine(a)
    elif a.cmd == "character":
        cmd_character(a)
    elif a.cmd == "cube-overlap":
        cmd_cube_overlap(a)
    elif a.cmd == "cube-tdm":
        cmd_cube_tdm(a)
    else:
        cmd_wfn_nac(a)
    if a.cmd not in ("build-merged", "hvib", "overlap", "ipr", "check", "ot-diag", "popanal", "character", "vasp2cp2k", "combine", "wfn-nac", "cube-overlap", "cube-tdm"):
        print(CP2K_NOTE)

if __name__ == "__main__":
    main()
