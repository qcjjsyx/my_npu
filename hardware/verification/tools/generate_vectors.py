#!/usr/bin/env python3
"""Generate portable fixed-width hexadecimal TXT vectors (no RTL execution).

Run from any directory: python hardware/verification/tools/generate_vectors.py --out DIR
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
from reference.mac_contract import SourceFormat as F, ComputeMode as M, AccKind as K, _decode_finite_float
from reference.packed_mac import PackedStep, PackedAccumulator

DEFAULT_SEED = 0x2342026
INPUT_WIDTHS = [32,32,3,1,2,2,1,8,1,8,8,8,8,4,4]
EXPECTED_WIDTHS = [32,32,1,2,1,4,4,2] + [1,32,36,8,2]*2
INTER_WIDTHS = [32,32,9,6,2,2,56,36,36,8,8]
LANE_WIDTHS = [24,18,9,6,2,32,24]


def hexline(values, widths):
    if len(values) != len(widths):
        raise ValueError('field count mismatch')
    return ' '.join(f'{value & ((1<<width)-1):0{(width+3)//4}x}' for value,width in zip(values,widths))+'\n'


def pack_expected(raw, fmt, groups):
    """Standalone combinational pack behavior, including disabled groups."""
    if fmt in (F.FP16,F.BF16):
        if not groups & 1: return 0
        exp, sig = _decode_finite_float(raw, fmt)
        return (exp<<16) | (sig & 0xfff)
    if fmt in (F.E4M3,F.E5M2):
        result=0
        for g in range(2):
            if groups & (1<<g):
                exp,sig=_decode_finite_float((raw>>(8*g))&255,fmt)
                result |= (exp<<(16+5*g)) | ((sig&31)<<(5*g))
        return result
    width={F.INT16:16,F.INT8:8,F.INT4:4}[fmt]
    return sum((raw & (((1<<width)-1)<<(width*g))) for g in range(1 if width==16 else 2) if groups & (1<<g))


def pack_cases(seed, exhaustive):
    rng=random.Random(seed)
    for fmt in F:
        raws=set((0,1,0x7fff,0x8000,0xffff,0x7c00,0xfc00,0x7e00,0x7f80,0xff80,0x7fc0,0x0400,0x3ff,0x80,0x7f,0x88))
        raws.update(rng.randrange(65536 if fmt!=F.INT4 else 256) for _ in range(512))
        if fmt in (F.E4M3,F.E5M2):
            raws.update(x | (x<<8) for x in range(256))
            raws.update(x | ((255-x)<<8) for x in range(256))
        # Main file uses legal raw containers. Invalid groups, including zero,
        # deliberately exercise local combinational masking (not entry legality).
        if fmt==F.INT4: raws={x&255 for x in raws}
        if fmt==F.INT16: raws.discard(0x8000)
        for raw in sorted(raws):
            for gv in range(4):
                if fmt in (F.INT8,F.INT4):
                    width=8 if fmt==F.INT8 else 4
                    if any(gv&(1<<g) and ((raw>>(width*g))&((1<<width)-1))==1<<(width-1) for g in range(2)): continue
                yield int(fmt),gv,raw,pack_expected(raw,fmt,gv)
        if exhaustive and fmt in (F.FP16,F.BF16,F.INT16):
            for raw in range(65536):
                if fmt==F.INT16 and raw==0x8000: continue
                yield int(fmt),1,raw,pack_expected(raw,fmt,1)
        if exhaustive and fmt in (F.E4M3,F.E5M2):
            for raw in range(65536):
                yield int(fmt),3,raw,pack_expected(raw,fmt,3)


def f19(sig,exp): return (sig&0xfff) | (exp<<16)
def f9(s0,e0,s1,e1): return (s0&31) | ((s1&31)<<5) | (e0<<16) | (e1<<21)
def ints(a,b,width): return (a&((1<<width)-1)) | ((b&((1<<width)-1))<<width)


def mac_sequences(seed, random_count, stress):
    """Generate operands directly. No raw converter/enter() in the MAC path."""
    rng=random.Random(seed)
    seq=0
    def steps(mode,bias,scaled,rows):
        nonlocal seq
        seen=0
        index=0
        for a,b,gv,sa,sb,force in rows:
            first=(gv & ~seen) | (gv if force else 0)
            seen |= gv
            tag=(index&15,(index*7+3)&15)
            base=PackedStep(seq,index,mode,K(int(mode in (M.FP19,M.FP9) or scaled)),gv,first,0,bias,scaled,sa,sb,tag,tuple(a),tuple(b))
            yield base
            index+=1
            if mode==M.INT16:
                yield replace(base,step=index,phase=1,acc_first=0)
                index+=1
        seq+=1
    def row(a,b,gv=3,sa=(127,127),sb=(127,127),first=False):
        return ([a]*16 if isinstance(a,int) else a,[b]*16 if isinstance(b,int) else b,gv,sa,sb,first)
    # Synthetic packed FP9 Emax32 / bias14 -> exact E=150; independent group exponents/scales; cancellation/zero.
    yield from steps(M.FP9,14,True,[
        row(f9(8,16,8,10),f9(8,16,8,10)),
        row(f9(-8,16,8,12),f9(8,16,8,12),sa=(127,129),sb=(127,125)),
        row(0,0), row(f9(1,1,14,15),f9(1,1,14,15),gv=2),
    ])
    # FP19 positive/negative ALIGN halfway: odd P shifted9, d=10.
    for sign in (1,-1):
        yield from steps(M.FP19,30,False,[row(
            [f19(1024,20), f19(sign*1025,15)]+[0]*14,
            [f19(1024,20), f19(1025,15)]+[0]*14,gv=1)])
    # Exact and near cancellation through FP19 packed operands.
    yield from steps(M.FP19,30,False,[
        row([f19(1025,15)]+[0]*15,[f19(1024,15)]+[0]*15,gv=1),
        row([f19(-1024,15)]+[0]*15,[f19(1024,15)]+[0]*15,gv=1),
        row([f19(1024,15)]+[0]*15,[f19(1024,15)]+[0]*15,gv=1,first=True),
        row([f19(-1024,15)]+[0]*15,[f19(1024,15)]+[0]*15,gv=1),
    ])
    # Legal MX E=1 produces FTZ; E=254 with large coefficient overflows ACC.
    yield from steps(M.INT8,0,True,[
        row(ints(1,-1,8),ints(1,1,8),sa=(61,61),sb=(62,62)),
        row(ints(127,-127,8),ints(127,127,8),sa=(188,188),sb=(188,188)),
        row(ints(127,-127,8),ints(127,127,8),sa=(188,188),sb=(188,188)),
        row(ints(-127,127,8),ints(127,127,8)),
        row(0,0,first=True),
    ])
    # Saturation at phase1, sign/low-byte decomposition and phase0 first.
    yield from steps(M.INT16,0,False,[row(a&65535,b&65535,gv=1,first=True) for a,b in
        [(32767,32767),(-32767,32767),(-1,257),(255,-257),(0,32767)]])
    configs=[(M.FP19,30,False),(M.FP19,254,False),(M.FP9,14,False),
             (M.FP9,30,False),(M.FP9,22,True),(M.INT8,0,True),
             (M.INT8,0,False),(M.INT4,0,False),(M.INT16,0,False)]
    for mode,bias,scaled in configs:
        rows=[]
        for i in range(random_count):
            gv=1 if mode in (M.FP19,M.INT16) else rng.choice((1,2,3))
            def operand():
                if mode==M.FP19:
                    sign=rng.choice((-1,1))
                    if bias==254:
                        return f19(sign*rng.randrange(128,256)*8,rng.randrange(120,135))
                    return f19(sign*rng.randrange(1024,2048),rng.randrange(1,31))
                if mode==M.FP9:
                    # bias22 uses E4M3 on A and E5M2 on B; alternate per array below.
                    e5=(bias==30) or (bias==22 and operand_is_b)
                    fields=[]
                    for _ in range(2):
                        exponent=rng.randrange(1,31 if e5 else 16)
                        magnitude=rng.choice((8,10,12,14)) if e5 else rng.randrange(8,15 if exponent==15 else 16)
                        fields.extend((rng.choice((-1,1))*magnitude,exponent))
                    return f9(*fields)
                if mode==M.INT16:return rng.randrange(-32767,32768)&65535
                w=8 if mode==M.INT8 else 4
                limit=(1<<(w-1))-1
                return ints(rng.randrange(-limit,limit+1),rng.randrange(-limit,limit+1),w)
            operand_is_b=False
            a=[operand() for _ in range(16)]
            operand_is_b=True
            b=[operand() for _ in range(16)]
            if i%19==0:a=[0]*16
            rows.append(row(a,b,gv,tuple(rng.randrange(119,136) for _ in range(2)),tuple(rng.randrange(119,136) for _ in range(2)),i%17==0))
        yield from steps(mode,bias,scaled,rows)
    if stress:
        # Both signs saturate after 8,322+ dot16 updates; include margin.
        yield from steps(M.INT8,0,False,(row(ints(127,-127,8),ints(127,127,8)) for _ in range(8400)))


def exact(coefficient, exponent):
    """Finite coefficient/exponent value for coverage classification only."""
    return Fraction(coefficient) * (Fraction(2) ** (exponent - 161))


def input_header(s):
    return [s.sequence,s.step,int(s.mode),int(s.kind),s.group_valid,s.acc_first,s.phase,s.bias_sum,int(s.mx_enable),*s.scale_a,*s.scale_b,*s.tag]


def write_input(f,s):
    f.write(hexline(input_header(s),INPUT_WIDTHS))
    for a,b in zip(s.operand_a,s.operand_b): f.write(hexline([a,b],[26,26]))


def generate(out,seed=DEFAULT_SEED,random_count=96,exhaustive=False,stress=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pack_count=0
    with (out/'pack_vectors.txt').open('w') as f:
        for i,case in enumerate(pack_cases(seed,exhaustive)):
            f.write(hexline([i,*case],[32,3,2,16,26]));pack_count+=1
    acc=PackedAccumulator();count=0;commits=0;first_step=None;coverage={"align_inexact":0,"acc_inf_positive":0,"acc_inf_negative":0,"acc_ftz":0,"acc_cancellation":0,"int32_positive_sat":0,"int32_negative_sat":0}
    with (out/'mac_inputs.txt').open('w') as inp,(out/'mac_expected.txt').open('w') as exp,(out/'mac_intermediate.txt').open('w') as inter:
        for s in mac_sequences(seed,random_count,stress):
            old_float=list(acc.floating)
            expected,(m,a,p,x)=acc.evaluate(s)
            coverage["align_inexact"]+=int(any(a.inexact))
            for g in range(2):
                if not s.group_valid & (1<<g): continue
                if s.kind==K.TC_FLOAT:
                    state=acc.floating[g]
                    coverage["acc_inf_positive"]+=int(state.infinity_sign==1)
                    coverage["acc_inf_negative"]+=int(state.infinity_sign==-1)
                    if not state.infinity_sign and state.sig==0:
                        old=old_float[g]
                        if not old.infinity_sign or s.acc_first & (1<<g):
                            total=exact(x.contribution[g],x.exp[g])
                            if not s.acc_first & (1<<g):total+=exact(old.sig,old.exp)
                            coverage["acc_ftz"]+=int(total!=0)
                            coverage["acc_cancellation"]+=int(total==0 and old.sig!=0 and x.contribution[g]!=0 and not s.acc_first & (1<<g))
                elif expected[0]:
                    coverage["int32_positive_sat"]+=int(acc.integer[g]==2147483647)
                    coverage["int32_negative_sat"]+=int(acc.integer[g]==-2147483648)
            if first_step is None:first_step=s
            write_input(inp,s)
            exp.write(hexline([s.sequence,s.step,*expected],EXPECTED_WIDTHS))
            nz=int(a.has_nonzero[0]) | int(a.has_nonzero[1])<<1
            inex=int(a.inexact[0]) | int(a.inexact[1])<<1
            inter.write(hexline([s.sequence,s.step,*a.emax,nz,inex,p.packed56,*x.contribution,
                                *(x.exp[g] if x.contribution[g] else 0 for g in range(2))],INTER_WIDTHS))
            for j in range(16):
                inter.write(hexline([m.product0[j],m.product1[j],m.exp_sum0[j],m.exp_sum1[j],int(m.nz0[j]) | int(m.nz1[j])<<1,a.lane0[j],a.lane1[j]],LANE_WIDTHS))
            count+=1;commits+=expected[0]
    # Standalone negative runs: no numeric expected result is assigned.
    invalid={
        'missing_first':replace(first_step,acc_first=0),
        'scale255':replace(first_step,scale_a=(255,127)),
        'mx_exponent':replace(first_step,scale_a=(254,254),scale_b=(254,254)),
        'phase1':PackedStep(0,0,M.INT16,K.INT32,1,0,1,0,False,(0,0),(0,0),(0,0),(1,)*16,(1,)*16),
    }
    for name,step in invalid.items():
        with (out/f'mac_invalid_{name}.txt').open('w') as f:write_input(f,step)
    write_readme(out,seed,random_count,exhaustive,stress,pack_count,count,commits,coverage)
    return pack_count,count,commits


def write_readme(out,seed,random_count,exhaustive,stress,pack_count,count,commits,coverage):
    text=f'''TXT vector format v1 / TAG_W=4 / seed={seed:#x}
pack records={pack_count}; MAC phase steps={count}; expected commits={commits}
random transactions per configuration={random_count}; exhaustive_pack={exhaustive}; stress={stress}
Coverage counters (group updates, except align_inexact counts steps):
{coverage}
All numerical fields hexadecimal, fixed ceil(bit_width/4) digits, no 0x prefix.
Signed fields use width-limited two's complement. No headers/comments in data files.
MAC inputs are constructed directly as packed26 operands. No enter/raw conversion
is used to calculate MAC expected results. Unused groups may contain nonzero bits
on purpose. The scoreboard masks them like mul_frontend.
Directed Emax32/bias14 is a synthetic compute-interface vector (not an E4M3 raw encoding).
Random FP19/FP9 fields are source-format representable; inactive fields may be nonzero.
Only one reset before the first record. Sequence IDs are labels, NOT reset commands.
first starts each group's sequence; all inactive banks retain prior bits.
INT16 phases explicitly repeat operand arrays and tags; phase1 first=0.

pack_vectors.txt: one row, widths 32 3 2 16 26:
case_id src_fmt group_valid raw expected_payload
Group=00 and WIDE group=10/11 test local pack masking, not entry acceptance.
Integer reserved minima are excluded for active groups; INT4 raw high byte=0.
With exhaustive_pack: all 65536 raw FP16/BF16 codes, all legal INT16 codes,
and all 65536 raw byte pairs for each FP8 format. Other masks use directed/random.

mac_inputs.txt: header widths {INPUT_WIDTHS}
sequence step mode kind group_valid first phase bias_sum mx_enable
scale_a0 scale_a1 scale_b0 scale_b1 tag0 tag1
Then 16 rows: operand_a operand_b (26 26 bits; 7 hex digits each).
One step = one M input handshake. INT16 transaction = two steps.

mac_expected.txt: one row per input step, widths {EXPECTED_WIDTHS}
sequence step update_valid update_groups update_kind update_tag0 update_tag1 state_valid
kind0 int32_0 float_sig0 float_exp0 inf_sign0
kind1 int32_1 float_sig1 float_exp1 inf_sign1
All fields describe state AFTER accepting this step at ACC, including phase0.
update_kind/tag retain old values if no commit. inf_sign is signed2 (0,1,3).
Both integer and float banks are checked, including inactive/previous-mode banks.
No final FP32 conversion exists in the current DUT.

mac_intermediate.txt: header widths {INTER_WIDTHS}
sequence step emax0 emax1 group_nz inexact psum56 contribution0 contribution1 exp0 exp1
Then 16 rows, widths {LANE_WIDTHS}:
product0 product1 exp_sum0 exp_sum1 product_nz aligned0 aligned1
Zero MX coefficients have physical E=0. Nonzero E outside1..254 is rejected,
never wrapped or assigned an invented numerical output.

mac_invalid_*.txt: standalone negative input files (same format as mac_inputs).
No numerical expected output. Required existing RTL assertion substrings:
missing_first: ACC state requires acc_first
scale255: MX active group requires finite E8M0 scales
mx_exponent: MX input exponent boundary deferred
phase1: INT16 phase1 arrived without phase0
Run separately; expected fatal terminates simulation. See run_txt_vectors.py.

Reproduce:
python hardware/verification/tools/generate_vectors.py --out verification/vectors --seed {seed} --random-count {random_count}{' --exhaustive-pack' if exhaustive else ''}{' --stress' if stress else ''}
Check all files against RTL (requires iverilog/vvp):
python verification/run_txt_vectors.py --vectors verification/vectors
Runner also checks every intermediate stage, adds stalls/bubbles, and runs negatives.
Current MAC TXT TB capacity: 32768 phase steps per run.
Test-only M output register wraps 16 mul_frontend instances. No mac_entry_m_stage,
operand_pack or format converters are compiled into the MAC simulation.
DesignWare simulation stand-ins are used; no synthesis or PPA claims.

SHA256 of data files:
'''
    for p in sorted(out.glob('*.txt')):
        if p.name!='vectors_readme.txt':text+=f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
    (out/'vectors_readme.txt').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'vectors')
    parser.add_argument('--seed',type=lambda s:int(s,0),default=DEFAULT_SEED)
    parser.add_argument('--random-count',type=int,default=96)
    parser.add_argument('--exhaustive-pack',action='store_true')
    parser.add_argument('--stress',action='store_true')
    args=parser.parse_args()
    if args.random_count<1:parser.error('random-count must be positive')
    counts=generate(args.out,args.seed,args.random_count,args.exhaustive_pack,args.stress)
    print(f'Generated {counts[0]} pack records, {counts[1]} MAC steps, {counts[2]} commits in {args.out}')

if __name__=='__main__': main()
