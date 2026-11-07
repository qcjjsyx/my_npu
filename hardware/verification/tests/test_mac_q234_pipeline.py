"""Real entry/M -> ALIGN -> PSUM -> MX -> ACC chain with stalls and reset."""
from pathlib import Path
import random
import subprocess
import pytest
from reference.mac_contract import ReferenceAccumulator, SourceFormat as F, Transaction

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    'hardware/rtl/systemverilog/mac_types_pkg.sv', 'hardware/verification/tb/dwbb_sim_models.sv',
    'hardware/rtl/systemverilog/mac_fp19_convert.sv', 'hardware/rtl/systemverilog/mac_fp9_convert.sv', 'hardware/rtl/systemverilog/mac_operand_pack.sv',
    'hardware/rtl/systemverilog/mac_mul_frontend.sv', 'hardware/rtl/systemverilog/mac_entry_m_stage.sv', 'hardware/rtl/systemverilog/mac_align_max16.sv',
    'hardware/rtl/systemverilog/mac_align_rne.sv', 'hardware/rtl/systemverilog/mac_align_lane0.sv', 'hardware/rtl/systemverilog/mac_align_lane1.sv',
    'hardware/rtl/systemverilog/mac_align_stage.sv', 'hardware/rtl/systemverilog/mac_psum_stage.sv', 'hardware/rtl/systemverilog/mac_mx_stage.sv',
    'hardware/rtl/systemverilog/mac_acc_float_prepare36.sv', 'hardware/rtl/systemverilog/mac_acc_stage.sv',
    'hardware/verification/tb/tb_mac_q234_pipeline.sv',
]


def _transactions(fmt_a, fmt_b, scaled):
    rng = random.Random(0x234 + int(fmt_a)*11 + int(fmt_b))
    seen = 0
    def raw(fmt):
        if fmt == F.FP16:
            return rng.choice((0,1,0x3C00,0xBC00,0x7BFF,0xFBFF,0x3555,0x0400))
        if fmt == F.BF16:
            return rng.choice((0,0x3F80,0xBF80,0x3F01,0x4001,0xBE80,0x4080))
        if fmt in (F.E4M3,F.E5M2):
            return rng.randrange(256) | (rng.randrange(256)<<8)
        if fmt == F.INT16:
            return rng.randrange(-32767,32768)&0xFFFF
        width=8 if fmt==F.INT8 else 4
        limit=(1<<(width-1))-1
        return (rng.randrange(-limit,limit+1)&((1<<width)-1)) | ((rng.randrange(-limit,limit+1)&((1<<width)-1))<<width)
    for index in range(80):
        gv=1 if fmt_a in (F.FP16,F.BF16,F.INT16) else rng.choice((1,2,3))
        first=gv & ~seen
        if index%17==0: first=gv
        seen |= gv
        yield Transaction(
            (0,)*16 if index%13==0 else tuple(raw(fmt_a) for _ in range(16)),
            tuple(raw(fmt_b) for _ in range(16)),fmt_a,fmt_b,gv,first,
            (index&15,(index*7)&15),scaled,
            (rng.randrange(119,136),rng.randrange(119,136)),
            (rng.randrange(119,136),rng.randrange(119,136)),
        )


@pytest.mark.parametrize('fmt_a,fmt_b,scaled', [
    (F.FP16,F.FP16,False),(F.BF16,F.BF16,False),
    (F.E4M3,F.E4M3,False),(F.E5M2,F.E5M2,False),
    (F.E4M3,F.E5M2,True),(F.E5M2,F.E4M3,True),
    (F.INT16,F.INT16,False),(F.INT8,F.INT8,False),
    (F.INT8,F.INT8,True),(F.INT4,F.INT4,False),
])
def test_q234_real_pipeline(tmp_path,fmt_a,fmt_b,scaled):
    txs=list(_transactions(fmt_a,fmt_b,scaled))
    acc=ReferenceAccumulator(tag_w=4)
    vec=tmp_path/'pipeline.txt'
    with vec.open('w') as f:
        f.write(f'{int(fmt_a)} {int(fmt_b)} {int(scaled)} {len(txs)}\n')
        for tx in txs:
            snapshot=acc.step(tx).snapshot
            fields=[tx.group_valid,tx.acc_first,*tx.tag,*tx.scale_a,*tx.scale_b,int(snapshot.kind)]
            for g in range(2):
                fields += [snapshot.data[g] or 0,snapshot.exp[g] or 0,(snapshot.infinity_sign[g] or 0)&3]
            f.write(' '.join(f'{x:x}' for x in fields)+'\n')
            for a,b in zip(tx.raw_a,tx.raw_b): f.write(f'{a:04x} {b:04x}\n')
    image=tmp_path/'pipeline.vvp'
    compiled=subprocess.run(['iverilog','-g2012','-s','tb_mac_q234_pipeline','-o',str(image),*SOURCES],cwd=ROOT,capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    simulated=subprocess.run(['vvp',str(image),f'+VEC={vec}'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert simulated.returncode==0,simulated.stdout+simulated.stderr
    assert f'mac_q234_pipeline PASS: {len(txs)} transactions' in simulated.stdout
