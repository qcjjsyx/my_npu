#!/usr/bin/env python3
"""Compile TXT testbenches and verify generated files, without a generator dependency."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PACK_SOURCES=['hardware/rtl/systemverilog/mac_types_pkg.sv','hardware/rtl/systemverilog/mac_fp19_convert.sv','hardware/rtl/systemverilog/mac_fp9_convert.sv','hardware/rtl/systemverilog/mac_operand_pack.sv','hardware/verification/tb/tb_pack_txt.sv']
MAC_SOURCES=['hardware/rtl/systemverilog/mac_types_pkg.sv','hardware/verification/tb/dwbb_sim_models.sv','hardware/rtl/systemverilog/mac_mul_frontend.sv',
 'hardware/rtl/systemverilog/mac_align_max16.sv','hardware/rtl/systemverilog/mac_align_rne.sv','hardware/rtl/systemverilog/mac_align_lane0.sv','hardware/rtl/systemverilog/mac_align_lane1.sv',
 'hardware/rtl/systemverilog/mac_align_stage.sv','hardware/rtl/systemverilog/mac_psum_stage.sv','hardware/rtl/systemverilog/mac_mx_stage.sv','hardware/rtl/systemverilog/mac_acc_float_prepare36.sv',
 'hardware/rtl/systemverilog/mac_acc_stage.sv','hardware/verification/tb/tb_mac_packed_txt.sv']
NEGATIVES={'missing_first':'ACC state requires acc_first','scale255':'MX active group requires finite E8M0 scales',
 'mx_exponent':'MX input exponent boundary deferred','phase1':'INT16 phase1 arrived without phase0'}


def compile_tb(top,sources,image):
    p=subprocess.run(['iverilog','-g2012','-s',top,'-o',str(image),*sources],cwd=ROOT,capture_output=True,text=True,timeout=90)
    image.with_suffix('.compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)


def simulate(image,args,expect_failure=None):
    p=subprocess.run(['vvp',str(image),*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
    output=p.stdout+p.stderr
    if expect_failure:
        if p.returncode==0 or expect_failure not in output:raise RuntimeError(output)
        return 'Expected assertion PASS: '+expect_failure
    if p.returncode or 'PASS:' not in output:raise RuntimeError(output)
    return output.strip()


def run(vectors,build):
    vectors=Path(vectors).resolve();build=Path(build);build.mkdir(parents=True,exist_ok=True)
    pack=build/'pack.vvp';mac=build/'mac.vvp'
    compile_tb('tb_pack_txt',PACK_SOURCES,pack)
    compile_tb('tb_mac_packed_txt',MAC_SOURCES,mac)
    lines=[simulate(pack,[f'+VEC={vectors/"pack_vectors.txt"}'])]
    common=[f'+INPUTS={vectors/"mac_inputs.txt"}',f'+EXPECTED={vectors/"mac_expected.txt"}',f'+INTERMEDIATE={vectors/"mac_intermediate.txt"}']
    for schedule in (0,1):lines.append(simulate(mac,[*common,f'+SCHEDULE={schedule}']))
    for name,message in NEGATIVES.items():
        lines.append(simulate(mac,[f'+INPUTS={vectors/f"mac_invalid_{name}.txt"}','+NEGATIVE=1'],message))
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors',type=Path,default=Path(__file__).resolve().parents[1]/'vectors')
    p.add_argument('--build',type=Path)
    a=p.parse_args()
    if a.build:print(run(a.vectors,a.build),end='')
    else:
        with tempfile.TemporaryDirectory(prefix='mac-txt-') as build:print(run(a.vectors,build),end='')

if __name__=='__main__':main()
