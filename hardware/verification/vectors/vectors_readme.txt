TXT vector format v1 / TAG_W=4 / seed=0x2342026
pack records=345168; MAC phase steps=9385; expected commits=9284
random transactions per configuration=96; exhaustive_pack=True; stress=True
Coverage counters (group updates, except align_inexact counts steps):
{'align_inexact': 434, 'acc_inf_positive': 2, 'acc_inf_negative': 2, 'acc_ftz': 2, 'acc_cancellation': 2, 'int32_positive_sat': 101, 'int32_negative_sat': 87}
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

mac_inputs.txt: header widths [32, 32, 3, 1, 2, 2, 1, 8, 1, 8, 8, 8, 8, 4, 4]
sequence step mode kind group_valid first phase bias_sum mx_enable
scale_a0 scale_a1 scale_b0 scale_b1 tag0 tag1
Then 16 rows: operand_a operand_b (26 26 bits; 7 hex digits each).
One step = one M input handshake. INT16 transaction = two steps.

mac_expected.txt: one row per input step, widths [32, 32, 1, 2, 1, 4, 4, 2, 1, 32, 36, 8, 2, 1, 32, 36, 8, 2]
sequence step update_valid update_groups update_kind update_tag0 update_tag1 state_valid
kind0 int32_0 float_sig0 float_exp0 inf_sign0
kind1 int32_1 float_sig1 float_exp1 inf_sign1
All fields describe state AFTER accepting this step at ACC, including phase0.
update_kind/tag retain old values if no commit. inf_sign is signed2 (0,1,3).
Both integer and float banks are checked, including inactive/previous-mode banks.
No final FP32 conversion exists in the current DUT.

mac_intermediate.txt: header widths [32, 32, 9, 6, 2, 2, 56, 36, 36, 8, 8]
sequence step emax0 emax1 group_nz inexact psum56 contribution0 contribution1 exp0 exp1
Then 16 rows, widths [24, 18, 9, 6, 2, 32, 24]:
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
python verification/generate_vectors.py --out verification/vectors --seed 36970534 --random-count 96 --exhaustive-pack --stress
Check all files against RTL (requires iverilog/vvp):
python verification/run_txt_vectors.py --vectors verification/vectors
Runner also checks every intermediate stage, adds stalls/bubbles, and runs negatives.
Current MAC TXT TB capacity: 32768 phase steps per run.
Test-only M output register wraps 16 mul_frontend instances. No mac_entry_m_stage,
operand_pack or format converters are compiled into the MAC simulation.
DesignWare simulation stand-ins are used; no synthesis or PPA claims.

SHA256 of data files:
6832e2af86bc3f96eb972aac80bf85843379811c691a87b1e728d971b6c6a048  mac_expected.txt
bf010fde9ecf3a908e31a636128e6355359858a59c151601ff1f3f5c66130149  mac_inputs.txt
82d9cf7a754b6745e7b505b47df2a82fa95707b587366ec60865167903100a26  mac_intermediate.txt
63a1fc71bd22e058debb0c931dc5d9cb9a30d7d448766587596ff4bc64b13261  mac_invalid_missing_first.txt
22292e4517e4c15481856468b1ec660e7375ab18f562503505253ef001c3b6b3  mac_invalid_mx_exponent.txt
58d54c896e19bd1606e91bfcf6a1c5e48948c7f17aa3231e7e49b26ac6421c8e  mac_invalid_phase1.txt
6500694fe2746d95293e4428f89210f81712bc66298ef3e1a61a10b36c89d5e1  mac_invalid_scale255.txt
5b2d32f870e2dcf82022b1552a62151191cb6dffcb9138a4b00fb7e9f77eb976  pack_vectors.txt
