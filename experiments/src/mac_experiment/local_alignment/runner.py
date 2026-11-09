"""Configuration-driven local-only experiment and inspectable numerical results."""
import csv
from collections import Counter
import gzip
import hashlib
import itertools
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from ..arithmetic.dyadic import Dyadic
from ..widthselect.distributions import WIDTH_DISTRIBUTIONS, distribution_metadata, sample_fp32_matrix
from .model import LAYOUTS, PAIRS, align, decode, family, output_fp32, postfold_product
from .quantize import quantize

VERSION = "local-alignment-v2"
DEFAULT_F = [0, 1, 2, 3, 4, 6, 8, 12, 16]
DEFAULT_LENGTHS = [32, 128, 512, 2048, 8192, 16384]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    defaults = dict(seeds=[1], sequences=1, lengths=DEFAULT_LENGTHS, formats=[mode for mode in PAIRS if mode != "bf16_postfold"],
                    F=DEFAULT_F, rounding=["floor", "rne"], distributions=list(WIDTH_DISTRIBUTIONS),
                    output_root="results/numerical/local_alignment/runs")
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError("configuration must be a mapping containing only " + str(list(defaults)))
    config = defaults | config
    for key in ("seeds", "lengths", "formats", "F", "rounding", "distributions"):
        if not isinstance(config[key], list) or not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be a nonempty unique list")
    for key in ("seeds", "lengths", "F"):
        if any(type(x) is not int for x in config[key]):
            raise ValueError(f"{key} must contain integers")
    if type(config["sequences"]) is not int or config["sequences"] < 1:
        raise ValueError("sequences must be a positive integer")
    if any(x < 32 or x > 16384 or x % 32 for x in config["lengths"]) or any(x < 0 for x in config["F"]):
        raise ValueError("lengths must be multiples of 32 in [32,16384]; F must be >=0")
    if set(config["formats"]) - PAIRS.keys() or set(config["rounding"]) - {"floor", "rne"}:
        raise ValueError("unsupported format/rounding")
    if set(config["distributions"]) - set(WIDTH_DISTRIBUTIONS):
        raise ValueError("unsupported distribution")
    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise ValueError("output_root must be a nonempty path")
    return config


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class DistributionCounts:
    """Exact empirical quantiles, storage proportional to distinct values."""
    def __init__(self):
        self.values = Counter()
        self.count = 0

    def add(self, value):
        self.count += 1
        self.values[value] += 1

    def quantiles(self, prefix):
        ranks = {"p50": max(1, (self.count + 1)//2), "p99": max(1, (99*self.count + 99)//100)}
        result = {prefix + "_" + name: None for name in ranks}
        cumulative = 0
        for value, count in sorted(self.values.items()):
            cumulative += count
            for name, rank in ranks.items():
                if result[prefix + "_" + name] is None and cumulative >= rank:
                    result[prefix + "_" + name] = value.to_float() if isinstance(value, Dyadic) else float(value)
        return result


class Errors:
    def __init__(self):
        self.count = self.exact = self.positive = self.negative = self.zero_l1 = 0
        self.signed_sum = self.absolute_sum = self.maximum = Dyadic.zero()
        self.absolute = DistributionCounts()
        self.normalized = DistributionCounts()
        self.normalized_sum = 0.0
        self.normalized_max = None

    def add(self, candidate, reference, l1):
        error = candidate - reference
        self.count += 1
        self.exact += not error.m
        self.positive += error.m > 0
        self.negative += error.m < 0
        self.signed_sum += error
        self.absolute_sum += abs(error)
        self.maximum = max(self.maximum, abs(error))
        self.absolute.add(abs(error))
        if l1.m:
            normalized = abs(error).to_fraction() / l1.to_fraction()
            self.normalized.add(normalized)
            self.normalized_sum += float(normalized)
            self.normalized_max = max(self.normalized_max or 0.0, float(normalized))
        else:
            self.zero_l1 += 1

    def summary(self):
        n = self.count
        return dict(count=n, exact_match=self.exact, exact_match_rate=self.exact / n if n else None,
                    positive=self.positive, negative=self.negative,
                    positive_ratio=self.positive / n if n else None, negative_ratio=self.negative / n if n else None,
                    signed_mean=self.signed_sum.to_float() / n if n else None,
                    absolute_mean=self.absolute_sum.to_float() / n if n else None,
                    absolute_max=self.maximum.to_float(), zero_l1=self.zero_l1,
                    error_over_l1_count=self.normalized.count,
                    error_over_l1_mean=self.normalized_sum / self.normalized.count if self.normalized.count else None,
                    error_over_l1_max=self.normalized_max,
                    **self.absolute.quantiles("absolute"), **self.normalized.quantiles("error_over_l1"))


def ordered(bits):
    # Both zeros share a coordinate; zero sign is compared separately.
    mag = bits & 0x7fffffff
    return -mag if bits >> 31 else mag


def blocks(a, b, mode, length):
    layout = family(mode)
    K = LAYOUTS[layout][3]
    af, bf = PAIRS[mode]
    for start in range(0, length, K):
        products = []
        for i in range(start, min(start + K, length)):
            ma, qa = decode(a.codes[i], af)
            mb, qb = decode(b.codes[i], bf)
            product = (ma * mb, qa + qb)
            products.append(postfold_product(*product) if mode == "bf16_postfold" else product)
        products.extend([(0, 0)] * (K - len(products)))
        h = a.scales[start // 32] + b.scales[start // 32] - 254 if layout == "fp8" else 0
        yield products, h


def evaluate(a, b, mode, length, F, rounding, local=None):
    local = local if local is not None else Errors()
    S = T = L1 = Dyadic.zero()
    changed = zeroed = nonzero = total = 0
    trace = None
    for index, (products, h) in enumerate(blocks(a, b, mode, length)):
        result = align(products, family(mode), F, rounding, h)
        local.add(result.q, result.s, result.l1)
        S, T, L1 = S + result.s, T + result.q, L1 + result.l1
        changed += result.changed
        zeroed += result.zeroed
        nonzero += sum(p != 0 for p, _ in products)
        total += len(products)
        if trace is None and (result.changed or index == (length - 1) // LAYOUTS[family(mode)][3]):
            trace = dict(block=index, h=h, **{k: v.to_dict() if isinstance(v, Dyadic) else v
                                            for k, v in asdict(result).items()})
            # asdict recursively converts Dyadic; preserve canonical integer-as-string serialization.
            trace.update(q=result.q.to_dict(), s=result.s.to_dict(), l1=result.l1.to_dict())
            C = LAYOUTS[family(mode)][0]
            trace.update(multiplication_product_width=18 if mode == "bf16_postfold" else C,
                         alignment_coefficient_width=C)
            if mode == "bf16_postfold":
                trace["recoding"] = "P'=P<<r, r=T%4, Q=T//4; terms P/T are P'/Q"
                start = index * LAYOUTS[family(mode)][3]
                for offset, term in enumerate(trace["terms"]):
                    i = start + offset
                    if i < length:
                        ma, ea = decode(a.codes[i], "bf16")
                        mb, eb = decode(b.codes[i], "bf16")
                        term.update(original_P=ma*mb, original_T=ea+eb, fold_r=(ea+eb)%4)
                    else:
                        term.update(original_P=0, original_T=0, fold_r=0)
    ref, ref_sat = output_fp32(S)
    out, out_sat = output_fp32(T)
    return dict(S=S, T=T, L1=L1, local=local, changed=changed, zeroed=zeroed,
                nonzero=nonzero, total=total, reference=ref, output=out,
                reference_saturated=ref_sat, output_saturated=out_sat, trace=trace)


def run_from_file(path, verbose=True):
    config = load_config(path)
    root = Path(config["output_root"])
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    buckets = {}
    quant_events = {}
    maximum = max(config["lengths"])
    with gzip.open(run / "sources.jsonl.gz", "wt") as sources, gzip.open(run / "samples.jsonl.gz", "wt") as samples:
        for dist, seed, seq in itertools.product(config["distributions"], config["seeds"], range(config["sequences"])):
            source_id = f"{dist}:{seed}:{seq}"
            abits = sample_fp32_matrix(dist, seed, "x", 1, maximum, seq)
            bbits = sample_fp32_matrix(dist, seed, "w", 1, maximum, seq)
            targets = {}
            for fmt in sorted({f for mode in config["formats"] for f in PAIRS[mode]}):
                targets[fmt] = (quantize(abits, fmt, fmt.startswith("e")), quantize(bbits, fmt, fmt.startswith("e")))
                for side, q in zip("AB", targets[fmt]):
                    key = f"{dist}:{fmt}:{side}"
                    bucket = quant_events.setdefault(key, {k: 0 for k in q.events})
                    for k, v in q.events.items():
                        bucket[k] += v
            sources.write(json.dumps(dict(id=source_id, A_fp32=abits, B_fp32=bbits,
                                           targets={f: [asdict(q) for q in qs] for f, qs in targets.items()})) + "\n")
            for mode, length, F, rounding in itertools.product(config["formats"], config["lengths"], config["F"], config["rounding"]):
                af, bf = PAIRS[mode]
                key = (dist, mode, length, F, rounding)
                if key not in buckets:
                    buckets[key] = dict(local=Errors(), dot=Errors(), fp_error=Errors(),
                                        counts=dict(changed=0, zeroed=0, nonzero=0, total=0, fp32_bitmatch=0,
                                                    fp32_unsaturated_bitmatch=0, fp32_unsaturated_count=0,
                                                    reference_saturated=0, output_saturated=0,
                                                    saturated_bitmatch=0, signed_zero_mismatch=0), ulps=[], representative=None)
                bucket = buckets[key]
                result = evaluate(targets[af][0], targets[bf][1], mode, length, F, rounding, bucket["local"])
                bucket["dot"].add(result["T"], result["S"], result["L1"])
                out, ref = result["output"], result["reference"]
                bucket["fp_error"].add(out.exact_dyadic(), ref.exact_dyadic(), result["L1"])
                sat = result["output_saturated"] or result["reference_saturated"]
                counts = bucket["counts"]
                for name in ("changed", "zeroed", "nonzero", "total", "reference_saturated", "output_saturated"):
                    counts[name] += result[name]
                counts["fp32_bitmatch"] += out.bits == ref.bits
                counts["fp32_unsaturated_count"] += not sat
                counts["fp32_unsaturated_bitmatch"] += not sat and out.bits == ref.bits
                counts["saturated_bitmatch"] += sat and out.bits == ref.bits
                counts["signed_zero_mismatch"] += out.is_zero() and ref.is_zero() and out.sign != ref.sign
                ulp = abs(ordered(out.bits) - ordered(ref.bits))
                bucket["ulps"].append(ulp)
                error = abs(result["T"] - result["S"])
                if bucket["representative"] is None or error > bucket["representative"][0]:
                    bucket["representative"] = (error, dict(source_id=source_id, trace=result["trace"]))
                samples.write(json.dumps(dict(source_id=source_id, distribution=dist, mode=mode, length=length, F=F,
                                              rounding=rounding, S=result["S"].to_dict(), T=result["T"].to_dict(),
                                              L1=result["L1"].to_dict(), output_bits=out.bits, reference_bits=ref.bits,
                                              reference_saturated=result["reference_saturated"],
                                              output_saturated=result["output_saturated"], ulp=ulp)) + "\n")
            if verbose:
                print(f"completed {source_id}")
    rows, representatives = [], []
    for key, bucket in buckets.items():
        dist, mode, length, F, rounding = key
        C, g, b, K = LAYOUTS[family(mode)]
        row = dict(distribution=dist, mode=mode, length=length, F=F, rounding=rounding,
                   product_width=18 if mode == "bf16_postfold" else C,
                   coefficient_width=C, window_width=C+F, sum_width=C+F+(K-1).bit_length(), K=K,
                   **bucket["counts"])
        for label in ("local", "dot", "fp_error"):
            row.update({f"{label}_{k}": v for k, v in bucket[label].summary().items()})
        ulps = sorted(bucket["ulps"])
        row.update(ulp_mean=sum(ulps)/len(ulps), ulp_max=ulps[-1], ulp_p99=ulps[(99*len(ulps)-1)//100])
        row.update(changed_rate=row["changed"] / row["nonzero"] if row["nonzero"] else None,
                   zeroed_rate=row["zeroed"] / row["nonzero"] if row["nonzero"] else None,
                   fp32_bitmatch_rate=row["fp32_bitmatch"] / row["dot_count"],
                   fp32_unsaturated_bitmatch_rate=row["fp32_unsaturated_bitmatch"] / row["fp32_unsaturated_count"] if row["fp32_unsaturated_count"] else None)
        rows.append(row)
        representatives.append(dict(distribution=dist, mode=mode, length=length, F=F, rounding=rounding,
                                    **bucket["representative"][1]))
    write_json(run / "summary.json", rows)
    with (run / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run / "representatives.json", representatives)
    write_json(run / "quantization.json", quant_events)
    def git(*args):
        p = subprocess.run(["git", *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    package = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob("*.py"))}
    write_json(run / "metadata.json", dict(version=VERSION, python=platform.python_version(), git_head=git("rev-parse", "HEAD"),
               git_status=git("status", "--short"), implementation_sha256=hashes,
               config_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               distributions={d: distribution_metadata(d) for d in config["distributions"]},
               quantiles="exact nearest-rank p50/p99 from Dyadic/Fraction frequency tables; memory proportional to distinct observations; numeric reporting uses binary64",
               postfold="bf16_postfold: signed18 P=mA*mB, T=EA+EB -> signed21 P'=P<<(T%4), Q=T//4; alignment g=4,b=268; original bf16 unchanged",
               contract="exact product -> local floor/RNE -> exact dyadic accumulation -> final FP32 RNE finite saturation; no pair/state rounding",
               quantization="FP32 shared sources; per-side 32-element smallest covering power-of-two E8M0 scale; target RNE, subnormal kept, overflow saturates",
               representative_policy="maximum absolute T-S, first on tie; source_id joins sources.jsonl.gz; replay with evaluate",
               quantization_event_scope="full maximum-length input, counted once per distribution/format/side across seeds and sequences",
               saturation_metrics="overall FP32 matches include saturation; unsaturated counts exclude either-side saturation; T-S always exact"))
    return run


def replay(run, index=0):
    """Replay one maximum-|T-S| representative from saved codes; no resampling."""
    from .quantize import Quantized
    run = Path(run)
    records = json.loads((run / "representatives.json").read_text())
    if not 0 <= index < len(records):
        raise ValueError("representative index out of range")
    record = records[index]
    source = None
    with gzip.open(run / "sources.jsonl.gz", "rt") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["id"] == record["source_id"]:
                source = candidate
                break
    if source is None:
        raise ValueError("representative source missing")
    af, bf = PAIRS[record["mode"]]
    a = Quantized(**source["targets"][af][0])
    b = Quantized(**source["targets"][bf][1])
    result = evaluate(a, b, record["mode"], record["length"], record["F"], record["rounding"])
    return dict(**{k: v for k, v in record.items() if k != "trace"},
                S=result["S"].to_dict(), T=result["T"].to_dict(), L1=result["L1"].to_dict(),
                reference_bits=result["reference"].bits, output_bits=result["output"].bits,
                reference_saturated=result["reference_saturated"], output_saturated=result["output_saturated"],
                trace=result["trace"])
