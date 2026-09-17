"""Frozen captured-workload MoE driver: correctness, graph benchmark, profile."""
import argparse
import json
import statistics

import torch

from fixture_io import configure_runtime, contract, load_layer, variant
from reference import compare, load_goldens


def capture(run, inputs, warmup=3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            run(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(inputs)
    return graph, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--bench-mode', action='store_true')
    modes.add_argument('--profile-run', action='store_true')
    p.add_argument('--profile-case')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--repeat', type=int, default=3)
    args = p.parse_args()
    if min(args.warmup, args.iters, args.repeat) < 1:
        p.error('warmup/iters/repeat must be positive')
    torch.set_grad_enabled(False)
    source_hash = configure_runtime()
    from kernel import moe_forward
    config = contract()
    cases = config['scored_cases']
    if args.profile_run:
        cid = args.profile_case or cases[0]['id']
        selected = next((c for c in cases if c['id'] == cid), None)
        if selected is None:
            p.error('Unknown profile case')
        inputs, _ = load_layer(selected['layer_id'], include_output=False)
        graph, output = capture(moe_forward, inputs, args.warmup)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        return 0
    goldens = load_goldens(config)
    policy = config['numerics']
    all_results, bench_results = [], []
    for case in cases:
        inputs, captured = load_layer(case['layer_id'], include_output=False)
        variants = ['real'] if args.bench_mode else config['correctness_variants']
        for mode in variants:
            current = variant(inputs, mode)
            case_policy = dict(policy, min_snr_db=policy['case_min_snr_db'][f"{case['id']}/{mode}"])
            expected = goldens[f"{case['id']}/{mode}"].cuda()
            before = {k: v.view(torch.uint8).clone() for k, v in current.items() if isinstance(v, torch.Tensor)}
            graph, output = capture(moe_forward, current, args.warmup)
            def check():
                result = compare(output, expected, case_policy)
                result['ok'] &= all(torch.equal(current[k].view(torch.uint8), value) for k, value in before.items())
                result['ok'] &= output.data_ptr() != current['hidden_states'].data_ptr()
                if mode == 'zeros':
                    result['ok'] &= bool((output == 0).all())
                return result
            for _ in range(3):
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                result = check()
                all_results.append(result)
                if not result['ok']:
                    print('FAILED_CASE', case['id'], mode, json.dumps(result))
                    print(f"SNR: {result['snr_db']:.6f} dB\nallclose: False")
                    return 1
            if args.bench_mode:
                samples = []
                for _ in range(args.repeat):
                    for _ in range(args.warmup):
                        graph.replay()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iters):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / args.iters)
                    if not check()['ok']:
                        raise RuntimeError('Benchmark replay changed output or inputs')
                value = statistics.median(samples)
                bench_results.append(value)
                print('# samples_ms', case['id'], json.dumps(samples))
                print(f"case_ms: {case['id']} {value:.9f}")
            del graph, output, current, before, expected
        del inputs, captured
        torch.cuda.empty_cache()
    if args.bench_mode:
        print(f'mean_ms: {statistics.mean(bench_results):.9f}')
    else:
        print(f"SNR: {min(r['snr_db'] for r in all_results):.6f} dB")
        print('allclose: True')
        print(f'# correctness replay checks: {len(all_results)}; source {source_hash}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
