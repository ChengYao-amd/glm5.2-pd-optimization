"""Prepare immutable original-implementation outputs before any candidate edit."""
import json
from pathlib import Path

import torch

from fixture_io import configure_runtime, contract, load_layer, variant
from reference import compare
from driver import capture


def main():
    torch.set_grad_enabled(False)
    fingerprint = configure_runtime()
    from kernel import moe_forward
    cfg = contract()
    goldens, calibration = {}, []
    loose = {'rtol': 0.05, 'atol': 0.01, 'min_snr_db': 35}
    for case in cfg['scored_cases']:
        inputs, captured_output = load_layer(case['layer_id'])
        nonzero = int((inputs['hidden_states'] != 0).any(1).sum().item())
        if nonzero != 192:
            raise RuntimeError(f'Captured input has only {nonzero} nonzero rows; investigate padding before scoring')
        for mode in cfg['correctness_variants']:
            current = variant(inputs, mode)
            graph, output = capture(moe_forward, current)
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise RuntimeError('Original MoE produced nonfinite output')
            if mode == 'real':
                original = compare(output, captured_output, loose)
                if not original['ok']:
                    raise RuntimeError(f'Captured output cannot be reproduced: {case["id"]} {original}')
                calibration.append({'case': case['id'], 'comparison': 'serving-capture', **original})
            expected = output.clone()
            goldens[f"{case['id']}/{mode}"] = expected.cpu()
            for repeat in range(3):
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                result = compare(output, expected, loose)
                if not result['ok']:
                    raise RuntimeError(f'Original repeated replay is unstable: {result}')
                calibration.append({'case': case['id'], 'variant': mode, 'repeat': repeat, **result})
            print(case['id'], mode, result, flush=True)
            del graph, output, expected, current
        del inputs, captured_output
        torch.cuda.empty_cache()
    observed = min(x['snr_db'] for x in calibration)
    case_minimum = {}
    for item in calibration:
        key = item['case'] + '/' + item.get('variant', 'real')
        case_minimum[key] = min(case_minimum.get(key, 200), item['snr_db'])
    cfg['numerics'] = {'min_snr_db': min(60.0, max(35.0, observed - 3.0)),
                       'rtol': 0.05, 'atol': 0.01,
                       'calibrated': True,
                       'case_min_snr_db': {k: min(60.0, max(35.0, v - 3.0)) for k, v in case_minimum.items()},
                       'calibration_min_snr_db': observed,
                       'basis': 'Frozen original AITER outputs; captured-output reproduction and repeated graph replay'}
    fixture_dir = Path(cfg['fixture_dir'])
    torch.save(goldens, fixture_dir / 'goldens.pt')
    (Path(__file__).parent / 'cases.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (fixture_dir / 'calibration.json').write_text(json.dumps({'source_hash': fingerprint,
        'numerics': cfg['numerics'], 'comparisons': calibration}, indent=2) + '\n')
    print('Original MoE goldens and calibrated policy saved', cfg['numerics'], flush=True)


if __name__ == '__main__':
    main()
