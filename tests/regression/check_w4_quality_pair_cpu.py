"""CPU report-contract tests; synthetic PASS data is not model-quality evidence."""
import argparse, copy, importlib.util, json, pathlib, unittest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('gate',a.root/'candidate/benchmarks/check_w4_quality_pair.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    base=json.loads((a.root/'quality/fp16_validation_001/report.json').read_text())
    quant=json.loads((a.root/'quality/marlin_validation_001/report.json').read_text())
    manifest=json.loads(pathlib.Path('/home/hrxu/nano-vllm-archive/home-cleanup-20261025-120138/experiments/nano-spec-quant-20261023-yefFcZZX/main/artifacts/w4_quality_frozen_003/manifest.json').read_text())
    class Tests(unittest.TestCase):
        def test_original_fail(self):self.assertEqual(gate.compare(base,quant,manifest)['status'],'FAIL_QUALITY')
        def test_synthetic_pass(self):
            fake=copy.deepcopy(quant);fake['rows']=copy.deepcopy(base['rows']);fake['mean_teacher_forced_nll']=base['mean_teacher_forced_nll']
            self.assertEqual(gate.compare(base,fake,manifest)['status'],'PASS_QUALITY')
        def test_rejections(self):
            mutations=[lambda q:q.update(status='PASS'),lambda q:q.update(split='calibration'),lambda q:q.update(dtype='bfloat16'),lambda q:q.update(quantization=None),lambda q:q.update(packed_modules={}),lambda q:q.update(source_sha256={}),lambda q:q.update(mean_teacher_forced_nll=0),lambda q:q.update(finite_positions=0),lambda q:q['rows'].pop(),lambda q:q['rows'][0]['tokens'].__setitem__(0,0),lambda q:q['rows'][0]['nll'].__setitem__(0,float('nan')),lambda q:q['rows'][0]['predicted_tokens'].pop()]
            for i,fn in enumerate(mutations):
                with self.subTest(case=i):
                    q=copy.deepcopy(quant);fn(q)
                    with self.assertRaises(ValueError):gate.compare(base,q,manifest)
        def test_threshold_tampering(self):
            m=copy.deepcopy(manifest);m['thresholds']['max_validation_mean_nll_increase_nats']=.2
            with self.assertRaises(ValueError):gate.compare(base,quant,m)
            m['sha256']=gate.digest({k:v for k,v in m.items() if k!='sha256'})
            with self.assertRaises(ValueError):gate.compare(base,quant,m)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()
