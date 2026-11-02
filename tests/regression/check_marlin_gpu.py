"""Allocated-GPU1-only native Marlin micro, actual official Qwen AWQ weights."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

from tests.regression.check_awq_gpu import LIMITS, check_error, timing

UUID = "GPU-01bfd1ad-d71e-726e-4265-b3869b42b5d2"
MODEL_MANIFEST_SHA = "001a5bb5834740435fa7791dae7aca704e93f24784d339486fd85efd180a50a1"


def run_projection(torch, model, index, names, library):
    from safetensors import safe_open
    from nanovllm.layers.awq_packed import reference_weight_cpu
    from nanovllm.layers.awq_triton_kernel import awq_packed_gemm
    from nanovllm.layers.awq_marlin_sidecar import MarlinAWQState, derive_fp16_scales_cpu
    payload = {}
    for field in ("qweight", "qzeros", "scales"):
        values = []
        for prefix in names:
            key = "model.layers.0." + prefix + "." + field
            with safe_open(model / index[key], framework="pt", device="cpu") as file:
                values.append(file.get_tensor(key))
        payload[field] = torch.cat(values, dim=1).contiguous()
    k, n = payload["qweight"].shape[0], payload["qweight"].shape[1] * 8
    fp16_scales, scale_report = derive_fp16_scales_cpu(payload["scales"])
    dense = reference_weight_cpu(payload["qweight"], payload["qzeros"], fp16_scales).cuda().T.contiguous()
    dense_bf = dense.bfloat16()
    w, z, s = (payload[field].cuda() for field in ("qweight", "qzeros", "scales"))
    sf = fp16_scales.cuda()
    original_bytes = sum(t.numel() * t.element_size() for t in (w,z,s))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    prepare_base = torch.cuda.memory_allocated()
    state = MarlinAWQState.prepare(w,z,s,library=library,allow_lossless_fp16_scale_cache=True)
    torch.cuda.synchronize()
    prepare_extra = torch.cuda.max_memory_allocated() - prepare_base
    memory = dict(canonical_awq_bytes_retained=original_bytes, marlin_persistent_bytes=state.stats()["persistent_bytes"],
                  prepare_peak_extra_bytes=prepare_extra,
                  dense_references="test-only separate allocations excluded from these incremental counts")
    results = []
    gen = torch.Generator(device="cpu").manual_seed(20261023)
    for m in (1,2,4,8,32):
        x = torch.randn((m,k), generator=gen, dtype=torch.float32).half().cuda()
        xb = x.bfloat16()
        expected = torch.mm(x.float(), dense.T.float())
        eager = state(x)
        first = check_error(torch, eager, expected)
        stream = torch.cuda.Stream()
        graph_state = state.for_graph(stream)
        if graph_state.workspace.data_ptr() == state.workspace.data_ptr():
            raise AssertionError("capture and eager workspaces alias")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph_state(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = graph_state(x)
        output_address = graph_output.data_ptr()
        changed = []
        for i in range(10):
            fresh = torch.zeros((m,k),dtype=torch.float16) if i == 0 else torch.randn((m,k),generator=gen).half()
            x.copy_(fresh)
            graph.replay()
            direct = state(x)
            ref = torch.mm(x.float(), dense.T.float())
            row = check_error(torch,graph_output,ref)
            row["graph_eager_exact"] = torch.equal(graph_output,direct)
            row["fixed_output_address"] = graph_output.data_ptr() == output_address
            row["zero_exact"] = bool((graph_output == 0).all()) if i == 0 else True
            changed.append(row)
            del direct,ref
        # Independent second capture uses a different mutable workspace. Both
        # replay sequentially on default stream and must preserve the first.
        second_state = state.for_graph(stream)
        if second_state.workspace.data_ptr() in (state.workspace.data_ptr(),graph_state.workspace.data_ptr()):
            raise AssertionError("second graph workspace aliases")
        with torch.cuda.stream(stream):
            second_state(x)
        torch.cuda.current_stream().wait_stream(stream)
        second_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second_graph,stream=stream):
            second_output = second_state(x)
        graph.replay()
        second_graph.replay()
        second_exact = torch.equal(graph_output,second_output)
        torch.cuda.synchronize()
        before_a,before_r = torch.cuda.memory_allocated(),torch.cuda.memory_reserved()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        growth = (torch.cuda.memory_allocated()-before_a,torch.cuda.memory_reserved()-before_r)
        x.copy_(torch.randn((m,k),generator=gen).half())
        xb.copy_(x.bfloat16())
        methods = dict(marlin_eager=lambda: state(x), marlin_graph=graph.replay,
            triton_eager=lambda: awq_packed_gemm(x,w,z,sf),
            dense_fp16=lambda: torch.nn.functional.linear(x,dense),
            dense_bf16=lambda: torch.nn.functional.linear(xb,dense_bf))
        keep = []
        for name in ("triton_eager","dense_fp16","dense_bf16"):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    methods[name]()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                out = methods[name]()
            keep.append((g,out))
            methods[name.replace("_eager","")+"_graph"] = g.replay
        for fn in methods.values():
            for _ in range(10):
                fn()
        trials = {name:[] for name in methods}
        order = list(methods)
        for repeat in range(5):
            for name in order[repeat:]+order[:repeat]:
                trials[name].append(timing(torch,methods[name],30))
        medians = {name:statistics.median(r["cuda_ms"] for r in runs) for name,runs in trials.items()}
        passed = first["passed"] and second_exact and growth == (0,0) and all(
            c["passed"] and c["graph_eager_exact"] and c["fixed_output_address"] and c["zero_exact"] for c in changed)
        result = dict(projections=names,shape=[m,k,n],passed=passed,first_accuracy=first,changed_inputs=changed,
            second_graph_exact=second_exact,replay_memory_growth=growth,scale_cache=scale_report,
            memory=memory,median_cuda_ms=medians,trials=trials,
            fp16_eager_over_marlin=medians["dense_fp16"]/medians["marlin_eager"],
            fp16_graph_over_marlin=medians["dense_fp16_graph"]/medians["marlin_graph"])
        results.append(result)
        print(json.dumps(dict(shape=result["shape"],passed=passed,median_cuda_ms=medians)),flush=True)
        del graph,graph_state,second_state,second_graph,keep,methods,graph_output,second_output
        gc.collect()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",type=Path,default=Path("/home/hrxu/huggingface/Qwen3-8B-AWQ"))
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--authorize-gpu-window",action="store_true")
    args=parser.parse_args()
    if not args.authorize_gpu_window or os.environ.get("CUDA_VISIBLE_DEVICES") != UUID or os.environ.get("NANOVLLM_BENCH_GPU_INDEX") != "1":
        raise ValueError("explicit physical GPU1 lease required")
    if hashlib.sha256((args.model/"download_manifest.json").read_bytes()).hexdigest()!=MODEL_MANIFEST_SHA:
        raise ValueError("unexpected official checkpoint manifest")
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__),root/"nanovllm/layers/awq_marlin_sidecar.py",root/"tests/regression/check_awq_gpu.py"]
    plan=dict(thresholds=LIMITS,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              real_checkpoint_manifest_sha256=MODEL_MANIFEST_SHA,m_values=[1,2,4,8,32],repeats=5,iterations=30)
    (args.output/"plan.json").write_text(json.dumps(plan,indent=2))
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    ensure_gpu1_available(evidence_path=args.output/"guard.jsonl")
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    index=json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    projections=[("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"),("mlp.gate_proj","mlp.up_proj"),("mlp.down_proj",)]
    records=[]
    with torch.inference_mode():
        for names in projections:
            ensure_gpu1_available(evidence_path=args.output/"guard.jsonl")
            rows=run_projection(torch,args.model,index,names,args.library)
            records.extend(rows)
            with (args.output/"cases.jsonl").open("a") as out:
                for row in rows:
                    out.write(json.dumps(row)+"\n")
            gc.collect()
            torch.cuda.empty_cache()
    report=dict(passed=all(r["passed"] for r in records),cases=records,thresholds=LIMITS,
        torch=torch.__version__,device=torch.cuda.get_device_name(),gpu_uuid=UUID,
        scope="real weights synthetic inputs; not model NLL or end-to-end performance")
    (args.output/"result.json").write_text(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit("fixed threshold failure; do not relax")


if __name__=="__main__":
    main()
