"""CPU-only ownership/stream regression tests; not CUDA correctness evidence.

Run: CUDA_VISIBLE_DEVICES='' python -B tests/regression/check_quant_graph_cpu.py \
    --source-root /absolute/path/to/main
Only CUDA entry points are mocked; the helper and Marlin for_graph are real.
No extension is loaded, checkpoint opened, or GPU tensor allocated.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStream:
    def __init__(self, handle, device="cuda:0"):
        self.cuda_stream = handle
        self.device = torch.device(device)
        self.synchronize = Mock()
        self.wait_stream = Mock()


class QuantGraphCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("Run this CPU test in a fresh process without CUDA initialization")
        cls.helper = load_file("_review_quant_graph", "nanovllm/layers/quant_graph.py")
        packed = load_file("_review_awq_packed", "nanovllm/layers/awq_packed.py")
        # Avoid importing nanovllm.__init__ and the full engine dependency tree.
        with patch.dict(sys.modules, {"nanovllm.layers.awq_packed": packed}):
            cls.sidecar = load_file("_review_marlin", "nanovllm/layers/awq_marlin_sidecar.py")

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.streams, self.captures = [], []
        self.current = FakeStream(100)
        self.capturing = False
        self.capture_failure = None
        self.stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("unexpected CUDA initialization")))
        self.new_stream = self.stack.enter_context(patch.object(torch.cuda, "Stream", side_effect=self.make_stream))
        self.stack.enter_context(patch.object(torch.cuda, "current_stream", side_effect=lambda *a, **k: self.current))
        self.stack.enter_context(patch.object(torch.cuda, "stream", side_effect=self.stream_scope))
        self.stack.enter_context(patch.object(torch.cuda, "graph", side_effect=self.capture_scope))
        self.sync = self.stack.enter_context(patch.object(torch.cuda, "synchronize"))
        self.stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", side_effect=lambda: self.capturing))
        self.stack.enter_context(patch.object(torch.ops, "load_library", side_effect=AssertionError("unexpected native library load")))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def make_stream(self, device=None):
        stream = FakeStream(200 + len(self.streams), device or "cuda:0")
        self.streams.append(stream)
        return stream

    @contextmanager
    def stream_scope(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    @contextmanager
    def capture_scope(self, graph, pool=None, stream=None):
        if self.capture_failure == "enter":
            raise RuntimeError("injected capture enter")
        self.captures.append((graph, pool, stream))
        with self.stream_scope(stream or self.make_stream()):
            self.capturing = True
            try:
                yield
                if self.capture_failure == "exit":
                    raise RuntimeError("injected capture exit")
            finally:
                self.capturing = False

    def state(self):
        state = self.sidecar.MarlinAWQState()
        state.device = torch.device("cuda:0")  # Metadata only; tensors remain CPU.
        state.weight = torch.arange(32, dtype=torch.int32)
        state.zeros = torch.arange(8, dtype=torch.int32)
        state.scales = torch.ones(8, dtype=torch.float16)
        state.workspace = torch.zeros(16, dtype=torch.int32)
        state.empty = torch.empty(0, dtype=torch.int32)
        state.execution_stream = 100
        state.python_dispatches = 7
        state.workspace_scope = "eager_single_stream"
        return state

    def model(self, count=2):
        layers = []
        for _ in range(count):
            layer = torch.nn.Module()
            layer.backend = "awq_marlin"
            layer._marlin_state = layer._active_marlin = self.state()
            layer._marlin_graph_states = []
            layers.append(layer)
        return torch.nn.Sequential(*layers)

    def assert_restored(self, model, count):
        for layer in model:
            self.assertIs(layer._active_marlin, layer._marlin_state)
            self.assertEqual(len(layer._marlin_graph_states), count)
            names = [n for n in layer._buffers if n.startswith("_marlin_graph_workspace_")]
            self.assertEqual(len(names), count)
            self.assertEqual(set(names), layer._non_persistent_buffers_set)
            for index, state in enumerate(layer._marlin_graph_states, 1):
                self.assertIs(layer._buffers[f"_marlin_graph_workspace_{index}"], state.workspace)

    def test_shared_pool_uses_one_capture_stream_per_model(self):
        model, pool = self.model(), object()
        for _ in range(3):
            with self.helper.capture_model_graph(model, object(), pool=pool):
                pass
        self.assertEqual(self.new_stream.call_count, 1)
        self.assertTrue(all(p is pool and s is model._marlin_capture_stream for _, p, s in self.captures))
        self.assert_restored(model, 3)

    def test_different_models_have_different_capture_streams(self):
        a, b = self.model(), self.model()
        with self.helper.model_graph_workspace(a):
            pass
        with self.helper.model_graph_workspace(b):
            pass
        self.assertIsNot(a._marlin_capture_stream, b._marlin_capture_stream)

    def test_explicit_stream_is_honored_without_rebinding_default(self):
        model = self.model()
        with self.helper.model_graph_workspace(model) as default:
            pass
        explicit = FakeStream(987)
        with self.helper.model_graph_workspace(model, explicit) as chosen:
            self.assertIs(chosen, explicit)
            self.assertTrue(all(l._active_marlin.execution_stream == 987 for l in model))
        self.assertIs(model._marlin_capture_stream, default)
        self.assert_restored(model, 2)

    def test_unique_workspaces_share_immutable_data_and_keep_eager_state(self):
        model = self.model()
        snapshots = [{n: getattr(l._marlin_state, n).clone() for n in ("weight", "zeros", "scales")} for l in model]
        for _ in range(2):
            with self.helper.model_graph_workspace(model):
                for layer in model:
                    self.assertIsNot(layer._active_marlin, layer._marlin_state)
        for layer, snapshot in zip(model, snapshots):
            eager = layer._marlin_state
            states = [eager, *layer._marlin_graph_states]
            self.assertEqual(len({s.workspace.data_ptr() for s in states}), 3)
            for graph_state in states[1:]:
                for name in ("weight", "zeros", "scales", "empty"):
                    self.assertIs(getattr(graph_state, name), getattr(eager, name))
                self.assertEqual(graph_state.python_dispatches, 0)
                self.assertEqual(graph_state.workspace_scope, "dedicated_graph_workspace")
                graph_state.workspace.fill_(5)
            self.assertTrue(torch.equal(eager.workspace, torch.zeros_like(eager.workspace)))
            self.assertEqual(eager.execution_stream, 100)
            self.assertEqual(eager.python_dispatches, 7)
            for name, saved in snapshot.items():
                self.assertTrue(torch.equal(saved, getattr(eager, name)))
        self.assert_restored(model, 2)

    def test_body_failure_removes_only_new_workspaces_and_retry_succeeds(self):
        model = self.model()
        with self.helper.model_graph_workspace(model):
            pass
        kept = [l._marlin_graph_states[0] for l in model]
        with self.assertRaisesRegex(RuntimeError, "body"):
            with self.helper.model_graph_workspace(model):
                raise RuntimeError("body")
        self.assert_restored(model, 1)
        self.assertTrue(all(l._marlin_graph_states[0] is s for l, s in zip(model, kept)))
        with self.helper.model_graph_workspace(model):
            pass
        self.assert_restored(model, 2)

    def test_partial_layer_preparation_failure_rolls_back(self):
        model = self.model()
        with patch.object(model[1]._marlin_state, "for_graph", side_effect=RuntimeError("prepare")):
            with self.assertRaisesRegex(RuntimeError, "prepare"):
                with self.helper.model_graph_workspace(model):
                    self.fail("must not enter body")
        self.assert_restored(model, 0)

    def test_register_buffer_failure_rolls_back_partial_layers(self):
        model = self.model()
        with patch.object(model[1], "register_buffer", side_effect=RuntimeError("register")):
            with self.assertRaisesRegex(RuntimeError, "register"):
                with self.helper.model_graph_workspace(model):
                    self.fail("must not enter body")
        self.assert_restored(model, 0)

    def test_capture_enter_and_exit_failures_roll_back(self):
        for point in ("enter", "exit"):
            with self.subTest(point=point):
                model = self.model()
                self.capture_failure = point
                with self.assertRaisesRegex(RuntimeError, point):
                    with self.helper.capture_model_graph(model, object()):
                        pass
                self.assert_restored(model, 0)

    def check_constructor_failure_rollback(self, filename, class_name):
        # Compile the actual constructor, not a handwritten imitation. Only
        # tensor device/forward/CUDA operations are mocked; workspace ownership
        # and Marlin for_graph execute their real implementations.
        source = ROOT / "nanovllm/engine" / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        init.decorator_list = []
        self.stack.enter_context(patch.object(torch.cuda, "Stream", side_effect=lambda **kw: self.make_stream("cuda:0")))
        self.stack.enter_context(patch.object(torch.cuda, "CUDAGraph", side_effect=lambda: SimpleNamespace(replay=Mock())))
        for point in ("warmup", "capture_forward", "enter", "exit"):
            with self.subTest(constructor=class_name, failure=point):
                model = self.model()
                # A previous successful graph must survive failed construction.
                with self.helper.model_graph_workspace(model):
                    pass
                kept = [layer._marlin_graph_states[0] for layer in model]
                reset = Mock()
                scope = {"torch": torch, "model_graph_workspace": self.helper.model_graph_workspace,
                         "reset_context": reset}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])),
                             str(source), "exec"), scope)
                injected = RuntimeError("injected " + point)
                self.capture_failure = point if point in ("enter", "exit") else None

                def forward():
                    self.assertTrue(all(layer._active_marlin is not layer._marlin_state for layer in model))
                    if point == "warmup" or (point == "capture_forward" and self.capturing):
                        raise injected
                    return torch.zeros(3)

                graph = SimpleNamespace(_step=Mock(side_effect=forward), _forward=Mock(side_effect=forward))
                block_table = torch.zeros((1, 1), dtype=torch.int32)
                args = (SimpleNamespace(model=model, block_table=block_table, capacity=16, device="cpu"), 2) \
                    if class_name == "_LogitsGraph" else (model, block_table, 16, 2, "cpu")
                with self.assertRaisesRegex(RuntimeError, "injected") as caught:
                    scope["__init__"](graph, *args)
                if point in ("warmup", "capture_forward"):
                    self.assertIs(caught.exception, injected)
                step = graph._forward if class_name == "_LogitsGraph" else graph._step
                self.assertEqual(step.call_count, 1 if point == "warmup" else 3 if point == "enter" else 4)
                reset.assert_called_once_with()
                self.assertIsNone(graph.graph)
                self.assertFalse(self.capturing)
                self.assertEqual(self.current.cuda_stream, 100)
                self.assert_restored(model, 1)
                for layer, state in zip(model, kept):
                    self.assertIs(layer._marlin_graph_states[0], state)
                # The same model must support a subsequent successful capture.
                self.capture_failure = None
                step.side_effect = lambda: torch.zeros(3)
                reset.reset_mock()
                scope["__init__"](graph, *args)
                self.assertIsNotNone(graph.graph)
                self.assert_restored(model, 2)
                reset.assert_called_once_with()

    def test_token_constructor_warmup_and_capture_rollback(self):
        self.check_constructor_failure_rollback("draft_graph.py", "DraftTokenGraph")

    def test_suffix_constructor_warmup_and_capture_rollback(self):
        self.check_constructor_failure_rollback("draft_graph.py", "DraftSuffixGraph")

    def test_logits_constructor_warmup_and_capture_rollback(self):
        self.check_constructor_failure_rollback("stochastic_draft.py", "_LogitsGraph")

    def test_success_retains_workspace_until_model_destruction(self):
        model = self.model(1)
        with self.helper.model_graph_workspace(model):
            state_ref = weakref.ref(model[0]._active_marlin)
            buffer_ref = weakref.ref(model[0]._active_marlin.workspace)
        gc.collect()
        self.assertIsNotNone(state_ref())
        self.assertIsNotNone(buffer_ref())
        del model
        gc.collect()
        self.assertIsNone(state_ref())
        self.assertIsNone(buffer_ref())

    def test_nested_scope_restores_previous_active_not_always_eager(self):
        model = self.model(1)
        with self.helper.model_graph_workspace(model):
            outer = model[0]._active_marlin
            with self.helper.model_graph_workspace(model):
                self.assertIsNot(model[0]._active_marlin, outer)
            self.assertIs(model[0]._active_marlin, outer)
        self.assert_restored(model, 2)

    def test_non_marlin_model_does_not_allocate_stream_or_workspace(self):
        model = torch.nn.Linear(2, 2, device="cpu")
        with self.helper.model_graph_workspace(model) as stream:
            self.assertIsNone(stream)
        explicit = FakeStream(50)
        with self.helper.model_graph_workspace(model, explicit) as stream:
            self.assertIs(stream, explicit)
        self.new_stream.assert_not_called()
        self.sync.assert_not_called()

    def test_for_graph_rejects_capture_and_wrong_device(self):
        state = self.state()
        self.capturing = True
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            state.for_graph(FakeStream(7))
        self.capturing = False
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            state.for_graph(FakeStream(7, "cuda:1"))
        self.sync.assert_not_called()

    def test_replay_binds_first_runtime_stream_not_capture_stream(self):
        model = self.model()
        with self.helper.model_graph_workspace(model):
            pass
        self.assertNotEqual(model._marlin_capture_stream.cuda_stream, self.current.cuda_stream)
        self.helper.assert_graph_replay_stream(model, "cuda:0")
        self.assertEqual(model._replay_stream_id, 100)
        self.helper.assert_graph_replay_stream(model, "cuda:0")
        self.current = FakeStream(101)
        with self.assertRaisesRegex(RuntimeError, "execution streams"):
            self.helper.assert_graph_replay_stream(model, "cuda:0")
        self.assertEqual(model._replay_stream_id, 100)

    def test_replay_default_stream_zero_is_not_unbound(self):
        owner = SimpleNamespace()
        self.current = FakeStream(0)
        self.helper.assert_graph_replay_stream(owner, "cuda:0")
        self.current = FakeStream(4)
        with self.assertRaises(RuntimeError):
            self.helper.assert_graph_replay_stream(owner, "cuda:0")
        self.assertEqual(owner._replay_stream_id, 0)

    def test_replay_cpu_noop_and_independent_owners(self):
        cpu, a, b = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
        self.helper.assert_graph_replay_stream(cpu, "cpu")
        self.assertFalse(hasattr(cpu, "_replay_stream_id"))
        self.helper.assert_graph_replay_stream(a, "cuda:0")
        self.current = FakeStream(101)
        self.helper.assert_graph_replay_stream(b, "cuda:0")
        self.assertEqual((a._replay_stream_id, b._replay_stream_id), (100, 101))

    def test_real_replay_entrypoints_reject_before_static_access(self):
        # Execute actual method ASTs with decorators removed. A sentinel owner
        # permits only device discovery; any static read/write before the guard
        # raises AssertionError. This avoids importing GPU engine dependencies.
        cases = [
            ("draft_graph.py", "DraftTokenGraph", "run", (1, 0, 1), {"state": SimpleNamespace(device="cuda:0")}),
            ("draft_graph.py", "DraftSuffixGraph", "run", ([1, 2], 0), {"state": SimpleNamespace(device="cuda:0")}),
            ("stochastic_draft.py", "_LogitsGraph", "run", (SimpleNamespace(device="cuda:0"), 0), {"ids": SimpleNamespace(device="cuda:0")}),
            ("native_verification_graph.py", "NativeVerificationGraphs", "run", (SimpleNamespace(device="cuda:0"), None, None), {}),
            ("native_verification_graph.py", "NativeVerificationGraphs", "prepare_cpu", (2, [], [], [], [], 2), {"metadata_packets": {2: {"packet": SimpleNamespace(device="cuda:0")}}}),
            ("native_verification_graph.py", "NativeVerificationGraphs", "replay_prepared", (2,), {"metadata_packets": {2: {"packet": SimpleNamespace(device="cuda:0")}}}),
        ]
        for filename, cls_name, method, args, fields in cases:
            with self.subTest(entry=f"{cls_name}.{method}"):
                tree = ast.parse((ROOT / "nanovllm/engine" / filename).read_text(encoding="utf-8"))
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
                node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
                node.decorator_list = []
                code = ast.Module(body=[node], type_ignores=[])
                scope = {"assert_graph_replay_stream": self.helper.assert_graph_replay_stream, "torch": torch}
                exec(compile(ast.fix_missing_locations(code), filename, "exec"), scope)

                class Owner:
                    def __getattr__(self, name):
                        raise AssertionError(f"static access before rejection: {name}")

                owner = Owner()
                owner.__dict__.update(fields, _replay_stream_id=999)
                with self.assertRaisesRegex(RuntimeError, "execution streams"):
                    scope[method](owner, *args)

    def test_decode_rejects_before_static_updates_or_replay(self):
        filename = ROOT / "nanovllm/engine/model_runner.py"
        tree = ast.parse(filename.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_model")
        method.decorator_list = []
        buffers = {name: torch.full((1, 2), 17, dtype=torch.int64) for name in
                   ("input_ids", "positions", "slot_mapping", "context_lens", "block_tables", "outputs")}
        original = {name: value.clone() for name, value in buffers.items()}
        graph = SimpleNamespace(replay=Mock())
        owner = SimpleNamespace(graph_counters=Counter(), enforce_eager=False,
                                graph_bs=[1], graphs={1: graph}, graph_vars=buffers,
                                _replay_stream_id=999, _plan_flashinfer_decode=Mock())
        ids = SimpleNamespace(device="cuda:0", size=lambda dimension: 1)
        context = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
        scope = {"torch": torch, "Counter": Counter, "get_context": lambda: context}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(filename), "exec"), scope)
        with patch.dict(sys.modules, {"nanovllm.layers.quant_graph": self.helper}):
            with self.assertRaisesRegex(RuntimeError, "execution streams"):
                scope["run_model"](owner, ids, None, False)
        for name, value in buffers.items():
            self.assertTrue(torch.equal(value, original[name]), name)
        graph.replay.assert_not_called()
        owner._plan_flashinfer_decode.assert_not_called()
        self.assertEqual(owner.graph_counters, Counter())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, unittest_args = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    for relative in ("nanovllm/layers/quant_graph.py", "nanovllm/layers/awq_marlin_sidecar.py",
                     "nanovllm/engine/draft_graph.py", "nanovllm/engine/stochastic_draft.py"):
        print(f"SOURCE {relative} sha256={hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()}", flush=True)
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
