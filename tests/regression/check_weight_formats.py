"""CPU 构造/加载回归：量化格式、融合 shard、缺失普通权重以及显式开关。"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config
from nanovllm.utils.quant_config import configure_weight_format
from nanovllm.utils.loader import load_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.awq_packed import AWQPackedLinear


META = dict(quant_method='awq',bits=4,group_size=128,zero_point=True,version='gemm')


class WeightFormatTests(unittest.TestCase):
    def test_memory_counts_quant_buffers_and_tied_storage_once(self):
        from nanovllm.utils.weight_memory import weight_storage_bytes
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.empty(16,dtype=torch.float16))
        model.b = torch.nn.Parameter(model.a.data)
        model.backend = 'awq_marlin'
        weight = torch.empty(32,dtype=torch.int32)
        model._marlin_state = SimpleNamespace(weight=weight,zeros=torch.empty(4,dtype=torch.int32),
                                              scales=torch.empty(8,dtype=torch.float16))
        model.register_buffer('repacked',weight)
        model.register_buffer('unrelated_workspace',torch.empty(999,dtype=torch.int32))
        self.assertEqual(weight_storage_bytes(model),32+128+16+16)

    def config(self):
        return Qwen3Config(hidden_size=128,intermediate_size=256,num_hidden_layers=1,
            num_attention_heads=1,num_key_value_heads=1,head_dim=128,vocab_size=32,
            max_position_embeddings=32,attention_bias=False,dtype=torch.float16,
            quantization_config=META,tie_word_embeddings=False)

    def test_explicit_format_dtype_and_reject_unsupported(self):
        hf = self.config()
        with self.assertRaises(ValueError): configure_weight_format(hf)
        with self.assertRaises(ValueError): configure_weight_format(hf,'gptq')
        with self.assertRaises(ValueError): configure_weight_format(hf,'awq','bfloat16')
        configure_weight_format(hf,'awq','float16')
        self.assertEqual(hf.nanovllm_quantization,'awq')
        hf.quantization_config = dict(META,version='gemv')
        with self.assertRaises(ValueError): configure_weight_format(hf,'awq')

    def make_model(self):
        hf = configure_weight_format(self.config(),'awq')
        with patch('torch.distributed.get_world_size',return_value=1), \
             patch('torch.distributed.get_rank',return_value=0), torch.device('cpu'):
            return Qwen3ForCausalLM(hf)

    def checkpoint(self,model):
        state = {}
        for name,param in model.named_parameters():
            # 非融合字段直接保存；融合字段写回原 checkpoint 的独立 shard。
            module = model.get_submodule(name.rsplit('.',1)[0])
            field = name.rsplit('.',1)[1]
            value = torch.ones_like(param,device='cpu').contiguous()
            if isinstance(module,AWQPackedLinear) and len(module.shard_ids)>1:
                sizes = [s//8 if field in ('qweight','qzeros') else s for s in module.output_sizes]
                pieces = value.split(sizes,dim=1)
                old = ('q_proj','k_proj','v_proj') if 'qkv_proj' in name else ('gate_proj','up_proj')
                merged = 'qkv_proj' if 'qkv_proj' in name else 'gate_up_proj'
                for key,piece in zip(old,pieces): state[name.replace(merged,key)] = piece.contiguous()
            else:
                state[name] = value
        return state

    def test_complete_fused_load_has_no_dense_linear_weights(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as folder:
            save_file(self.checkpoint(model),str(Path(folder)/'model.safetensors'))
            load_model(model,folder)
        self.assertEqual(model.quantization_load_report['packed_modules'],4)
        for module in model.modules():
            if isinstance(module,AWQPackedLinear):
                module.assert_loaded()
                self.assertFalse(hasattr(module,'weight'))

    def test_native_bf16_scales_are_preserved_without_rounding(self):
        model = self.make_model()
        state = self.checkpoint(model)
        for name in state:
            if name.endswith('.scales'):
                state[name] = torch.full_like(state[name],1e-9,dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as folder:
            save_file(state,str(Path(folder)/'model.safetensors'))
            load_model(model,folder)
        for module in model.modules():
            if isinstance(module,AWQPackedLinear):
                self.assertEqual(module.scales.dtype,torch.bfloat16)
                self.assertTrue(bool((module.scales>0).all()))
                self.assertEqual(module.scales.flatten()[0].item(),torch.tensor(1e-9,dtype=torch.bfloat16).item())

    def test_missing_unquantized_weight_fails(self):
        model = self.make_model()
        state = self.checkpoint(model)
        del state['model.norm.weight']
        with tempfile.TemporaryDirectory() as folder:
            save_file(state,str(Path(folder)/'model.safetensors'))
            with self.assertRaisesRegex(ValueError,'incomplete AWQ'):
                load_model(model,folder)

    def test_missing_fused_shard_fails(self):
        model = self.make_model()
        state = self.checkpoint(model)
        del state['model.layers.0.self_attn.v_proj.scales']
        with tempfile.TemporaryDirectory() as folder:
            save_file(state,str(Path(folder)/'model.safetensors'))
            with self.assertRaises((ValueError,RuntimeError)):
                load_model(model,folder)


if __name__=='__main__':
    unittest.main()
