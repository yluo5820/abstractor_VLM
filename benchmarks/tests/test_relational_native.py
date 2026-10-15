"""Small random native LFM integration: run inside a CPU/GPU Slurm allocation.

No pretrained weights or dataset questions are needed. This tests actual HF
forward/generation hooks in addition to the real-weight H200 correctness pilots.
"""
import unittest
from types import SimpleNamespace

import torch
from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
from peft import LoraConfig, get_peft_model

from vlm_bench.relational import install, VARIANTS


class NativeIntegration(unittest.TestCase):
    def test_native_forward_backward_and_cache(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(31)
                config=Lfm2VlConfig(image_token_id=396, downsample_factor=2, projector_hidden_size=64,
                    text_config=dict(model_type='lfm2',hidden_size=32,intermediate_size=64,
                        block_multiple_of=8, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, layer_types=['conv','conv','full_attention','conv'],
                        vocab_size=512, pad_token_id=0, bos_token_id=1, eos_token_id=2),
                    vision_config=dict(model_type='siglip2_vision_model', hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1,num_attention_heads=4,patch_size=2,num_patches=16,vision_use_head=False))
                base=Lfm2VlForConditionalGeneration(config)
                base.eval();base.requires_grad_(False)
                inp=dict(input_ids=torch.tensor([[1]+[396]*8+[9,10,11,12]]),
                         attention_mask=torch.ones(1,13,dtype=torch.long),
                         pixel_values=torch.randn(2,16,12),pixel_attention_mask=torch.ones(2,16,dtype=torch.long),
                         spatial_shapes=torch.tensor([[4,4],[4,4]]))
                with torch.no_grad(): expected=base(**inp,use_cache=False).logits
                targets=[n for n,m in base.named_modules() if n.startswith('model.language_model.') and isinstance(m,torch.nn.Linear)]
                model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=targets))
                processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k:[10,11]))
                ctl=install(base,processor,variant)
                model.eval()
                with torch.no_grad(): actual=model(**inp,use_cache=False).logits
                self.assertTrue(torch.equal(expected,actual),variant)
                optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.01)
                for _ in range(2):
                    optimizer.zero_grad()
                    logits=model(**inp,use_cache=False).logits
                    torch.nn.functional.cross_entropy(logits[:,-2],torch.tensor([12])).backward()
                    optimizer.step()
                self.assertGreater(ctl.probe.layers[0].relation.q.weight.grad.abs().sum().item(),0)
                model.eval()
                other={**inp,'input_ids':inp['input_ids'].clone()};other['input_ids'][0,-1]=22
                with torch.no_grad():
                    first=model(**inp,use_cache=False).logits[:,11]
                    second=model(**other,use_cache=False).logits[:,11]
                self.assertTrue(torch.equal(first,second),variant)
                prefix={k:(v[:,:-1] if k in ('input_ids','attention_mask') else v) for k,v in inp.items()}
                with torch.no_grad():
                    cached=model.generate(**prefix,max_new_tokens=3,do_sample=False,use_cache=True)
                    plain=model.generate(**prefix,max_new_tokens=3,do_sample=False,use_cache=False)
                self.assertTrue(torch.equal(cached,plain),variant)
                print('PASS',variant,flush=True)


if __name__=='__main__': unittest.main(verbosity=2)
