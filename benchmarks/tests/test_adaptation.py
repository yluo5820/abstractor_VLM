import unittest
import torch
from vlm_bench.adaptation import suffix_loss
from vlm_bench.adaptation_data import partition_ids

class AdaptationTests(unittest.TestCase):
    def test_family_without_logit_optimization_has_identical_loss_and_gradients(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.values=torch.nn.Parameter(torch.randn(1,7,11))
            def forward(self,input_ids,use_cache):
                return type('Result',(),{'logits':self.values})()
        model=Model(); inputs={'input_ids':torch.ones((1,7),dtype=torch.long)}
        labels=torch.tensor([[-100,-100,-100,-100,2,3,4]])
        got=suffix_loss(model,inputs,labels,4)
        expected=torch.nn.functional.cross_entropy(model.values[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(got,expected)
        g1=torch.autograd.grad(got,model.values,retain_graph=True)[0]
        g2=torch.autograd.grad(expected,model.values)[0]
        torch.testing.assert_close(g1,g2)
    def test_each_clevr_task_preserves_disjoint_full_split(self):
        for task in ['rmts','idrules']:
            train,val=partition_ids(task=task)
            self.assertEqual(len(train),9000);self.assertEqual(len(val),1000)
            self.assertFalse(set(train)&set(val))
            self.assertTrue(all('/'+task+'/' in x for x in train+val))

if __name__=='__main__': unittest.main()
