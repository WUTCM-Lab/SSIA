from mindspore import nn, load_checkpoint, load_param_into_net, context, Tensor
from mindspore.common.initializer import Normal
from mindspore.nn import CosineSimilarity
import copy
import mindspore
from .resnet import resnet18
from .crossformer import CrossFormer
from .AMFMN_Modules import CrossAttention

class BaseModel(nn.Cell):
    def __init__(self, opt={}):
        super(BaseModel, self).__init__()

        self.model = CrossFormer()
        # param_dict = load_checkpoint("layers/crossformer-s.ckpt")
        # load_param_into_net(self.model, param_dict)
        
        self.aud_feature = resnet18(num_classes=64)
        # param_dict = load_checkpoint("layers/audioset_audio_pretrain.ckpt")
        # load_param_into_net(self.aud_feature, param_dict)
        
        self.cross_attention_s = CrossAttention()
        
        
    def construct(self, img, text):
        
        text_feature, x1 = self.aud_feature(text)

        mvsa_feature_nt = self.model(img, x1)

        Ft, mvsa_feature = self.cross_attention_s(mvsa_feature_nt, text_feature)
        
        return Ft, mvsa_feature, mvsa_feature_nt, text_feature


def factory(opt):
    opt = copy.deepcopy(opt)
    model = BaseModel(opt)
    return model

if __name__ == '__main__':
    img = mindspore.ops.Ones()((60,3,224,224), mindspore.float32)
    text = mindspore.ops.Ones()((60,1,300,64), mindspore.float32)
    model = BaseModel()
    
    dual_sim, loss = model(img, text, True)
    print(dual_sim.shape)
    print(loss)
    # done!