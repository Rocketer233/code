import torch
from torch import nn, einsum
import warnings


#from transformers.models.llama import LlamaTokenizer
#from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device=device)

    #tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    #model = AutoModelForCausalLM.from_pretrained("Cheng98/llama-160m")

    torch.manual_seed(42)

    b = 4 # batch size
    d = 8 # model dim
    m = 4 # number of previous tokens, i.e. trying to predict token number m+1
    h = 4 # number of heads
    k = v = d // h # dimension of q, k, v heads

    x = torch.randn(b, m, d)
    P_q = torch.randn(h, d, k)
    P_k = torch.randn(d, k)
    P_v = torch.randn(d, v)
    P_o = torch.randn(h, d, v)

    mask = torch.tril(torch.ones(m, m))
    mask[mask==0] = -torch.inf
    mask[mask==1] = 0

    def mqa_logic(x, mask=mask, prev_K=None, prev_V=None, P_q=P_q, P_k=P_k, P_v=P_v, P_o=P_o):
        if prev_K is not None and prev_V is not None:
            inp = x[:, -1, :]
            queries = einsum("bd,hdk->bhk", inp, P_q)
            key = einsum("bd,dk->bk", inp, P_k).unsqueeze(dim=1)
            new_K = torch.cat(prev_K, key, dim=1)
            value = einsum("bd,dv->bv", inp, P_v).unsqueeze(dim=1)
            new_V = torch.cat(prev_V, value, dim=1)

            logits = einsum("bhk,bmk->bhm", queries, new_K)
            weights = nn.functional.softmax(logits, dim=-1)
            o = einsum("bhm,bmv->bhv", weights, new_V)
            y = einsum("bhv,hdv->bd", o, P_o)

        else:
            queries = einsum("bnd,hdk->bhnk", x, P_q)
            new_K = einsum("bmd,dk->bmk", x, P_k)
            new_V = einsum("bmd,dv->bmv", x, P_v)

            logits = einsum("bhnk,bmk->bhnm", queries, new_K)
            weights = nn.functional.softmax(logits + mask, dim=-1)
            o = einsum("bhnm,bmv->bhnv", weights, new_V)
            y = einsum("bhnv,hdv->bnd", o, P_o)
        return y, new_K, new_V


def mha2mqa(state_dict, num_layers: int, num_heads: int, transpose_layer=True):
    warnings.warn("Need to manually set the layer name if model is changed! Default is llama-160m")

# transpose_layer should always be True, TESTED
    for layer_id in range(num_layers):
        layer_name = f'model.layers.{layer_id}.self_attn.k_proj.weight' # name of the attention layer projection matrices
        layer = state_dict[layer_name]
        if transpose_layer: layer = layer.transpose(0, 1)
        layer = torch.stack(torch.tensor_split(layer, num_heads, dim=1), dim=0) #size[1] -> model dimension, for llama-160m o/p size is [12, 768, 64]
        layer = torch.mean(layer, dim=0)
        state_dict[layer_name] = layer.transpose(0, 1)

    for layer_id in range(num_layers):
        layer_name = f'model.layers.{layer_id}.self_attn.v_proj.weight' # name of the attention layer projection matrices
        layer = state_dict[layer_name]
        if transpose_layer: layer = layer.transpose(0, 1)
        layer = torch.stack(torch.tensor_split(layer, num_heads, dim=1), dim=0) #size[1] -> model dimension, for llama-160m o/p size is [12, 768, 64]
        layer = torch.mean(layer, dim=0)
        state_dict[layer_name] = layer.transpose(0, 1)

    return state_dict


def mha2gqa(state_dict, groups_idx, num_heads, transpose_layer=True):
    """Uniform grouping"""
    warnings.warn("Need to manually set the layer name if model is changed! Default is llama-160m")
    
    # transpose_layer should always be True, TESTED
    # There is a chance that the passed groupings might have the same head in two or more groups, 
    # either raise an error, or change the way the groupings are passed so this may not happen
    num_layers = len(groups_idx["k"])
    for t in ("k", "v"):
        for layer_id in range(num_layers):
            layer_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.weight' # name of the attention layer projection matrices
            layer = state_dict[layer_name]
            if transpose_layer: layer = layer.transpose(0, 1)
            # Code below unit tested
            # torch.Size([768, 768])
            # torch.Size([12, 768, 64])
            # torch.Size([768, 512])
            # print("fuckfuckfuck")
            layer = torch.stack(torch.tensor_split(layer, num_heads, dim=1), dim=0)
            layer = torch.cat([torch.mean(layer[group, :, :], dim=0) for group in groups_idx[t][layer_id]], dim=1) # [768, 64 * num_groups]
            state_dict[layer_name] = layer.transpose(0, 1) # [64 * num_groups, 768]
            
            # layer_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.bias' # name of the attention layer projection matrices
            # layer = state_dict[layer_name]
            # # print(layer.shape)
            # layer = torch.stack(torch.tensor_split(layer, num_heads))
            # # print(layer.shape)
            # # print([torch.mean(layer[group, :], dim=0) for group in groups_idx[layer_id]])
            # layer = torch.cat([torch.mean(layer[group, :], dim=0) for group in groups_idx[t][layer_id]])
            # # print(layer.shape)
            # state_dict[layer_name] = layer

    return state_dict


def mha2gqa_lora(state_dict, groups_idx, num_heads, transpose_layer=True):
    """Uniform grouping"""
    warnings.warn("Need to manually set the layer name if model is changed! Default is llama-160m")
    
    num_layers = len(groups_idx["k"])
    for layer_id in range(num_layers):
        for head in ["k", "v"]:
            layer_name = f'model.decoder.layers.{layer_id}.self_attn.{head}_proj.lora_B.eng_alpaca.weight'
            layer = state_dict[layer_name]
            if transpose_layer: layer = layer.transpose(0, 1)
            layer = torch.stack(torch.tensor_split(layer, num_heads, dim=1), dim=0)
            layer = torch.cat([torch.mean(layer[group, :, :], dim=0) for group in groups_idx[head][layer_id]], dim=1) # [768, 64 * num_groups]
            state_dict[layer_name] = layer.transpose(0, 1) # [64 * num_groups, 768]
            
            layer_name = f'model.decoder.layers.{layer_id}.self_attn.{head}_proj.weight' # name of the attention layer projection matrices
            layer = state_dict[layer_name]
            if transpose_layer: layer = layer.transpose(0, 1)
            layer = torch.stack(torch.tensor_split(layer, num_heads, dim=1), dim=0)
            layer = torch.cat([torch.mean(layer[group, :, :], dim=0) for group in groups_idx[head][layer_id]], dim=1) # [768, 64 * num_groups]
            state_dict[layer_name] = layer.transpose(0, 1) # [64 * num_groups, 768]
            
            # layer_name = f'model.decoder.layers.{layer_id}.self_attn.{head}_proj.bias' # name of the attention layer projection matrices
            # layer = state_dict[layer_name]
            # # print(layer.shape)
            # layer = torch.stack(torch.tensor_split(layer, num_heads))
            # # print(layer.shape)
            # # print([torch.mean(layer[group, :], dim=0) for group in groups_idx[layer_id]])
            # layer = torch.cat([torch.mean(layer[group, :], dim=0) for group in groups_idx[layer_id]])
            # # print(layer.shape)
            # state_dict[layer_name] = layer

    
    return state_dict