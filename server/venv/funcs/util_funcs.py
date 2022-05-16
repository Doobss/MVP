import torch as T
comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')



'''UTIL FUNCS'''
def build_mult_tensor(vals, fill_val=0):
    #inputs a 1D tensor of integer values.
    
    #returns a 2D tensor where the first dimenson length is equal to the number of values past and the second dimenson length is equal to the 
    #greatest value in the input tensor. The return tensor has a continuous sequence of 1's along the second dimenson up to the value passed 
    #in the input tensor.
    #example: 
    #input=[2,4,6]
    #return=[[1,1,0,0,0,0],
    #        [1,1,1,1,0,0],
    #        [1,1,1,1,1,1]]
    val_max=int(vals.max())
    fill_tensor_vals=T.ones([vals.shape[0], val_max], dtype=T.long, device=comp_dev)
    dim_1_vals=T.tensor([i for i in range(val_max)], dtype=T.long, device=comp_dev).unsqueeze(dim=0)
    dim_0_vals_cat=T.cat([vals.unsqueeze(dim=1) for i in range(val_max)], dim=1)
    dim_1_vals_cat=T.cat([dim_1_vals for i in range(vals.shape[0])], dim=0)
    fill_tensor_vals=T.ones(dim_1_vals_cat.shape, dtype=vals.dtype, device=comp_dev)
    returned_vals=T.where(dim_0_vals_cat > dim_1_vals_cat, fill_tensor_vals, ((dim_0_vals_cat * 0)+fill_val))
    return returned_vals


def build_comparison_tensors(ten_0, ten_1):
    #inputs are two one dimensonal tensors. 
    
    #returns two tensors that are the same shape and ready to compare
    cat_list=[]
    cat_shape=[ten_0.shape[0],ten_1.shape[0]]
    to_cat_list=[ten_1,ten_0]
    for i, vec in enumerate(to_cat_list):
        cat=T.cat([vec.unsqueeze(dim=i) for k in range(cat_shape[i])],dim=i)
        cat_list.append(cat)
    return cat_list