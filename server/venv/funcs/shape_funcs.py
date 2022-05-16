import torch as T



comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')





def protein_shape_calc(passed_prot_vec, passed_am_coding_vals, amino_dims=3):
    prot_added_vec=T.zeros([amino_dims],dtype=passed_am_coding_vals.dtype,device=comp_dev)
    prot_type=T.ones([1],dtype=passed_am_coding_vals.dtype,device=comp_dev)
    prot_shape=T.zeros([amino_dims],dtype=T.long,device=comp_dev).unsqueeze(dim=0)
    for i, val in enumerate(passed_prot_vec):
        if val==(-1):
            break
        am_vals=passed_am_coding_vals[val]
        am_type=am_vals[0]
        am_type_mult_prot=am_type.mul(prot_type)
        am_added=am_vals[1:].mul(am_type_mult_prot)
        prot_type=am_type_mult_prot
        if i!=0:
            prot_added_vec=prot_added_vec.add(am_added)
            prot_shape=T.cat([prot_shape, prot_added_vec.unsqueeze(dim=0).clone().detach().to(prot_shape.dtype)],dim=0)
    prot_dim_min=T.tensor([int(dim.min()) for dim in prot_shape.split(1,dim=1)],dtype=T.long,device=comp_dev)
    prot_shape=prot_shape.add(-prot_dim_min)
    return prot_shape, prot_added_vec




def find_protein_shapes(passed_prot_vecs, passed_am_coding_vals):
    prot_shapes_list=[]
    prot_added_vec_list=[]
    for j, vec in enumerate(passed_prot_vecs):
        prot_shape, prot_added_vec=protein_shape_calc(vec, passed_am_coding_vals)
        #print(prot_shape.tolist())
        #prot_shapes_list.append(prot_shape.tolist())
        prot_shapes_list.append(prot_shape.tolist())
        prot_added_vec_list.append(prot_added_vec)
    prot_added_vecs=T.cat([vecs.unsqueeze(dim=0) for vecs in prot_added_vec_list],dim=0)
    return prot_shapes_list, prot_added_vecs





