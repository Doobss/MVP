import torch as T
import math
comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')

import util_funcs





'''PROT INIT FUNCS'''
def assign_amino_to_codon(num_codons, num_aminos, num_stop_aminos, start_amino_val, stop_amino_val):
    #inputs the number of codons you want to use, the number of aminos you want to use, how many stop aminos you want, the start amino val and
    #the stop amino vals.

    #returns a 1D tensor wuth random amino values assigned to a codon.
    codon_amino_vals=T.full([num_codons],start_amino_val,dtype=T.long,device=comp_dev)
    stop_index_val=num_stop_aminos+1
    codon_amino_vals[1:stop_index_val]=stop_amino_val
    num_of_coding_am=(num_aminos-2)
    rand_vals=T.randint(2,num_aminos,[(num_codons-stop_index_val)],device=comp_dev)
    codon_amino_vals[stop_index_val:]=rand_vals
    return codon_amino_vals, num_of_coding_am













'''PROT UTIL FUNCS'''
def build_dna_combo_seq(dna, len_to_split):
    #inputs a dna base sequence and the codon length.

    #returns a 2D tensor that has every combonation of each index within the codon length.
    dna_len=dna.shape[0]
    codon_comb_vec=T.cat([dna[i:(i+len_to_split)].unsqueeze(dim=0) for i in range((dna_len-(len_to_split-1)))],dim=0)
    return codon_comb_vec


    '''Para'''
def build_dna_combo_seq_para(dna, len_to_split):
    #Parallel version of the func above. It does the same but for multiple dna sequences at once.
    #inputs a dna base sequence and the codon length.

    #returns a 2D tensor that has every combonation of each index within the codon length.
    dna_len=dna.shape[1]
    codon_comb_vec=T.cat([dna[:,i:(i+len_to_split)].unsqueeze(dim=1) for i in range((dna_len-(len_to_split-1)))],dim=1)
    return codon_comb_vec


    '''Para'''
def build_para_dna_tensor(dna_list):
    #inputs Takes a list of dna 1D tensors.

    #returns one 2D tensor with all of the dna sequences in it. An empty space in represented by a (-1).
    num_dna=len(dna_list)
    dna_len=T.tensor([dna.shape[0] for dna in dna_list],dtype=T.long,device=dna_list[0].device)
    max_len=int(dna_len.max())
    dna_para=T.full([num_dna,max_len],-1,dtype=dna_list[0].dtype,device=dna_list[0].device)
    dna_ind=T.nonzero(util_funcs.build_mult_tensor(dna_len))
    #added one so I can tell the difference between an empty cell and the first index
    dna_vals=T.cat(dna_list,dim=0)
    dna_para[dna_ind.split(1,dim=1)]=dna_vals.unsqueeze(dim=1)
    return dna_para














'''PROT MAIN FUNCS'''
def find_amino_base_difference(dna_len, start_amino_ind, stop_amino_ind, codon_len):
    #inputs a 1D tensor of dna bases, the amino val that commands to start building an amino chain, the amino val that commands to stop
    #building an amino chain, and the number of dna bases a codon has (aka codon length).

    #returns a 2D tensor where the first dimension length is equal to the number of matched start codons and the second dimension is always 2
    #The valuse in the first coloum are the indexes with a start command and the values in the second coloum are the indexes of the stop
    #command. There isnt always a stop command or more than CODON_LEN bases to be found so for the former case we go to the end of the dna seq
    #and round down to the last readable codon and in the later case we ignore the start codon because there isnt enough bases to read even
    #one codon.

    #TLDR:calculates the number of aminos between the start amino index to the stop amino index for all indexes passed.
    stop_amino_ind_adj=stop_amino_ind.add(-codon_len)
    compare_list=util_funcs.build_comparison_tensors(start_amino_ind, stop_amino_ind_adj)
    ind_dif=compare_list[0].add(-compare_list[1]).to(T.float32)
    le_zero_ind=T.nonzero(ind_dif.le(0)).to(T.long)
    ind_dif[le_zero_ind.split(1,dim=1)]=1
    mod=ind_dif.fmod(codon_len)
    mod_eq_zero=mod.eq(0)
    mod_eq_zero_sum=mod_eq_zero.sum(dim=1)
    no_end_amino_ind=T.nonzero(mod_eq_zero_sum.eq(0)).flatten()
    mod_zero_ind=T.nonzero(mod_eq_zero)
    zero_vals=T.full(mod.shape,dna_len,dtype=T.long,device=start_amino_ind.device)
    zero_vals[mod_zero_ind.split(1,dim=1)]=mod_zero_ind[:,1].unsqueeze(dim=1)
    min_vals_ind=zero_vals.min(dim=1)[1]
    start_amino_ind_to_set=T.tensor([i for i in range(start_amino_ind.shape[0])],dtype=T.long,device=start_amino_ind.device)
    stop_amino_for_start_amino=T.zeros(start_amino_ind.shape,dtype=T.long,device=start_amino_ind.device)
    stop_amino_for_start_amino[start_amino_ind_to_set]=stop_amino_ind_adj[min_vals_ind].to(stop_amino_for_start_amino.dtype)
    returned_prots=T.ones(start_amino_ind.shape,dtype=T.long,device=start_amino_ind.device)
    no_stop_amino_start_ind=start_amino_ind[no_end_amino_ind]
    no_stop_dif=no_stop_amino_start_ind.add(-dna_len).abs()
    no_stop_mod=no_stop_dif.fmod(codon_len)
    no_stop_seq_end=no_stop_dif.add(-no_stop_mod)
    lt_codon_len=T.nonzero(no_stop_seq_end.lt(codon_len)).flatten()
    no_stop_seq_end=no_stop_seq_end.add(no_stop_amino_start_ind)
    stop_amino_for_start_amino[no_end_amino_ind]=no_stop_seq_end
    returned_prots[no_end_amino_ind[lt_codon_len]]=0
    returned_prots_ind=T.nonzero(returned_prots).flatten()
    stop_amino_for_start_amino=stop_amino_for_start_amino[returned_prots_ind]
    start_amino_returned=start_amino_ind[returned_prots_ind]
    returned_vals=T.cat([start_amino_returned.unsqueeze(dim=1), stop_amino_for_start_amino.unsqueeze(dim=1)],dim=1)
    return returned_vals


def build_protein_base_sequence(dna, start_amino_ind, stop_amino_ind, codon_len):
    #inputs a 1D tensor of dna bases, the amino val that commands to start building an amino chain, the amino val that commands to stop
    #building an amino chain, and the number of dna bases a codon has (aka codon length).

    #returns a 2D tensor where the first dimenson is the number of protiens built by the given inputs and the second dimenson is the base
    #sequence of those protiens. It also returns a 1D tensor that contains the number of bases in each protien.
    dna_len=dna.shape[0]
    amino_inds=find_amino_base_difference(dna_len, start_amino_ind, stop_amino_ind, codon_len)
    num_of_bases=amino_inds[:,1].add(-amino_inds[:,0])
    mult_ind_tensor=T.tensor([i for i in range(int(num_of_bases.max()))],dtype=T.long,device=start_amino_ind.device).add(1)
    all_mult_tensor=util_funcs.build_mult_tensor(num_of_bases)
    all_mult_tensor=all_mult_tensor.mul(mult_ind_tensor)
    amino_adj_inds=all_mult_tensor.rot90(1,[0,1]).add(amino_inds[:,0]).rot90(1,[1,0]).mul(all_mult_tensor.gt(0))
    dna_mult_inds=T.nonzero(amino_adj_inds)
    dna_base_ind_vals=amino_adj_inds[dna_mult_inds.split(1,dim=1)].flatten()
    base_vals=T.full(amino_adj_inds.shape,-1,dtype=dna.dtype,device=start_amino_ind.device)
    #base_vals[dna_mult_inds.split(1,dim=1)]=dna[dna_base_inds[:,1].add(-1)].unsqueeze(dim=1)
    base_vals[dna_mult_inds.split(1,dim=1)]=dna[dna_base_ind_vals.add(-1)].unsqueeze(dim=1)
    return base_vals, num_of_bases


def build_protein_codon_sequence(dna, start_amino_ind, stop_amino_ind, vec_to_ind, codon_len):
    #inputs a 1D tensor of dna bases, the amino val that commands to start building an amino chain, the amino val that commands to stop
    #building an amino chain, a multiplyer that helps compute a base sequence to its codon val, and the number of dna bases a codon has (aka
    #codon length)

    #returns a 2D tensor where the first dimenson is the number of protiens built by the given inputs and the second dimenson is the codon
    #sequence of those protiens. It also returns a 1D tensor that contains the number of codons in each protien.
    base_seq,num_of_bases=build_protein_base_sequence(dna, start_amino_ind, stop_amino_ind, codon_len)
    codon_seq=T.cat([codon.unsqueeze(dim=1) for codon in base_seq.split(codon_len,dim=1)],dim=1).mul(vec_to_ind).sum(dim=2).to(T.long)
    codon_seq=T.where(codon_seq==(-int(vec_to_ind.sum())), ((codon_seq*0)-1),codon_seq)
    num_of_codons=num_of_bases.true_divide(codon_len).to(T.long)
    return codon_seq, num_of_codons


def find_amino_index(dna, vec_index, amino_val, amino_codon_vals, codon_len):
    #inputs a 1D tensor of dna bases, a multiplyer that helps compute a base sequence to its codon val, an amino val,  a tensor that gives the
    #amino val for each codon, the number of dna bases a codon has (aka codon length).

    #returns a 1D tensor of indexes that match the amino val.
    amino_codon_list=T.nonzero(amino_codon_vals.eq(amino_val)).flatten().tolist()
    dna_len=dna.shape[0]
    codon_comb_vec=build_dna_combo_seq(dna, codon_len)
    codon_comb_index=codon_comb_vec.mul(vec_index).sum(dim=1).to(T.long)
    codon_eq_tot=T.zeros(codon_comb_index.shape, dtype=T.long,device=dna.device)
    for ind_val in amino_codon_list:
        ind_val=int(ind_val)
        codon_eq=codon_comb_index.eq(ind_val).to(T.long)
        codon_eq_tot=codon_eq_tot.add(codon_eq)
    codon_index=T.nonzero(codon_eq_tot).flatten()
    codon_index=codon_index.add(3)
    return codon_index


    '''Para'''
def find_amino_index_para(dna, vec_index, amino_val, amino_codon_vals, codon_len):
    #Parallel version of the func above. It does the same but for multiple dna sequences at once.
    #inputs a 1D tensor of dna bases, a multiplyer that helps compute a base sequence to its codon val, an amino val,  a tensor that gives the
    #amino val for each codon, the number of dna bases a codon has (aka codon length).

    #returns a 1D tensor of indexes that match the amino val.
    amino_codon_list=T.nonzero(amino_codon_vals.eq(amino_val)).tolist()
    dna_len=dna.shape[1]
    codon_comb_vec=build_dna_combo_seq_para(dna, codon_len)
    codon_comb_index=codon_comb_vec.mul(vec_index).sum(dim=2).to(T.long)
    codon_eq_tot=T.zeros(codon_comb_index.shape, dtype=T.long,device=dna.device)
    for ind_val in amino_codon_list:
        ind_val=int(ind_val)
        codon_eq=codon_comb_index.eq(ind_val).to(T.long)
        codon_eq_tot=codon_eq_tot.add(codon_eq)
    codon_index=T.nonzero(codon_eq_tot)
    codon_nums=codon_eq_tot.sum(dim=1)
    codon_max=int(codon_nums.max())
    codon_ind=T.nonzero(util_funcs.build_mult_tensor(codon_nums))
    codon_return=T.zeros([dna.shape[0],codon_max],dtype=T.long,device=dna.device)
    #add codon len to skip the zero codon
    codon_return[codon_ind.split(1,dim=1)]=codon_index[:,1].unsqueeze(dim=1).add(codon_len)
    return codon_return, codon_nums


def calc_dna_seq_homo(dna, dna_seq_to_match):
    #inputs a dna base sequence.

    #returns all indexs in the dna sequence and the percent match to the input.
    match_seq_len=dna_seq_to_match.shape[0]
    dna_seq_combo=build_dna_combo_seq(dna, match_seq_len)
    dna_match_per=dna_seq_combo.eq(dna_seq_to_match).sum(dim=1).div(match_seq_len)
    return dna_match_per


def find_dna_seq_homo_ge_perc(dna, dna_seq_to_match, ge_perc_val):
    #inputs a dna base sequence and a percent val (0-1).

    #returns all indexs in the dna where the base sequence match is greater than or equal
    #to the base sequence input.
    dna_match_perc=calc_dna_seq_homo(dna, dna_seq_to_match)
    inds_ge_val=T.nonzero(dna_match_perc.ge(ge_perc_val)).flatten().add(dna_seq_to_match.shape[0])
    return inds_ge_val


def build_protien_amino_sequence(dna, start_amino_vals, stop_amino_vals, vec_to_ind, codon_to_amino, codon_len, perc_match=1):
    #inputs a 1D tensor of dna bases, the amino val that commands to start building an amino chain, the amino val that commands to stop
    #building an amino chain, a multiplyer that helps compute a base sequence to its codon val, a tensor that gives the amino val for each
    #codon, the number of dna bases a codon has (aka codon length), and finally an optional percent match (range 0-1) for bases sequences
    #close to but not an exact match for the start codon command.

    #returns a 2D tensor where the first dimenson is the number of protiens built by the given inputs and the second dimenson is the amino
    #sequence of those protiens. It also returns a 1D tensor that contains the number of aminos in each protien.
    if type(start_amino_vals)==type(dna):
        start_amino_ind=find_dna_seq_homo_ge_perc(dna, start_amino_vals, perc_match)
    else:
        start_amino_ind=find_amino_index(dna,vec_to_ind,start_amino_vals,codon_to_amino, codon_len)
    stop_amino_ind=find_amino_index(dna,vec_to_ind,stop_amino_vals,codon_to_amino, codon_len)
    codon_seq,num_codons=build_protein_codon_sequence(dna, start_amino_ind, stop_amino_ind, vec_to_ind, codon_len)
    codon_to_amino_adj=codon_to_amino.clone().detach()
    added_val=T.full([1],-1,dtype=codon_to_amino_adj.dtype,device=codon_to_amino_adj.device)
    codon_to_amino_adj=T.cat([codon_to_amino_adj,added_val],dim=0)
    codon_seq_len=codon_seq.shape[1]
    codon_flat=T.cat(codon_seq.split(1),dim=1).flatten()
    amino_seq_vals=codon_to_amino_adj[codon_flat]
    amino_seq_vals=T.cat(amino_seq_vals.unsqueeze(dim=0).split(codon_seq_len,dim=1),dim=0)
    return amino_seq_vals, num_codons , codon_seq


def remove_aminos_from_sequence(amino_seq, amino_num_per_prot, amino_ind_to_remove_list):
    #inputs a 2D tensor of protiens with amino val sequences on the 2nd D, the number of aminos per protien and a list of amino vals to remove
    #from the 2D.

    #returns the 2D protien tensor with all instances of the amino vals given removed and how many aminos are in each protien after removal.
    amino_to_remove=T.zeros(amino_seq.shape,dtype=T.int8,device=amino_seq.device)
    for amino_ind in amino_ind_to_remove_list:
        num_eq_per_prot=amino_seq.eq(amino_ind)
        amino_to_remove=amino_to_remove.add(num_eq_per_prot)
    num_removed_per_prot=amino_to_remove.sum(dim=1).to(amino_num_per_prot.dtype)
    num_amino_sub_eq=amino_num_per_prot.add(-num_removed_per_prot)
    max_amino=int(num_amino_sub_eq.max())
    amino_to_copy=amino_to_remove.clone().detach()
    amino_to_copy=amino_to_copy.add(amino_seq.eq(-1))
    nonzero_old_inds=T.nonzero(amino_to_copy.add(-1))
    old_inds_vals=amino_seq[nonzero_old_inds.split(1,dim=1)]
    new_seq_tensor=T.full([amino_seq.shape[0],max_amino],-1,dtype=amino_seq.dtype,device=amino_seq.device)
    new_mult_tensor=util_funcs.build_mult_tensor(num_amino_sub_eq)
    new_inds=T.nonzero(new_mult_tensor)
    new_seq_tensor[new_inds.split(1,dim=1)]=old_inds_vals
    return new_seq_tensor, num_amino_sub_eq












'''Prot shape funcs'''
'''NOT UPDATED AND USED CURRENTLY'''
'''
def protein_shape_calc(passed_prot_vec, passed_am_code_vals, amino_dims=3):
    prot_added_vec=T.zeros([amino_dims],dtype=passed_am_code_vals.dtype,device=comp_dev)
    prot_type=T.ones([1],dtype=passed_am_code_vals.dtype,device=comp_dev)
    prot_shape=T.zeros([amino_dims],dtype=T.long,device=comp_dev).unsqueeze(dim=0)
    for i, val in enumerate(passed_prot_vec.add(-2)):
        am_vals=passed_am_code_vals[val]
        am_type=am_vals[0]
        am_type_mult_prot=am_type.mul(prot_type)
        am_added=am_vals[1:].mul(am_type_mult_prot)
        prot_added_vec=prot_added_vec.add(am_added)
        prot_shape=T.cat([prot_shape, prot_added_vec.unsqueeze(dim=0).clone().detach().to(prot_shape.dtype)],dim=0)
        prot_type=am_type_mult_prot
    prot_dim_min=T.tensor([int(dim.min()) for dim in prot_shape.split(1,dim=1)],dtype=T.long,device=comp_dev)
    prot_shape=prot_shape.add(-prot_dim_min)
    return prot_shape, prot_added_vec


def find_protein_shapes(passed_prot_vecs, passed_num_aminos, passed_am_code_vals):
    prot_shapes_list=[]
    prot_added_vec_list=[]
    for j, vec in enumerate(passed_prot_vecs):
        prot_vec=passed_prot_vecs[j,:passed_num_aminos[j]]
        prot_shape, prot_added_vec=protein_shape_calc(prot_vec, passed_am_code_vals)
        prot_shapes_list.append(prot_shape)
        prot_added_vec_list.append(prot_added_vec)
    prot_added_vecs=T.cat([vecs.unsqueeze(dim=0) for vecs in prot_added_vec_list],dim=0)
    return prot_shapes_list, prot_added_vecs
'''

'''

prot_amino_chains[:10,:15]

tensor([[11,  3,  7, 13, 11,  4,  3,  6,  7,  4, 10,  6, 13,  3,  4],
        [ 3,  4,  4,  8,  4, 12,  6,  5, 10,  2,  6,  4, 11,  4, 13],
        [11,  6,  9,  6,  8,  6,  8,  3,  3,  5,  2, 12, 12,  9,  9],
        [ 3, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 6,  7,  3, 11,  5, 11,  6,  4,  7, 11, 12,  2,  6,  8,  3],
        [ 4,  5, 13, 13,  2,  3,  5,  4,  9,  3, 13,  7, 13,  3,  4],
        [ 8,  9, 11,  3, 11,  8, 13,  3,  4,  8,  2,  9,  2, 11, 11],
        [ 4,  2, 13,  6,  7,  6, 13, 11,  4,  3,  3, 11, 13, 11,  6],
        [ 3,  4,  3, 13, 11,  6,  6,  2,  4, 11,  4, 11,  5,  2,  3],
        [ 3,  9,  7,  5,  5,  2,  4,  8, 10, 12, 11,  5,  3,  3,  6]],
       device='cuda:0')


def build_protein_amino_chain(dna, index_vals, codon_amino, amino_max, vec_index, codon_len):
    index_start=index_vals[0]
    index_end=index_vals[1]
    parsed_dna=dna[index_start:index_end]
    if parsed_dna.shape[0]%codon_len!=0:
        print('   parsed_dna%codon_len!=0 for protein starting at dna index:',index_start)
    codon_vec=T.cat([codon.unsqueeze(dim=0) for codon in parsed_dna.split(codon_len)],dim=0)
    codon_index=codon_vec.mul(vec_index).sum(dim=1).to(T.long)
    amino_vals=codon_amino[codon_index]
    #aminos zero and one val is a non coding amino so we are taking them out
    amino_coding_vals_ind=T.nonzero(amino_vals.gt(1)).flatten()
    amino_vals=amino_vals[amino_coding_vals_ind]
    prot_vec=T.zeros([amino_vals.shape[0]],dtype=T.long,device=comp_dev)
    prot_vec[:]=amino_vals
    return prot_vec


def build_protein_amino_chains(dna, codon_index_vals, codon_amino_vals, num_of_aminos, vec_to_ind_mult, codon_len):
    prot_num_of_codons=codon_index_vals[:,1].add(-codon_index_vals[:,0]).true_divide(codon_len).to(T.long)
    max_codon_dif=int(prot_num_of_codons.max())
    protein_vecs=T.zeros([codon_index_vals.shape[0], max_codon_dif],dtype=T.long, device=comp_dev)
    for i, ind_val in enumerate(codon_index_vals):
        try:
            prot_vec=build_protein_amino_chain(dna, ind_val, codon_amino_vals, num_of_aminos, vec_to_ind_mult, codon_len)
            protein_vecs[i,:prot_vec.shape[0]]=prot_vec
        except RuntimeError:
            print(prot_num_of_codons[i])
            print(codon_index_vals[i])
            print(prot_vec[:25])
            print(prot_vec[500:])
            print(prot_vec.shape)
            print(protein_vecs.shape)
            raise ValueError("Boken here")

    return protein_vecs, prot_num_of_codons



def find_protein_stop_amino(dna, index_start, codon_amino, vec_index, codon_len, codon_batch_size=100):
    index_batch_size=codon_batch_size*codon_len
    dna_parsed=dna[index_start:]
    dna_parsed_range=dna_parsed.shape[0]
    parsed_to_batch=int(math.floor(dna_parsed_range/index_batch_size))
    added_prot=True
    if dna_parsed_range<codon_len:
        #print('\nFor index_start:',index_start)
        #print('Less than three bases left. No amino chain can be constructed, so no protein vec returned.')
        prot_index_vals=T.zeros([2], dtype=T.long, device=comp_dev)
        added_prot=False
        return prot_index_vals, added_prot
    batch_size_list=[index_batch_size for i in range(parsed_to_batch)]
    dna_parsed_end=(dna_parsed_range-int(parsed_to_batch*index_batch_size))
    dna_parsed_end-=(dna_parsed_end%codon_len)
    if dna_parsed_end >=codon_len:
        batch_size_list.append(dna_parsed_end)
    cur_parsed_index=0
    stop_amino=False
    stop_amino_index=0
    for val in batch_size_list:
        end_index=val+cur_parsed_index
        codon_vec=T.cat([codon.unsqueeze(dim=0) for codon in dna_parsed[cur_parsed_index:end_index].split(codon_len)],dim=0)
        codon_index=codon_vec.mul(vec_index).sum(dim=1).to(T.long)
        amino_vals=codon_amino[codon_index]
        stop_amino_index_vals=T.nonzero(amino_vals.eq(1)).flatten()
        if stop_amino_index_vals.shape[0]!=0:
            stop_amino_index=(int(index_start)+int((codon_len*(stop_amino_index_vals.min()+1))+cur_parsed_index))
            stop_amino=True
            break
        cur_parsed_index+=val
    if stop_amino==False:
        stop_amino_index=(int(index_start)+cur_parsed_index)
        #print('No stop amino for protein starting at dna index:',index_start)

    prot_index_vals=T.zeros([2], dtype=T.long, device=comp_dev)
    prot_index_vals[0]=index_start
    prot_index_vals[1]=stop_amino_index
    return prot_index_vals, added_prot


def build_protein_base_sequence(dna, start_amino_inds, codon_amino_vals, vec_to_ind_mult, codon_len):
    protein_index_list=[]
    for i, start_amino_ind in enumerate(start_amino_inds):
        protein_index, added_prot_vec=find_protein_stop_amino(dna, start_amino_ind, codon_amino_vals, vec_to_ind_mult, codon_len)
        if added_prot_vec:
            protein_index_list.append(protein_index)
    dna_prot_index=T.cat([index.unsqueeze(dim=0) for index in protein_index_list],dim=0).to(T.long)
    return dna_prot_index


import torch as T

import os
import multiprocessing as MP
import time
#from model_env import Model_env
#from pynput import keyboard
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

#from protein_funcs import Protein_funcs
import prot_funcs
from dna_child import Dna_child
from genome_obj import Genome
import mp_funcs


comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')

#prot_funcs=Protein_funcs()
dna_gen_builder=Dna_child()

codon_len=3
num_of_bases=4
num_of_codons=(num_of_bases**codon_len)
dna_codon_len_mult=25000
min_perc_val=50

amino_dims=3
am_comb_init=T.eye(amino_dims,device=comp_dev)
am_comb_pos=T.cat([am_comb_init, am_comb_init.mul(-1)],dim=0)
am_mult_type_pos=T.ones([am_comb_pos.shape[0]],dtype=am_comb_pos.dtype,device=comp_dev)
am_mult_type=T.cat([am_mult_type_pos, am_mult_type_pos.mul(-1)],dim=0)
am_comb=T.cat([am_comb_pos,am_comb_pos],dim=0)
am_coding_vals=T.cat([am_mult_type.unsqueeze(dim=1), am_comb],dim=1)
num_of_aminos=am_comb.shape[0]+2


start_amino_val=0
stop_amino_val=1
num_of_stop_codons=1
codon_amino_vals, num_of_coding_aminos=prot_funcs.assign_amino_to_codon(num_of_codons, num_of_aminos, num_of_stop_codons, start_amino_val, stop_amino_val)

sum_val=0
for i in range(num_of_aminos):
    eq_vals=int(codon_amino_vals.eq(i).sum())
    sum_val+=eq_vals
    if eq_vals>1:
        print('  {0} codons for amino {1}.'.format(eq_vals,i))
    else:
        print('  {0} codon for amino {1}.'.format(eq_vals,i))
print('\nTotal sum of codon vals={0}'.format(sum_val))


vec_to_ind_mult=T.tensor([(num_of_bases**2),num_of_bases,1], device=comp_dev)




saved_dna_str='DNA_test_gen'

gen=T.load(saved_dna_str)


dna=gen.dna_list[0]


start_amino_ind=prot_funcs.find_amino_index(dna,vec_to_ind_mult,start_amino_val,codon_amino_vals)
stop_amino_ind=prot_funcs.find_amino_index(dna,vec_to_ind_mult,stop_amino_val,codon_amino_vals)
#dna_prot_index,dna_prot_num=prot_funcs.build_protein_base_sequence(dna, start_amino_ind, stop_amino_ind, codon_len)
prot_codon_chains,prot_num_of_codons=prot_funcs.build_protein_codon_sequence(dna, start_amino_ind, stop_amino_ind, vec_to_ind_mult, codon_len)

prot_shapes_list,prot_added_vecs=prot_funcs.find_protein_shapes(prot_amino_chains, prot_num_of_codons, am_coding_vals.clone().detach())


amino_11_ind=prot_funcs.find_amino_index(dna,vec_to_ind_mult,11,codon_amino_vals)
for i, val in enumerate(amino_11_ind[:50]):
    start=int(val-codon_len)
    end=int(val)
    codon=dna[start:end]
    ind=int(codon.mul(vec_to_ind_mult).sum())
    print('\nCodon = {0}'.format(codon.tolist()))
    print('Amino val = {0}'.format(int(codon_amino_vals[ind])))








def create_new_dna_list(codon_len, num_of_bases, dna_codon_len_mult, min_perc_val):
    num_of_dna=int(input('Enter a DNA pop number divisible by four please.\n'))
    while num_of_dna%4!=0:
        num_of_dna=int(input('Please enter a number divisable by four. \n'))
    dna_list=[prot_funcs.build_dna_base_sequence(codon_len, num_of_bases, dna_codon_len_mult, min_perc_val) for i in range(num_of_dna)]
    return dna_list, num_of_dna


def create_new_genome(codon_len, num_of_bases, dna_codon_len_mult, min_perc_val):
    dna_list,num_of_dna=create_new_dna_list(codon_len, num_of_bases, dna_codon_len_mult, min_perc_val)
    genome=Genome()
    init_mut_rates=[0.015,0.015,0.015]
    print('Initial mutation rates for each mutatuion type is:')
    print('Type order: \'substitution\',\'deletion\',\'insertion\'')
    print('        ',init_mut_rates)
    change_vals=input('Would you like to keep those vals?\nPlease enter Yes (Y) or No (N)').upper()
    if change_vals=='NO' or change_vals=='N':
        new_vals=input('Ok, please enter your numbers seperated by only a comma.\nLike this: 1,2,3\n')
        new_vals=new_vals.split(',')
        new_mut_list=[float(str_) for str_ in new_vals]
        init_mut_rates=new_mut_list
        print('Mutation rates changed to:')
        print('    ',init_mut_rates)
    else:
        print('Mutation rates for genome are:')
        print('    ',init_mut_rates)

    genome.load_vals(dna_list, dna_mut_rates=init_mut_rates)
    return genome


#gen=create_new_genome(codon_len, num_of_bases, dna_codon_len_mult, min_perc_val)






#with compressed DNA
def build_protein_base_sequence(dna, codons, codon_amino_vals, vec_to_ind_mult, codon_len):
    protein_index_list=[]
    for i, val in enumerate(codons):
        protein_index, added_prot_vec=find_protein_stop_amino(dna, val, codon_amino_vals, vec_to_ind_mult, codon_len)
        if added_prot_vec:
            protein_index_list.append(protein_index)
    dna_prot_index=T.cat([index.unsqueeze(dim=0) for index in protein_index_list],dim=0).to(T.long)
    dna_comp=T.cat([dna[prot_ind[0]:prot_ind[1]] for prot_ind in protein_index_list],dim=0).to(T.long)
    prot_index_dif=dna_prot_index[:,1].add(-dna_prot_index[:,0])
    mult_tensor=T.tensor([i for i in range(prot_index_dif.shape[0])], device=comp_dev).add(1)
    mult_tensor=util_funcs.build_mult_tensor(mult_tensor)
    dif_mult=mult_tensor.mul(prot_index_dif.unsqueeze(dim=0))
    index_end=dif_mult.sum(dim=1)
    index_start=T.zeros([index_end.shape[0]],device=comp_dev)
    index_start[1:]=index_end[:-1]
    dna_comp_prot_index=T.zeros([index_start.shape[0],2], dtype=T.long, device=comp_dev)
    dna_comp_prot_index[:,0]=index_start
    dna_comp_prot_index[:,1]=index_end
    return dna_comp, dna_prot_index, dna_comp_prot_index

'''
