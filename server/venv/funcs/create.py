#******CURRENT WORKING TEST CODE******


#EXTERNAL IMPORTS
import torch as T
#import os
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt



#INTERNAL IMPORTS
import prot_funcs
import shape_funcs
import dna_funcs



comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')





def create():
    init_params=[['codon_len','INT',3], ['num_of_bases','INT',4], ['num_of_codons','INT',(4**3)],['dna_base_min','INT',None],['dna_base_max','INT',None],['amino_dims','INT',3], ['num_of_seq','INT',1]]
    codon_len=3
    num_of_bases=4
    num_of_codons=(num_of_bases**codon_len)
    dna_base_min=7500
    dna_base_max=15000
    amino_dims=3
    num_of_seq=1

    dna, dna_num_bases=dna_funcs.build_dna_base_seqs(num_of_bases, dna_base_min, dna_base_max, num_of_seq)


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

    vec_to_ind_mult=T.tensor([(num_of_bases**2),num_of_bases,1], device=comp_dev)
    init_gene_base_seq=T.tensor([0,0,0,],dtype=dna.dtype,device=dna.device)
    init_gene_match_perc=1
    prot_amino_chains, prot_num_of_aminos, codon_seq=prot_funcs.build_protien_amino_sequence(dna, init_gene_base_seq, stop_amino_val, vec_to_ind_mult, codon_amino_vals, codon_len)
    aminos_remove_list=[start_amino_val, stop_amino_val]
    prot_amino_chains,prot_num_of_aminos=prot_funcs.remove_aminos_from_sequence(prot_amino_chains, prot_num_of_aminos, aminos_remove_list)
    prot_coding_amino_chains=T.where(prot_amino_chains!=(-1), (prot_amino_chains-len(aminos_remove_list)), prot_amino_chains)
    prot_shapes_list, prot_added_vecs=shape_funcs.find_protein_shapes(prot_coding_amino_chains, am_coding_vals)
    #shape_funcs.print_prot_shapes(prot_shapes_list, prot_num_of_aminos)

    #return [prot_shapes_list, prot_coding_amino_chains.tolist()]
    return [prot_shapes_list, prot_coding_amino_chains.tolist(), dna.tolist(), codon_seq.tolist()]
