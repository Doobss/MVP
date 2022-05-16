import torch as T
import util_funcs

comp_dev = 'cpu'
if T.cuda.is_available():
    comp_dev = T.device('cuda:0')
    
    
    


    
'''DNA FUNCS'''
def build_dna_base_seqs(num_of_bases, base_min, base_max, num_of_seqs):
    #inputs: number of bases the seq has, min number of bases, max number of bases, number of 
    #sequences to create.
    
    #returns: 2D tensor where dim 1 is how many seq are returned and dim 2 contains the base seq and
    #a 1D tensor which contains the number of bases per seq.
    rand_num_of_base_vals=T.rand([int(num_of_seqs)] ,device=comp_dev).mul((base_max-base_min)).add(base_min).to(T.long)
    mult_tensor=util_funcs.build_mult_tensor(rand_num_of_base_vals)
    rand_base_vals=T.randint(0,num_of_bases,mult_tensor.shape,device=comp_dev).add(1).mul(mult_tensor).add(-1)
    if num_of_seqs==1:
        rand_base_vals=rand_base_vals.flatten()
    return rand_base_vals, rand_num_of_base_vals
    
    
    
'''
Old function
def build_dna_base_sequence(codon_len, num_of_bases, dna_max_codons, min_val):  
    #inputs the length of a codon for this dna sequence, the number of bases for this dna sequence, a max number of codons wanted and a min 
    #val
    
    #returns a 1D dna sequence with values from 0 to (NUM_OF_BASES-1) randomly placed.
    dna_codon_mult=T.randint(min_val, 101,[1],device=comp_dev).div(100).mul(dna_max_codons).to(T.long)
    dna=T.rand([(int(dna_codon_mult)*codon_len)],device=comp_dev).mul((num_of_bases-1)).round()
    end_stop_codon=T.zeros([codon_len*2], device=comp_dev)
    end_stop_codon[-1]=1
    dna=T.cat([dna, end_stop_codon],dim=0)
    return dna
'''
    