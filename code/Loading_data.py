import numpy as np
GENOME_mm10 = '/home/chenxiaoyang/silencer/data/genome/mm10/'
GENOME_hg19 = '/home/chenxiaoyang/silencer/data/Chromosomes/'
def load_genome(name = 'hg19'):
    if name == 'hg19':
        print('Loading whole genome sequence...')
        chrs = list(range(1, 23))
        chrs.extend(['X', 'Y','M'])
        keys = ['chr' + str(x) for x in chrs]
        sequences = dict()
        for i in range(25):
            fa = open('%s%s.fa' % (GENOME_hg19, keys[i]), 'r')
            sequence = fa.read().splitlines()[1:]
            fa.close()
            sequence = ''.join(sequence)
            sequences[keys[i]]= sequence
        length = 0
        for i in range(25): 
            length += len(sequences[keys[i]])
        print('Length of whole genome sequence:',length)
        return sequences
    if name =='mm10':
        print('Loading whole genome sequence...')
        chrs = list(range(1, 20))
        chrs.extend(['X', 'Y','M'])
        keys = ['chr' + str(x) for x in chrs]

        sequences = dict()
        for i in range(22):
            fa = open('%s%s.fa' % (GENOME_mm10, keys[i]), 'r')
            sequence = fa.read().splitlines()[1:]
            fa.close()
            sequence = ''.join(sequence)
            sequences[keys[i]]= sequence
        length = 0
        for i in range(22): 
            length += len(sequences[keys[i]])
        print('Length of whole genome sequence:',length)
        return sequences

#silencer = [name+id,chrid,start,end,seq]
def cropseq(indexes, l, stride):
    """generate chunked silencer sequence according to loaded index"""
    print('Generating silencer samples with length {} bps...'.format(l))
    silencers = list()
    i = 0
    for index in indexes:
        try:
            [sampleid, chrkey, startpos, endpos, _] = index
        except:
            [sampleid, chrkey, startpos, endpos] = index
        l_orig = endpos - startpos
        if l_orig < l:
            for shift in range(0, l - l_orig, stride):
                start = startpos - shift
                end = start + l
                seq, legal = checkseq(chrkey, start, end)
                if legal:
                    silencers.append([sampleid, chrkey, start, end])
        elif l_orig >= l:
            chunks_ = chunks(range(startpos, endpos), l, l - stride)
            for chunk in chunks_:
                start = chunk[0]
                end = chunk[-1] + 1
                if (end - start) == l:
                    seq, legal = checkseq(chrkey, start, end)
                    silencers.append([sampleid, chrkey, start, end])
                elif (end - start) < l:
                    break

    print('Data augmentation: from {} indexes to {} samples'.format(len(indexes), len(silencers)))
    return silencers
# chunks_ = chunks(range(startpos, endpos), l, l - stride)
def chunks(l, n, o):
    """Yield successive n-sized chunks with o-sized overlap from l."""
    return [l[i: i + n] for i in range(0, len(l), n-o)]
def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec