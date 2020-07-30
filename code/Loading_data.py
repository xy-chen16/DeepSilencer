import numpy as np
import os
INDEX1 = '/home/gaozijing/gzj/DeepSilencer/data/bed/Candidate_silencers_and_uncharacterized_CREs_human_hg19_ENCODE_cell_types.txt'
INDEX2 = '/home/gaozijing/gzj/DeepSilencer/data/bed/Candidate_silencers_and_uncharacterized_CREs_human_hg19_roadmap_cell_types.txt'
INDEX3 = '/home/gaozijing/gzj/DeepSilencer/data/bed/Candidate_silencers_and_uncharacterized_CREs_mouse_mm10_ENCODE_cell_types.txt'

GENOME_mm10 = '/home/chenxiaoyang/data/mm10/'
GENOME_hg19 = '/home/gaozijing/gzj/DeepSilencer/data/Chromosomes/'

name2index = {'h_hg19_ENCODE': INDEX1,'h_hg19_roadmap':INDEX2,'m_mm19_ENCODE': INDEX3}
encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
acgt2num = {'A': 0,'C': 1,'G': 2,'T': 3}
complement = {'A': 'T','C': 'G','G': 'C','T': 'A'}
num2acgt = {0:'A',1:'C',2:'G',3:'T'}

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

def checkseq(chrkey, start, end,sequences):
    sequence = sequences[chrkey][start:end]
    legal = ('n' not in sequence) and ('N' not in sequence)
    return sequence, legal

def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=bool)  # True or false in mat
    for i in range(w):
        mat[acgt2num[seq[i]], i] = 1.
    return mat.reshape((1, -1))

def loadindex(sequences,name='ubiquitous'):
    """Load silencers indexes (id, chr, start, end)"""
    print('Loading %s silencer indexes...' % name)
    if name == 'ubiquitous':
        if os.path.isfile('data/temp/ubiquitous_index.hkl'):
            print ('Find corresponding hkl file')
            indexes = hkl.load('data/temp/ubiquitous_index.hkl')
            return indexes
        fr = gzip.open(INDEX, 'r')
        entries = fr.readlines()
        fr.close()
        n = len(entries)
        indexes = list()
        for i, entry in enumerate(entries):
            chrkey, start, end = entry.split('\t')[:3]
            start = int(start) - 1
            end = int(end) - 1
            seq, legal = checkseq(chrkey, start, end,sequences)
            if legal and len(entry.split('\t'))>4:
                indexes.append(['ubiquitous%05d' % i, chrkey, start, end])
        print('Totally {0} silencers in {1}'.format(n, INDEX))
        hkl.dump(indexes, 'data/temp/ubiquitous_index.hkl', 'w')
    else:
        if os.path.isfile('data/temp/specific/%s_index.hkl' % name):
            print('Find corresponding hkl file')
            indexes = hkl.load('temp/specific/%s_index.hkl' % name)
            return indexes
        f = open(name2index[name], 'r')
        entries = f.read().splitlines()
        f.close()
        n = len(entries)
        indexes = list()
        for i, entry in enumerate(entries):
            chrkey, start, end = entry.split('\t')[:3]
            start = int(start) -1
            end = int(end) - 1
            seq, legal = checkseq(chrkey, start, end,sequences)
            if legal and len(entry.split('\t'))>4:
                indexes.append(['%s%05d' % (name, i), chrkey, start, end,entry.split('\t')[4]])
        print('Totally {0} silencers in {1}'.format(n, name2index[name]))
    return indexes