import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    v_size = len(vocab)
    emb_vectors = {}
    emb = np.zeros((v_size,emb_size),dtype=np.float64)
    unknown_words = []

    with open(emb_file, 'r') as f:
        for line in f:
            ln = line.split()
            val = np.array(ln[1:], dtype=np.float64)
            emb_vectors[ln[0]] = val
    
    for word in vocab.word2id:
        id = vocab[word]
        if word in emb_vectors:
            emb[id][:] = emb_vectors[word]
        else:
            if word is '<pad>' or word is '<unk>':
                continue
            unknown_words.append(word)
            print("Embedding does not exist for word = ",word)

    unk_id = vocab['<unk>']
    emb[unk_id][:] = np.mean(emb,axis=0)
    
    for word in unknown_words:
        id = vocab[word]
        emb[id][:] = emb[unk_id]

    return emb
    #raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.n_vocab = len(vocab)
        self.n_embed = args.emb_size
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embed)
        self.fc1 = nn.Linear(self.n_embed, self.n_embed)
        self.z1 = nn.ReLU()
        self.fc2 = nn.Linear(self.n_embed, self.n_embed)
        self.z2 = nn.ReLU()
        return
        #raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.1)
        return
        #raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb).float())
        return
        #raise NotImplementedError()

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        x = self.embedding(x)
        #(batch_size,length_of_sentence,embedding_dim)
        x = torch.mean(x,1)
        #(batch_size,embedding_dim)
        x = self.fc1(x)
        x = self.z1(x)
        x = self.fc2(x)
        x = self.z2(x)
        return x


        raise NotImplementedError()
