

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader import *
from utility import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
data_folder = '/u/training/tra371/scratch/preprocess'  # folder with data files saved by create_input_files.py

data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = '/u/training/tra371/scratch/project3/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = '/u/training/tra371/scratch/preprocess/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)
end_word = torch.tensor([word_map['<end>']]).to(device)

# Normalization transform
data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def evaluate(beam_size):
    
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=data_transform),
        batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    
    references = list()
    hypotheses = list()
    embed_dim = 512
    
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
              

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 224, 224)
        #print(image.shape)
        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)

        
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

       
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
       
       
        seqs = k_prev_words  # (k, 1)

        
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        decoder.reset_state()
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size,-1)
       
        encoder_out = decoder.img_embedding(encoder_out)

        h = decoder.lstm(encoder_out)
        
        h = decoder.bn_lstm(h)
        h = decoder.dropout(h, dropout=0.3, train=False)

       
        
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            h = decoder.lstm(embeddings)  # (s, decoder_dim)
            h = decoder.bn_lstm(h)
            h = decoder.dropout(h, dropout=0.3, train=False)
            scores = decoder.decoder(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
           
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
           
       
            if step > 50:
                next_word_inds = torch.cat([next_word_inds[:-1], end_word], dim=0).to(device)

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            decoder.lstm.h = decoder.lstm.h[incomplete_inds]
            decoder.lstm.c = decoder.lstm.c[incomplete_inds]
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)
        x = np.asarray(references)
        np.save("original_captions",x)
        if(i==0):
            saved_image = np.asarray(image.cpu())
            saved_captions = np.asarray(seq)
            saved_actual_captions = np.asarray(img_captions)
            np.save("actual_captions",saved_actual_captions)
            np.save("seq",saved_captions)
            np.save("image",saved_image)
        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        y = np.asarray(hypotheses)
        np.save("generated_captions_beamsize_{}".format(beam_size),y)
        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses) #emulate_multibleu=True)

    return bleu4


if __name__ == '__main__':
    beam_size = 2
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))








