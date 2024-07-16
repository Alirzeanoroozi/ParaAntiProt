import os
import torch
from tqdm import tqdm
import pandas as pd
import pickle


def create_embeddings(config):
    df = None
    if config['dataset'] == 'nano':
        df = (pd.read_csv("data/NanoBERTa/nanotrain.csv")['sequence'].tolist() +
              pd.read_csv("data/NanoBERTa/nanoval.csv")['sequence'].tolist() +
              pd.read_csv("data/NanoBERTa/nanotest.csv")['sequence'].tolist())
    elif config['dataset'] == 'parapred':
        df = (pd.read_csv("data/cdrs_parapred.csv")['sequence'].to_list() +
              pd.read_csv("data/chains_parapred.csv")['sequence'].to_list())
    elif config['dataset'] == 'paragraph':
        df = pd.read_csv("data/cdrs_test.csv")

    if config['embedding'][0] == "berty":
        from igfold import IgFoldRunner
        igfold = IgFoldRunner()

        for i, seq in tqdm(enumerate(df)):
            file_name = "embeddings/embeddings/" + seq + ".pt"
            while not os.path.isfile(file_name):
                torch.save(igfold.embed(sequences={"H": seq}).bert_embs, file_name)
                print(i + 1, 'new')

        embedding_dict = {}
        for i, seq in tqdm(enumerate(df)):
            if seq not in embedding_dict:
                print(seq, 'new')
                embedding_dict[seq] = torch.load("embeddings/embeddings/" + seq + ".pt").squeeze(0)
        with open("embeddings/embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)

    # elif config['embedding'][0] == "ab":
    #     import ablang
    #
    #     heavy_ablang = ablang.pretrained("heavy")
    #     heavy_ablang.freeze()
    #
    #     light_ablang = ablang.pretrained("light")
    #     light_ablang.freeze()
    #
    #     for i, seq in tqdm(enumerate(df)):
    #         file_name = "ab_embeddings/" + seq + ".pt"
    #         while not os.path.isfile(file_name):
    #             # if cdr_df['cdr_type'].tolist()[i][2] == 'H':
    #             torch.save(heavy_ablang([seq], mode='rescoding'), file_name)
    #             # else:
    #             #     torch.save(light_ablang([seq], mode='rescoding'), file_name)
    #             print(i + 1, 'new')
    #
    #     # for i, seq in tqdm(enumerate(chain_df['sequence'].to_list())):
    #     #     file_name = "ab_embeddings/" + seq + ".pt"
    #     #     while not os.path.isfile(file_name):
    #     #         if chain_df['cdr_type'].tolist()[i][2] == 'H':
    #     #             torch.save(heavy_ablang([seq], mode='rescoding'), file_name)
    #     #         else:
    #     #             torch.save(light_ablang([seq], mode='rescoding'), file_name)
    #     #         print(i + 1, 'new')

    elif config['embedding'][0] == "ab":
        embedding_dict = {}
        for i, seq in tqdm(enumerate(df)):
            if seq not in embedding_dict:
                embedding_dict[seq] = torch.load("embeddings/ab_embeddings/" + seq + ".pt")[0]
                print(seq, 'new', len(seq), len(embedding_dict[seq]))
        with open("embeddings/ab_embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)

    # elif config['embedding'][0] == "prot_folder":
    #     from transformers import T5Tokenizer, T5EncoderModel
    #
    #     tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    #     model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    #
    #     for i, seq in tqdm(enumerate(cdr_df['sequence'].to_list() + chain_df['sequence'].to_list())):
    #         file_name = "prot_embeddings/" + seq + ".pt"
    #         while not os.path.isfile(file_name):
    #             ids = tokenizer([" ".join(list(seq))], return_tensors="pt")
    #             embedding_repr = model(input_ids=ids['input_ids'], attention_mask=ids['attention_mask'])
    #             torch.save(embedding_repr.last_hidden_state[0, :-1], file_name)
    #             print(i + 1, 'new')
    elif config['embedding'][0] == "prot":
        embedding_dict = {}
        for i, seq in tqdm(enumerate(pd.read_csv("data/cdrs_parapred.csv")['sequence'].to_list())):
            if seq not in embedding_dict:
                embedding_dict[seq] = torch.load("embeddings/prot_embeddings/" + seq + ".pt")
                print(seq, 'new', len(seq), len(embedding_dict[seq]))
        with open("embeddings/prot_embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)
    # elif config['embedding'][0] == "balm_folder":
    #     from BALM.modeling_balm import BALMForMaskedLM
    #     from transformers import EsmTokenizer
    #     import torch
    #
    #     tokenizer = EsmTokenizer.from_pretrained("BALM/tokenizer/vocab.txt", do_lower_case=False)
    #     model = BALMForMaskedLM.from_pretrained("./BALM/pretrained-BALM/")
    #
    #     # # final hidden layer representation [batch_sz * max_length * hidden_size]
    #     # final_hidden_layer = outputs.hidden_states[-1]
    #     # # final hidden layer sequence representation [batch_sz * hidden_size]
    #     # final_seq_embedding = final_hidden_layer[:, 0, :]
    #     # # final layer attention map [batch_sz * num_head * max_length * max_length]
    #     # final_attention_map = outputs.attentions[-1]
    #
    #     # for i, seq in tqdm(enumerate(cdr_df['sequence'].to_list() + chain_df['sequence'].to_list())):
    #     # for i, seq in tqdm(enumerate(cdr_df['sequence'].to_list())):
    #     for i, seq in tqdm(enumerate(nanoberta)):
    #         file_name = "balm_embeddings/" + seq + ".pt"
    #         while not os.path.isfile(file_name):
    #             tokenizer_input = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    #             outputs = model(**tokenizer_input, return_dict=True, output_hidden_states=True, output_attentions=True)
    #             torch.save(outputs.hidden_states[-1].squeeze(0), file_name)
    elif config['embedding'][0] == "balm":
        embedding_dict = {}
        for i, seq in tqdm(enumerate(df)):
            if seq not in embedding_dict:
                embedding_dict[seq] = torch.load("embeddings/balm_embeddings/" + seq + ".pt")
                print(seq, 'new', len(seq), len(embedding_dict[seq]))
        with open("embeddings/balm_embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)
    # elif config['embedding'][0] == "esm_folder":
    #     import esm
    #
    #     model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    #     batch_converter = alphabet.get_batch_converter()
    #     model.eval()  # disables dropout for deterministic results
    #
    #     for i, seq in tqdm(enumerate(cdr_df['sequence'].to_list() + chain_df['sequence'].to_list())):
    #         file_name = "esm_embeddings/" + seq + ".pt"
    #         while not os.path.isfile(file_name):
    #             batch_labels, batch_strs, batch_tokens = batch_converter([("", seq)])
    #
    #             with torch.no_grad():
    #                 results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    #             token_representations = results["representations"]
    #             print(token_representations.shape)
    #             exit()
    #             torch.save(token_representations[0, 1: -1], file_name)
    elif config['embedding'][0] == "esm":
        embedding_dict = {}
        for i, seq in tqdm(enumerate(pd.read_csv("data/cdrs_parapred.csv")['sequence'].to_list())):
            if seq not in embedding_dict:
                embedding_dict[seq] = torch.load("embeddings/esm_embeddings/" + seq + ".pt")
                print(seq, 'new', len(seq), len(embedding_dict[seq]))
        with open("embeddings/esm_embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)
    # elif config['embedding'][0] == "ig_folder":
    #     from transformers import BertModel, BertTokenizer
    #
    #     tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert_unpaired", do_lower_case=False)
    #     model = BertModel.from_pretrained("Exscientia/IgBert_unpaired", add_pooling_layer=False)
    #
    #     # for i, seq in tqdm(enumerate(cdr_df['sequence'].to_list() + chain_df['sequence'].to_list())):
    #     for i, seq in tqdm(enumerate(nanoberta)):
    #         file_name = "ig_embeddings/" + seq + ".pt"
    #         while not os.path.isfile(file_name):
    #             sequences = [' '.join(c) for c in seq]
    #             tokens = tokeniser.batch_encode_plus(
    #                 sequences,
    #                 add_special_tokens=True,
    #                 pad_to_max_length=True,
    #                 return_tensors="pt",
    #                 return_special_tokens_mask=True
    #             )
    #             output = model(
    #                 input_ids=tokens['input_ids'],
    #                 attention_mask=tokens['attention_mask']
    #             )
    #             torch.save(output.last_hidden_state, file_name)
    elif config['embedding'][0] == "ig":
        embedding_dict = {}
        for i, seq in tqdm(enumerate(df)):
            if seq not in embedding_dict:
                embedding_dict[seq] = torch.load("embeddings/ig_embeddings/" + seq + ".pt")
                print(seq, 'new', len(seq), len(embedding_dict[seq]))
        with open("embeddings/ig_embeddings.p", "wb") as f:
            pickle.dump(embedding_dict, f)
