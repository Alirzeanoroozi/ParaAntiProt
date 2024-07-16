import sys
import urllib.request
from lxml import html
import requests
import os
import Bio
import pandas as pd
from Bio.PDB import *
from Bio.PDB.Model import Model
import pickle


def extract_number_and_letter(residue):
    if residue.isdigit():
        return int(residue), ''
    else:
        return int(residue[:-1]), residue[-1]


def load_chains():
    with open("downloaded_seqs.p", "rb") as f:
        sequences = pickle.load(f)

    for _, entry in pd.read_csv("test_set.csv").iterrows():
        pdb_name = entry['pdb']
        ab_h_chain = entry['Hchain'].upper()
        ab_l_chain = entry['Lchain'].upper()
        ag_chain = entry['antigen_chain']

        structure = PDBParser().get_structure("", "pdb/{0}.pdb".format(pdb_name))
        model = structure[0]  # Structure only has one model

        if ";" in ag_chain:  # Several chains
            chain_ids = ag_chain.replace('"', "").split(";")
            ag_atoms = [a for c in chain_ids for a in Selection.unfold_entities(model[c.strip()], 'A')]
        else:  # 1 chain
            ag_atoms = Selection.unfold_entities(model[ag_chain], 'A')

        ag_search = Bio.PDB.NeighborSearch(ag_atoms)

        yield ag_search, model[ab_h_chain], model[ab_l_chain], sequences[pdb_name][1], (pdb_name, ab_h_chain, ab_l_chain)


def process_chains(ag_search, ab_h_chain, ab_l_chain, sequences, pdb):
    chain_mats = {}
    cont_mats = {}

    chains, contact = get_chains_contact_info(ag_search, ab_h_chain, ab_l_chain, sequences, pdb)

    for chain_name in ["H", "L"]:
        chain_mats[chain_name] = "".join([r[0] for r in chains[chain_name]])
        cont_mats[chain_name] = contact[chain_name]

    index_mats = {
        "H": extract_cdrs(sequences[pdb[1]], "H"),
        "L": extract_cdrs(sequences[pdb[2]], "L")
    }

    return chain_mats, cont_mats, index_mats


def get_chains_contact_info(ag_search, ab_h_chain, ab_l_chain, sequences, pdb):
    chains = {}
    chains.update(extract_chains(ab_h_chain, sequences[pdb[1]], "H"))
    chains.update(extract_chains(ab_l_chain, sequences[pdb[2]], "L"))

    contact = {}
    for name, chain in chains.items():
        contact[name] = "".join(['0' if (res[1] is None) or (not residue_in_contact_with(res[1], ag_search)) else '1' for res in chain])

    return chains, contact


def residue_in_contact_with(res, c_search, dist=4.5):
    return any(len(c_search.search(a.coord, dist)) > 0 for a in res.get_unpacked_list())


def extract_chains(chain, sequence, chain_type):
    chain_dict = {}
    pdb_residues = chain.get_unpacked_list()
    seq_residues = sorted(sequence)

    for res_id in seq_residues:
        pdb_res = find_pdb_residue(pdb_residues, res_id)
        chain_seq = chain_dict.get(chain_type, [])
        chain_seq.append((sequence[res_id], pdb_res, res_id))
        chain_dict[chain_type] = chain_seq
    return chain_dict


def find_pdb_residue(pdb_residues, residue_id):
    for pdb_res in pdb_residues:
        if (pdb_res.id[1], pdb_res.id[2].strip()) == residue_id:
            return pdb_res
    return None


def extract_cdrs(sequence, chain_type):
    cdrs_indexes = []
    for i, res_id in enumerate(sorted(sequence)):
        if residue_in_cdr(res_id, chain_type):
            cdrs_indexes.append(i)
    return cdrs_indexes


def residue_in_cdr(res_id, chain_type):
    num_extra_residues = 2  # The number of extra residues to include on the either side of a CDR
    chothia_cdr_def = {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
                       "H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}
    cdr_names = [chain_type + str(e) for e in [1, 2, 3]]  # L1-3 or H1-3

    for cdr_name in cdr_names:
        cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
        range_low, range_hi = -num_extra_residues + cdr_low, cdr_hi + num_extra_residues
        if range_low <= res_id[0] <= range_hi:
            return True
    return False


def download_annotated_seq(pdb, h_chain, l_chain):
    h_chain = h_chain.capitalize()
    l_chain = l_chain.capitalize()

    page = requests.get('https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/structureviewer/?pdb=' + pdb)

    tree = html.fromstring(page.content)

    fv_info = tree.xpath("//div[@id='chains']")

    chains_info_div_id = fv_info[0].xpath(".//div[@class='accordion-group']/a[div[contains(., '{}/{}')]]/@href"
                                          .format(h_chain, l_chain))[0]

    chains_info = fv_info[0].xpath(".//div[@id='{}']".format(chains_info_div_id[1:]))

    chains = chains_info[0].xpath(".//table[@class='table table-alignment']")

    antigen_chain = chains_info[0].xpath(".//table[@class='table table-results']")

    antigen_id_row = antigen_chain[2].xpath("./tr")[1]

    antigen_id = antigen_id_row.xpath("./td/text()")[0]

    if h_chain == l_chain:
        l_chain = l_chain.lower()

    output = {}
    for i, c in enumerate(chains):
        chain_id = h_chain if i == 0 else l_chain
        residues = c.xpath("./tr/th/text()")
        aa_names = c.xpath("./tr/td/text()")
        chain = {extract_number_and_letter(r.strip()): a for a, r in zip(aa_names, residues)}
        output[chain_id] = chain

    return antigen_id, output


if __name__ == "__main__":
    df = pd.read_csv("test_set.csv")
    if sys.argv[1] == "convert_tsv_csv":
        df = pd.read_csv("4_jan.tsv", delimiter="\t")
        df = df[df['antigen_type'] == 'protein']
        df = df[['pdb', 'Hchain', 'Lchain', 'antigen_chain']]
        df.drop_duplicates('pdb', keep='first', inplace=True)
        df.to_csv("dataset_4_jan.csv", index=False, header=True)
    elif sys.argv[1] == "download_pdb":
        dir_list = os.listdir("pdb")

        for pdb_name in df['pdb']:
            if pdb_name + ".pdb" in dir_list:
                print("already existed")
                continue
            else:
                print("downloading", pdb_name)
                urllib.request.urlretrieve("https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/" +
                                           str.lower(pdb_name) + "/?scheme=chothia", "pdb/" + pdb_name + ".pdb")
    elif sys.argv[1] == "download_annotate":
        try:
            with open("downloaded_seqs.p", "rb") as f:
                seqs = pickle.load(f)
        except:
            seqs = {}
        for pdb_name, h, l in zip(df['pdb'], df['Hchain'], df['Lchain']):
            if pdb_name not in seqs.keys():
                print(pdb_name, h, l, "new added")
                try:
                    seqs[pdb_name] = download_annotated_seq(pdb_name, h, l)
                except:
                    print("failed")
                    continue

                with open("downloaded_seqs.p", "wb+") as f:
                    pickle.dump(seqs, f)
    elif sys.argv[1] == "create_processed_dataset":
        pdbs = []
        chain_types = []
        sequences = []
        labels = []
        indexes = []

        for ag_search, ab_h_chain, ab_l_chain, seqs, pdb in load_chains():
            print(seqs)
            print(pdb)
            chains, lbls, idxs = process_chains(ag_search, ab_h_chain, ab_l_chain, seqs, pdb)

            for c_type in ['H', 'L']:
                pdbs.append(pdb[0])
                chain_types.append(c_type)
                sequences.append(chains[c_type])
                labels.append(lbls[c_type])
                indexes.append(idxs[c_type])

        df = pd.DataFrame({
            "pdb": pdbs,
            "chain_type": chain_types,
            "sequence": sequences,
            'paratope': labels,
            'cdrs': indexes,
        })

        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv("processed_dataset_test.csv", index=False)
