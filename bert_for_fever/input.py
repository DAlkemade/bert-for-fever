import pickle

import pandas as pd

EMPTY_TOKEN = 'EMPTY'


def get_doc_text(conn, id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT lines FROM documents WHERE id = ?",
        (id,)
    )
    result = cursor.fetchone()
    cursor.close()
    return result


def get_golden_docs(evidence):
    all_evi = [[e[2], e[3]] for eg in evidence for e in eg if e[3] is not None]  # from baseline scorer
    docs = []
    for entry in all_evi:
        id = entry[0]
        docs.append(id)

    return docs


def parse_doc(doc_raw):
    """
    Parse a list of lines from a raw document text, with the index in the list
    correponding to the line index in the data entries
    """
    new = []
    lines = doc_raw.split("\n")
    char_count = 0
    for line in lines:
        # print('Line: {}'.format(line))
        line = line.split("\t")
        #   TODO: THIS MIGHT DROP PARTS OF SENTENCES AFTER A TAB
        if len(line) > 1 and len(line[1]) > 1:
            new.append(line[1])
            char_count += len(line[1])
        else:
            new.append(EMPTY_TOKEN)
    return new


def load_or_create_evidence(load_previous: bool, claim_ids):
    if load_previous:
        try:
            with open('/content/drive/My Drive/Overig/docs_evidence_full_hnm.pkl', 'rb') as f:
                evidence = pickle.load(f)
            with open('/content/drive/My Drive/Overig/docs_evidence_all_full_hnm.pkl', 'rb') as f:
                evidence_all_scores = pickle.load(f)
        except AttributeError:
            raise AttributeError("Supply evidence location arguments when loading previous evidence.")
        print("Loaded evidence")
    else:
        evidence = dict((el, []) for el in dict.fromkeys(claim_ids))
        evidence_all_scores = dict((el, []) for el in dict.fromkeys(claim_ids))
    return evidence, evidence_all_scores


def retrieve_claim_doc_ids(data_fname, sentence_ids: bool = False):
    """Retrieve columns from original tsv preprocessed data."""
    data = pd.read_csv(data_fname)
    claim_ids = list(data.claim_id)
    doc_ids = list(data.doc_id)
    if sentence_ids:
        sentence_idxs = list(data.sentence_idx)
        return claim_ids, doc_ids, sentence_idxs
    return claim_ids, doc_ids
