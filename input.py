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
