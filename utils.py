import logging

def get_logger(logfile):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    DUTY = get_DUTY_entity(tag_seq, char_seq)

    return PER, LOC, ORG, DUTY

def get_PER_entity(tag_seq, char_seq):
    PER, temp = [], []
    for tag, char in zip(tag_seq, char_seq):
        if tag == 'B-PER' or tag == 'I-PER':
            temp.append(char)
        elif temp:
            PER.append(''.join(temp))
            temp = []
    if temp:
        PER.append(''.join(temp))
    return PER

def get_LOC_entity(tag_seq, char_seq):
    LOC, temp = [], []
    for tag, char in zip(tag_seq, char_seq):
        if tag == 'B-LOC' or tag == 'I-LOC':
            temp.append(char)
        elif temp:
            LOC.append(''.join(temp))
            temp = []
    if temp:
        LOC.append(''.join(temp))
    return LOC

def get_ORG_entity(tag_seq, char_seq):
    ORG, temp = [], []
    for tag, char in zip(tag_seq, char_seq):
        if tag == 'B-ORG' or tag == 'I-ORG':
            temp.append(char)
        elif temp:
            ORG.append(''.join(temp))
            temp = []
    if temp:
        ORG.append(''.join(temp))
    return ORG

def get_DUTY_entity(tag_seq, char_seq):
    DUTY, temp = [], []
    for tag, char in zip(tag_seq, char_seq):
        if tag == 'B-DUTY' or tag == 'I-DUTY':
            temp.append(char)
        elif temp:
            DUTY.append(''.join(temp))
            temp = []
    if temp:
        DUTY.append(''.join(temp))
    return DUTY