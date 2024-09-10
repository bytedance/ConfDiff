"""Fast-folding proteins infos

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
# Constants
# =============================================================================
DATASET_DIR = None
PRECOMPUTED_REF_DIR = None
DEFAULT_METADATA = None


PROTEIN_INFO = [
    ('BBA',         '1FME',     'PDB entry 1FME'),
    ('Villin',      '2F4K',     'PDB entry 2F4K'), 
    ('Trp-cage',    '2JOF',     'PDB entry 2JOF'),
    ('BBL',         '2WAV',     'PDB entry 2WXC'), 
    ('a3D',         'A3D',      'PDB entry 2A3D'),
    ('Chignolin',   'CLN025',   'Honda et al. (2008)'),
    ('WW domain',   'GTT',      'PDB entry 2F21'),
    ('Lambda',      'lambda',   'PDB entry 1LMB'),
    ('NTL9',        'NTL9',     'PDB entry NTL9'),
    ('Protein G',   'NuG2',     'PDB entry 1MIO'),
    ('Protein B',   'PRB',      'PDB entry 1PRB'),
    ('Homeodomain', 'UVF',      'PDB entry 2P6J'), 
]


PROTEIN_LIST = [entry[1] for entry in PROTEIN_INFO]


CHAIN_NAME_TO_PROT_NAME = {entry[1]: entry[0] for entry in PROTEIN_INFO}


CHAIN_NAME_TO_SEQRES = {
    "lambda":"PLTQEQLEAARRLKAIWEKKKNELGLSYESVADKMGMGQSAVAALFNGINALNAYNAALLAKILKVSVEEFSPSIAREIY",
    "2F4K":"LSDEDFKAVFGMTRSAFANLPLWLQQHLLKEKGLF",
    "CLN025":"YYDPETGTWY",
    "2JOF":"DAYAQWLADGGPSSGRPPPS",
    "1FME":"EQYTAKYKGRTFRNEKELRDFIEKFKGR",
    "GTT":"GSKLPPGWEKRMSRDGRVYYFNHITGTTQFERPSG",
    "NTL9":"MKVIFLKDVKGMGKKGEIKNVADGYANNFLFKQGLAIEA",
    "2WAV":"GSQNNDALSPAIRRLLAEWNLDASAIKGTGVGGRLTREDVEKHLAKA",
    "PRB":"LKNAIEDAIAELKKAGITSDFYFNAINKAKTVEEVNALVNEILKAHA",
    "UVF":"MKQWSENVEEKLKEFVKRHQRITQEELHQYAQRLGLNEEAIRQFFEEFEQRK",
    "NuG2":"DTYKLVIVLNGTTFTYTTEAVDAATAEKVFKQYANDAGVDGEWTYDAATKTFTVTE",
    "A3D":"MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEKEIAAFESELQAYKGKGNPEVEALRKEAAAIRDELQAYRHN",
}


TICA_LAGTIME = {
    '1FME': 2500,
    '2F4K': 2500,
    '2JOF': 2500,
    '2WAV': 25000,
    'A3D': 2500,
    'CLN025': 1600,
    'GTT': 4000,
    'lambda': 630,
    'NTL9': 10000,
    'NuG2': 4000,
    'PRB': 4000,
    'UVF': 4000
}


PWD_NEIGHBOR_EXCLUDE = 3

PWD_BINS = 50

RG_BINS = 50

TICA_BINS = 50
