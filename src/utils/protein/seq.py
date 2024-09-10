"""Methods for sequence alignment

----------------
Copyright (2023) Bytedance Ltd. and/or its affiliates
"""

# =============================================================================
# Imports
# =============================================================================
from collections import namedtuple
from packaging import version

import Bio
from Bio.SeqIO.PdbIO import _res2aacode as res2aacode


# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

if version.parse(Bio.__version__) < version.parse("1.80"):
    from Bio.pairwise2 import align

    pairwise_globalxx = align.globalxx
else:
    from Bio.Align import PairwiseAligner

    def pairwise_globalxx(seq1, seq2):
        """Replace deprecated pairwise2.align.globalxx with Align.PairwiseAligner
        globalxx: global alignment with no match parameters and no gap penalty

        Returns:
            A list of named tuples resembling pairwise2.globalxx output
                [PairwiseAlignment(seqA=, seqB=), ...]
        """
        aligner = PairwiseAligner()
        aligner.mode = "global"
        # default mismatch and gap scores are 0, no need to specify
        pairalign = namedtuple("PairwiseAlignment", field_names=["seqA", "seqB"])
        return [
            pairalign(seqA=record[0], seqB=record[1])
            for record in aligner.align(seq1, seq2)
        ]


def pairwise_local_with_score(seq1, seq2, substitution_matrix="BLOSUM62", return_top=1):
    """Pairwise local sequence alignment with provided substitution matrix
    Default using BLOSUM62
    """
    if version.parse(Bio.__version__) < version.parse("1.80"):
        raise NotImplementedError(
            "pairwise_local_with_score is not implemented for Biopython < 1.80"
        )

    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "local"
    if substitution_matrix is not None:
        if isinstance(substitution_matrix, str):
            from Bio.Align import substitution_matrices

            substitution_matrix = substitution_matrices.load(substitution_matrix)
        aligner.substitution_matrix = substitution_matrix
    alignments = sorted(aligner.align(seq1, seq2))
    pairalign = namedtuple("PairwiseAlignment", field_names=["seqA", "seqB", "score"])
    return [
        pairalign(seqA=align[0], seqB=align[1], score=align.score)
        for align in alignments[:return_top]
    ]


# =============================================================================
# Classes
# =============================================================================
