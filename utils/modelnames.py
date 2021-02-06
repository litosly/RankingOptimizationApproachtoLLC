from models.llc_score import LP1Simplified
from models.uac import UAC
from models.bac import BAC
from models.plrec import plrec
from models.llc_rank import LLCRank
from models.llc_rank_nonincremental import LLCRankNonIncremental


models = {
    "PLRec": plrec,
}

critiquing_models = {
    "llc_score": LP1Simplified,
    "uac": UAC,
    "bac": BAC,
    "llc_rank": LLCRank,
    "llc_rank2": LLCRankNonIncremental
}
