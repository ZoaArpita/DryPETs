"""
esm2_scorer.py
==============
Pure ESM-2 masked-marginal mutation scoring.

This module knows nothing about catalytic triads, hotspots, or PDB structures.
It only knows how to ask ESM-2 "how plausible is this amino acid substitution,
given evolutionary/statistical patterns the model learned from sequence data."
The biology-aware filtering/annotation lives in biology_annotator.py.

Method
------
Masked-marginal scoring (Meier et al., 2021, "Language models enable zero-shot
prediction of the effects of mutations on protein function"):

    score(wt -> mut @ position i) = log P(mut | seq_masked_at_i) - log P(wt | seq_masked_at_i)

For each position of interest, the wild-type residue is masked and the model's
softmax over the vocabulary at that position gives log-probabilities for every
amino acid. The mutation score is the log-odds of the mutant vs. wild-type
residue *conditioned on the rest of the (unmutated) sequence*. Positive scores
indicate the model finds the mutation more plausible than the wild-type residue
in that structural/evolutionary context; negative scores indicate the opposite.

This is a proxy for "does this mutation look tolerable/favorable by the
statistics of natural protein sequences" — it is NOT a functional prediction
and carries no information about catalysis, stability, or binding on its own.
That's the annotator's job.

Dependencies
------------
    pip install torch transformers
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM

logger = logging.getLogger(__name__)

# Standard 20 amino acids (single-letter codes). X / gap / special tokens excluded.
AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

# Reasonable default checkpoints, smallest to largest.
# Pick based on available compute; 650M is a good accuracy/speed default.
MODEL_CHECKPOINTS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
}


@dataclasses.dataclass(frozen=True)
class Mutation:
    """A single point substitution, 1-indexed to match biology convention (e.g. S238F)."""

    position: int  # 1-indexed position in the wild-type sequence
    wt_aa: str
    mut_aa: str

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"

    @classmethod
    def from_string(cls, s: str) -> "Mutation":
        """Parse strings like 'S238F' -> Mutation(238, 'S', 'F')."""
        s = s.strip()
        wt_aa, mut_aa = s[0], s[-1]
        position = int(s[1:-1])
        return cls(position=position, wt_aa=wt_aa, mut_aa=mut_aa)


@dataclasses.dataclass
class MutationScore:
    mutation: Mutation
    llr: float  # log-likelihood ratio: log P(mut) - log P(wt), masked-marginal
    wt_logprob: float
    mut_logprob: float

    def to_dict(self) -> dict:
        return {
            "mutation": str(self.mutation),
            "position": self.mutation.position,
            "wt_aa": self.mutation.wt_aa,
            "mut_aa": self.mutation.mut_aa,
            "llr": self.llr,
            "wt_logprob": self.wt_logprob,
            "mut_logprob": self.mut_logprob,
        }


class ESM2Scorer:
    """
    Wraps an ESM-2 masked language model to compute masked-marginal mutation
    scores for a fixed wild-type sequence.

    Usage
    -----
        scorer = ESM2Scorer(sequence=wt_seq, checkpoint="650M")
        scores = scorer.score_mutations([Mutation.from_string("S238F")])
        dms = scorer.deep_mutational_scan(positions=[160, 206, 237])
    """

    def __init__(
        self,
        sequence: str,
        checkpoint: str = "650M",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Args:
            sequence: wild-type amino acid sequence (single-letter codes, no header).
            checkpoint: key into MODEL_CHECKPOINTS, or a full HF model name/path.
            device: 'cuda', 'cpu', or None to auto-detect.
            batch_size: number of masked positions to score per forward pass.
        """
        self.sequence = sequence.strip().upper()
        self._validate_sequence(self.sequence)

        self.checkpoint_name = MODEL_CHECKPOINTS.get(checkpoint, checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info("Loading %s onto %s", self.checkpoint_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)
        self.model = EsmForMaskedLM.from_pretrained(self.checkpoint_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token_id = self.tokenizer.mask_token_id
        # Pre-tokenize the wild-type sequence once; position i of `sequence`
        # (1-indexed) corresponds to token index i in the tokenized sequence
        # because ESM tokenizers prepend a single <cls> token.
        encoded = self.tokenizer(self.sequence, return_tensors="pt")
        self.input_ids = encoded["input_ids"].to(self.device)  # shape (1, L+2)
        self.attention_mask = encoded["attention_mask"].to(self.device)

        # Cache amino-acid -> vocab id
        self._aa_to_id = {
            aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS
        }
        missing = [aa for aa, tid in self._aa_to_id.items() if tid is None or tid == self.tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"Tokenizer missing expected amino acid tokens: {missing}")

    @staticmethod
    def _validate_sequence(sequence: str) -> None:
        valid = set(AMINO_ACIDS) | {"X"}  # allow X (unknown) in wild-type, just not as a mutant target
        bad = set(sequence) - valid
        if bad:
            raise ValueError(f"Sequence contains non-amino-acid characters: {sorted(bad)}")
        if len(sequence) == 0:
            raise ValueError("Sequence is empty")

    def _token_index(self, position_1indexed: int) -> int:
        """Map a 1-indexed wild-type sequence position to a token index in input_ids."""
        if not (1 <= position_1indexed <= len(self.sequence)):
            raise ValueError(
                f"Position {position_1indexed} out of range for sequence of "
                f"length {len(self.sequence)} (1-indexed)."
            )
        # ESM tokenizers prepend <cls>, so token index = position (1-indexed pos
        # lands on token index `position` when token 0 is <cls>).
        return position_1indexed

    @torch.no_grad()
    def _masked_logprobs_at_positions(self, positions: Sequence[int]) -> dict[int, torch.Tensor]:
        """
        For each 1-indexed position, mask it independently and return the
        log-softmax over the vocabulary at that masked location.

        Batches multiple masked copies of the sequence together for speed.
        Returns: {position: log_probs_tensor of shape (vocab_size,)}
        """
        results: dict[int, torch.Tensor] = {}
        unique_positions = list(dict.fromkeys(positions))  # de-dup, preserve order

        for batch_start in range(0, len(unique_positions), self.batch_size):
            batch_positions = unique_positions[batch_start : batch_start + self.batch_size]
            batch_input_ids = self.input_ids.repeat(len(batch_positions), 1).clone()
            batch_attention = self.attention_mask.repeat(len(batch_positions), 1)

            token_indices = []
            for row, pos in enumerate(batch_positions):
                tok_idx = self._token_index(pos)
                token_indices.append(tok_idx)
                batch_input_ids[row, tok_idx] = self.mask_token_id

            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention)
            log_probs = F.log_softmax(outputs.logits, dim=-1)  # (batch, L+2, vocab)

            for row, (pos, tok_idx) in enumerate(zip(batch_positions, token_indices)):
                results[pos] = log_probs[row, tok_idx, :].detach().cpu()

        return results

    def score_mutations(self, mutations: Iterable[Mutation]) -> list[MutationScore]:
        """
        Score an explicit list of mutations using masked-marginal log-odds.

        Validates that each mutation's declared wild-type residue matches the
        actual sequence at that position (catches off-by-one / wrong-numbering
        errors early, which are a very common source of silent bugs in DMS work).
        """
        mutations = list(mutations)
        for m in mutations:
            actual_wt = self.sequence[m.position - 1]
            if actual_wt != m.wt_aa:
                raise ValueError(
                    f"Mutation {m} declares wild-type '{m.wt_aa}' at position "
                    f"{m.position}, but sequence has '{actual_wt}'. Check indexing/numbering."
                )

        positions = [m.position for m in mutations]
        logprob_by_position = self._masked_logprobs_at_positions(positions)

        scores = []
        for m in mutations:
            log_probs = logprob_by_position[m.position]
            wt_lp = log_probs[self._aa_to_id[m.wt_aa]].item()
            mut_lp = log_probs[self._aa_to_id[m.mut_aa]].item()
            scores.append(
                MutationScore(
                    mutation=m,
                    llr=mut_lp - wt_lp,
                    wt_logprob=wt_lp,
                    mut_logprob=mut_lp,
                )
            )
        return scores

    def deep_mutational_scan(
        self,
        positions: Sequence[int] | None = None,
        exclude_wt: bool = True,
    ) -> list[MutationScore]:
        """
        Score every possible amino acid substitution at the given positions
        (or all positions in the sequence if none specified).

        Args:
            positions: 1-indexed positions to scan. Defaults to the full sequence,
                which is expensive for long proteins — prefer passing a restricted
                set (e.g. active-site + surrounding residues, or all positions if
                you have the compute budget for full-DMS screening).
            exclude_wt: skip the "mutation" that is identical to wild-type.

        Returns:
            Flat list of MutationScore, sorted by descending LLR (most favorable
            substitutions first).
        """
        positions = list(positions) if positions is not None else list(range(1, len(self.sequence) + 1))
        logprob_by_position = self._masked_logprobs_at_positions(positions)

        scores = []
        for pos in positions:
            wt_aa = self.sequence[pos - 1]
            if wt_aa not in AMINO_ACIDS:
                logger.warning("Skipping position %d: wild-type residue '%s' not a standard AA", pos, wt_aa)
                continue
            log_probs = logprob_by_position[pos]
            wt_lp = log_probs[self._aa_to_id[wt_aa]].item()
            for mut_aa in AMINO_ACIDS:
                if exclude_wt and mut_aa == wt_aa:
                    continue
                mut_lp = log_probs[self._aa_to_id[mut_aa]].item()
                scores.append(
                    MutationScore(
                        mutation=Mutation(pos, wt_aa, mut_aa),
                        llr=mut_lp - wt_lp,
                        wt_logprob=wt_lp,
                        mut_logprob=mut_lp,
                    )
                )

        scores.sort(key=lambda s: s.llr, reverse=True)
        return scores

    def pseudo_perplexity(self, positions: Sequence[int] | None = None) -> float:
        """
        Optional diagnostic: average masked-marginal log-likelihood of the
        wild-type sequence itself over the given positions (or full sequence).
        Useful as a sanity check that the model 'likes' the wild-type protein
        overall before trusting its mutation scores.
        """
        positions = list(positions) if positions is not None else list(range(1, len(self.sequence) + 1))
        logprob_by_position = self._masked_logprobs_at_positions(positions)
        wt_logprobs = [
            logprob_by_position[pos][self._aa_to_id[self.sequence[pos - 1]]].item()
            for pos in positions
            if self.sequence[pos - 1] in AMINO_ACIDS
        ]
        avg_nll = -sum(wt_logprobs) / len(wt_logprobs)
        return avg_nll  # lower is better (model finds sequence more plausible)


def load_fasta(path: str | Path) -> str:
    """Minimal FASTA reader: returns the concatenated sequence from the first record."""
    path = Path(path)
    lines = path.read_text().splitlines()
    seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
    return "".join(seq_lines)


def scores_to_records(scores: Sequence[MutationScore]) -> list[dict]:
    """Convert scores to plain dicts, ready for pandas.DataFrame(...) or JSON dump."""
    return [s.to_dict() for s in scores]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ESM-2 masked-marginal mutation scoring")
    parser.add_argument("--fasta", type=str, required=True, help="Path to wild-type FASTA file")
    parser.add_argument("--checkpoint", type=str, default="650M", choices=list(MODEL_CHECKPOINTS))
    parser.add_argument(
        "--mutations",
        type=str,
        nargs="*",
        default=None,
        help="Explicit mutations to score, e.g. S238F N246D. If omitted, runs a full DMS.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="*",
        default=None,
        help="Restrict a full DMS to these 1-indexed positions (ignored if --mutations given).",
    )
    parser.add_argument("--output", type=str, default="scores.json")
    parser.add_argument("--top-k", type=int, default=20, help="Print top-k results by LLR to stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    wt_sequence = load_fasta(args.fasta)
    scorer = ESM2Scorer(sequence=wt_sequence, checkpoint=args.checkpoint)

    if args.mutations:
        muts = [Mutation.from_string(m) for m in args.mutations]
        results = scorer.score_mutations(muts)
        results.sort(key=lambda s: s.llr, reverse=True)
    else:
        results = scorer.deep_mutational_scan(positions=args.positions)

    Path(args.output).write_text(json.dumps(scores_to_records(results), indent=2))
    logger.info("Wrote %d scores to %s", len(results), args.output)

    print(f"\nTop {min(args.top_k, len(results))} mutations by masked-marginal LLR:")
    for s in results[: args.top_k]:
        print(f"  {s.mutation}\tLLR={s.llr:+.3f}")
