"""
biology_annotator.py
=====================
Fuses ESM-2 masked-marginal mutation scores (esm2_scorer.py — pure sequence
statistics) with domain-specific biological signals to produce a
biologically-grounded ranking of candidate PETase mutations.

ESM-2 alone has no notion of "catalytic triad," "solvent accessibility," or
"this exact mutation was already validated in a published thermostable
variant." This module is where that domain knowledge gets injected, as a
transparent, rule-based fusion layer — not a second learned model.

Signal sources
--------------
1. Catalytic-site proximity
   reference_data/petase_catalytic_residues.json (positions) +
   reference_data/petase_structure.pdb (3D Cα distances, if available)
2. Solvent-accessible surface area (SASA)
   computed from reference_data/petase_structure.pdb via FreeSASA, cached
   to reference_data/sasa_cache.json so it's only computed once
3. Literature hotspot overlap
   reference_data/known_hotspots.json (FAST-PETase, ThermoPETase, HotPETase, ...)

Degradation behavior
---------------------
If reference_data/petase_structure.pdb is missing (see
reference_data/PETASE_STRUCTURE_README.md), structural signals (3D distance,
SASA) are skipped and a warning is logged. Sequence-position-based signals
(exact catalytic-residue match, hotspot overlap) still work with JSON alone.

Dependencies
------------
    pip install biopython freesasa
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_DIR = Path(__file__).parent / "reference_data"


# --------------------------------------------------------------------------
# Reference data loaders
# --------------------------------------------------------------------------

class CatalyticSiteReference:
    """Loads and indexes petase_catalytic_residues.json for fast position lookup."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text())
        self.raw = data
        self.numbering_convention = data.get("numbering_convention", "unspecified")

        # position -> (site_name, residue_label, mutation_risk, role)
        self.position_index: dict[int, dict] = {}
        for site_key in ("catalytic_triad", "oxyanion_hole", "substrate_binding_notable_residues"):
            site = data.get(site_key)
            if not site:
                continue
            for pos, res_label in zip(site["positions"], site["residues"]):
                self.position_index[pos] = {
                    "site": site_key,
                    "residue_label": res_label,
                    "mutation_risk": site.get("mutation_risk", "unknown"),
                    "role": site.get("role", ""),
                }

    def lookup(self, position: int) -> Optional[dict]:
        return self.position_index.get(position)

    def all_reference_positions(self) -> list[int]:
        return sorted(self.position_index.keys())


class HotspotReference:
    """Loads and indexes known_hotspots.json for mutation-string overlap lookup."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text())
        self.raw = data
        self.verified_positions: set[int] = set(data.get("hotspot_positions_verified", []))

        # exact "S121E" style match -> variant name(s)
        self.mutation_index: dict[str, list[str]] = {}
        for variant in data.get("variants", []):
            if not variant.get("verified", False):
                continue
            for mut_str in variant.get("mutations") or []:
                self.mutation_index.setdefault(mut_str.upper(), []).append(variant["name"])

    def exact_match(self, mutation_str: str) -> list[str]:
        """Return variant names whose published mutation list contains this exact mutation."""
        return self.mutation_index.get(mutation_str.upper(), [])

    def position_overlap(self, position: int) -> bool:
        """True if this position was mutated in any verified published variant (any substitution)."""
        return position in self.verified_positions


# --------------------------------------------------------------------------
# Structural signals (optional — requires a real PDB file)
# --------------------------------------------------------------------------

class StructuralAnnotator:
    """
    Computes 3D Cα distance-to-catalytic-site and per-residue SASA from a PDB
    structure. Degrades to a no-op if the structure file isn't present —
    callers should check `.available` before relying on its outputs.
    """

    def __init__(self, pdb_path: Path, reference_positions: list[int], cache_path: Path):
        self.available = False
        self.pdb_path = Path(pdb_path)
        self.reference_positions = reference_positions
        self.cache_path = Path(cache_path)
        self._ca_coords: dict[int, tuple[float, float, float]] = {}
        self._sasa_by_position: dict[int, float] = {}

        if not self.pdb_path.exists():
            logger.warning(
                "Structure file not found at %s — 3D distance and SASA signals "
                "will be skipped. See PETASE_STRUCTURE_README.md.",
                self.pdb_path,
            )
            return

        try:
            self._load_structure()
            self._load_or_compute_sasa()
            self.available = True
        except ImportError as e:
            logger.warning(
                "Structural annotation disabled (%s). Install with: "
                "pip install biopython freesasa", e,
            )
        except Exception as e:
            logger.warning("Structural annotation disabled due to error: %s", e)

    def _load_structure(self) -> None:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("petase", str(self.pdb_path))
        model = next(structure.get_models())
        chain = next(model.get_chains())  # assumes single-chain monomer, true for IsPETase

        for residue in chain:
            if "CA" not in residue:
                continue
            resnum = residue.id[1]  # PDB author numbering
            self._ca_coords[resnum] = residue["CA"].coord

    def _load_or_compute_sasa(self) -> None:
        if self.cache_path.exists():
            logger.info("Loading cached SASA values from %s", self.cache_path)
            cached = json.loads(self.cache_path.read_text())
            self._sasa_by_position = {int(k): v for k, v in cached.items()}
            return

        import freesasa

        structure = freesasa.Structure(str(self.pdb_path))
        result = freesasa.calc(structure)
        residue_areas = result.residueAreas()

        # freesasa groups by chain; take the first (only) chain
        chain_id = next(iter(residue_areas.keys()))
        for resnum_str, area in residue_areas[chain_id].items():
            try:
                self._sasa_by_position[int(resnum_str)] = area.total
            except ValueError:
                continue  # skip insertion-code residues

        self.cache_path.write_text(json.dumps(self._sasa_by_position, indent=2))
        logger.info("Computed and cached SASA for %d residues to %s",
                     len(self._sasa_by_position), self.cache_path)

    def min_distance_to_reference(self, position: int) -> Optional[tuple[float, int]]:
        """Returns (distance_angstrom, nearest_reference_position), or None if unavailable."""
        if not self.available or position not in self._ca_coords:
            return None
        import numpy as np

        query = np.array(self._ca_coords[position])
        best = None
        for ref_pos in self.reference_positions:
            if ref_pos not in self._ca_coords:
                continue
            dist = float(np.linalg.norm(query - np.array(self._ca_coords[ref_pos])))
            if best is None or dist < best[0]:
                best = (dist, ref_pos)
        return best

    def sasa(self, position: int) -> Optional[float]:
        return self._sasa_by_position.get(position)

    @staticmethod
    def sasa_category(sasa_value: Optional[float]) -> str:
        """Rough buried/exposed bucketing. Thresholds are conventional rules of
        thumb (absolute SASA in Å²), not position-specific calibration."""
        if sasa_value is None:
            return "unknown"
        if sasa_value < 20:
            return "buried"
        if sasa_value < 60:
            return "partially_exposed"
        return "exposed"


# --------------------------------------------------------------------------
# Fusion layer
# --------------------------------------------------------------------------

@dataclasses.dataclass
class AnnotatedMutation:
    mutation: str          # e.g. "S238F"
    position: int
    wt_aa: str
    mut_aa: str
    llr: float             # from esm2_scorer

    catalytic_site_hit: Optional[str]     # residue label if position IS a reference site, else None
    mutation_risk: Optional[str]          # "very_high" / "high" / "moderate" / None
    distance_to_catalytic_site: Optional[float]  # Angstroms, None if structure unavailable
    nearest_catalytic_position: Optional[int]

    sasa: Optional[float]
    sasa_category: str

    hotspot_exact_match: list[str]        # variant names with this exact mutation published
    hotspot_position_overlap: bool        # position mutated in some published variant (any AA)

    composite_score: float
    flags: list[str]
    rationale: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class BiologyAnnotator:
    """
    Top-level orchestrator. Loads all reference data once, then annotates a
        batch of ESM-2 mutation scores.

    Usage
    -----
        annotator = BiologyAnnotator(reference_dir="models/non_finetuned/reference_data")
        annotated = annotator.annotate(esm2_scores)  # list of dicts from esm2_scorer
        top = annotator.top_candidates(annotated, n=10)
    """

    # Fusion weights — deliberately simple and exposed as constants so they're
    # easy to justify/tune during validation, rather than buried magic numbers.
    CATALYTIC_TRIAD_PENALTY = 8.0     # subtracted if position is the nucleophile/triad
    OXYANION_HOLE_PENALTY = 4.0       # subtracted if position is the oxyanion hole
    HOTSPOT_EXACT_MATCH_BONUS = 5.0   # this exact mutation already validated in literature
    HOTSPOT_POSITION_BONUS = 1.5      # position validated, different substitution
    NEAR_SITE_DISTANCE_ANGSTROM = 8.0 # Cα-Cα cutoff for "near active site" flag (informational only)

    def __init__(self, reference_dir: str | Path = DEFAULT_REFERENCE_DIR):
        reference_dir = Path(reference_dir)
        self.catalytic_ref = CatalyticSiteReference(reference_dir / "petase_catalytic_residues.json")
        self.hotspot_ref = HotspotReference(reference_dir / "known_hotspots.json")
        self.structural = StructuralAnnotator(
            pdb_path=reference_dir / "petase_structure.pdb",
            reference_positions=self.catalytic_ref.all_reference_positions(),
            cache_path=reference_dir / "sasa_cache.json",
        )

    def annotate_one(self, position: int, wt_aa: str, mut_aa: str, llr: float) -> AnnotatedMutation:
        mutation_str = f"{wt_aa}{position}{mut_aa}"
        flags: list[str] = []

        site_hit = self.catalytic_ref.lookup(position)
        catalytic_site_hit = site_hit["residue_label"] if site_hit else None
        mutation_risk = site_hit["mutation_risk"] if site_hit else None

        dist_result = self.structural.min_distance_to_reference(position)
        distance, nearest_pos = dist_result if dist_result else (None, None)

        sasa_val = self.structural.sasa(position)
        sasa_cat = StructuralAnnotator.sasa_category(sasa_val)

        exact_matches = self.hotspot_ref.exact_match(mutation_str)
        position_overlap = self.hotspot_ref.position_overlap(position)

        # ---- composite score ----
        score = llr

        if site_hit and site_hit["site"] == "catalytic_triad":
            score -= self.CATALYTIC_TRIAD_PENALTY
            flags.append("CATALYTIC_TRIAD_RESIDUE")
        elif site_hit and site_hit["site"] == "oxyanion_hole":
            score -= self.OXYANION_HOLE_PENALTY
            flags.append("OXYANION_HOLE_RESIDUE")
        elif site_hit and site_hit["site"] == "substrate_binding_notable_residues":
            flags.append("SUBSTRATE_BINDING_RESIDUE")  # informational, no penalty — these are often good targets

        if exact_matches:
            score += self.HOTSPOT_EXACT_MATCH_BONUS
            flags.append(f"MATCHES_PUBLISHED_VARIANT:{','.join(exact_matches)}")
        elif position_overlap:
            score += self.HOTSPOT_POSITION_BONUS
            flags.append("KNOWN_HOTSPOT_POSITION")

        if distance is not None and distance <= self.NEAR_SITE_DISTANCE_ANGSTROM and not site_hit:
            flags.append(f"NEAR_ACTIVE_SITE({distance:.1f}A)")

        if sasa_cat == "buried":
            flags.append("BURIED_RESIDUE")

        rationale = self._build_rationale(
            mutation_str, llr, site_hit, exact_matches, position_overlap,
            distance, nearest_pos, sasa_cat,
        )

        return AnnotatedMutation(
            mutation=mutation_str,
            position=position,
            wt_aa=wt_aa,
            mut_aa=mut_aa,
            llr=llr,
            catalytic_site_hit=catalytic_site_hit,
            mutation_risk=mutation_risk,
            distance_to_catalytic_site=distance,
            nearest_catalytic_position=nearest_pos,
            sasa=sasa_val,
            sasa_category=sasa_cat,
            hotspot_exact_match=exact_matches,
            hotspot_position_overlap=position_overlap,
            composite_score=score,
            flags=flags,
            rationale=rationale,
        )

    def annotate(self, esm2_scores: list[dict]) -> list[AnnotatedMutation]:
        """
        Args:
            esm2_scores: list of dicts as produced by esm2_scorer.scores_to_records(),
                each with keys position, wt_aa, mut_aa, llr (extra keys ignored).
        """
        annotated = [
            self.annotate_one(s["position"], s["wt_aa"], s["mut_aa"], s["llr"])
            for s in esm2_scores
        ]
        annotated.sort(key=lambda a: a.composite_score, reverse=True)
        return annotated

    @staticmethod
    def top_candidates(annotated: list[AnnotatedMutation], n: int = 10,
                        exclude_high_risk: bool = True) -> list[AnnotatedMutation]:
        """Convenience filter matching the team's 'top 10' funnel step before the
        finetuned model stage."""
        pool = annotated
        if exclude_high_risk:
            pool = [a for a in pool if "CATALYTIC_TRIAD_RESIDUE" not in a.flags]
        return pool[:n]

    @staticmethod
    def _build_rationale(
        mutation_str: str, llr: float, site_hit: Optional[dict],
        exact_matches: list[str], position_overlap: bool,
        distance: Optional[float], nearest_pos: Optional[int], sasa_cat: str,
    ) -> str:
        parts = [f"ESM-2 masked-marginal LLR={llr:+.2f}."]
        if site_hit:
            parts.append(
                f"Position IS a reference {site_hit['site'].replace('_', ' ')} "
                f"residue ({site_hit['residue_label']}, risk={site_hit['mutation_risk']})."
            )
        elif distance is not None:
            parts.append(f"Nearest catalytic-site residue is {distance:.1f}A away (position {nearest_pos}).")
        if exact_matches:
            parts.append(f"Exact mutation already validated in: {', '.join(exact_matches)}.")
        elif position_overlap:
            parts.append("Position was mutated (different substitution) in a published thermostable variant.")
        if sasa_cat != "unknown":
            parts.append(f"SASA category: {sasa_cat}.")
        return " ".join(parts)


def annotated_to_records(annotated: list[AnnotatedMutation]) -> list[dict]:
    return [a.to_dict() for a in annotated]


def _load_fasta_sequence(path: str | Path) -> str:
    """Minimal FASTA reader (kept local so this module doesn't have to import
    esm2_scorer just for one helper)."""
    lines = Path(path).read_text().splitlines()
    seq_lines = [line.strip() for line in lines if line and not line.startswith(">")]
    return "".join(seq_lines)


def build_ten_mutations_payload(
    wt_sequence: str,
    top: list[AnnotatedMutation],
    fasta_path: Optional[str | Path] = None,
    pdb_path: Optional[str | Path] = None,
) -> dict:
    """
    Package the top-N annotated candidates into everything petase_app.py
    needs to load them straight into the WT-vs-mutant 3D comparison: the WT
    sequence itself, a derived full mutant sequence per candidate (WT with
    that single substitution applied), and a pointer to the reference
    structure file if one is available on disk.
    """
    candidates = []
    for a in top:
        actual_wt = wt_sequence[a.position - 1]
        if actual_wt != a.wt_aa:
            raise ValueError(
                f"WT sequence mismatch at position {a.position}: sequence has "
                f"'{actual_wt}', annotation expects '{a.wt_aa}'. The --fasta "
                "passed to this script must be the same wild-type sequence "
                "that was scored by esm2_scorer.py."
            )
        mutant_sequence = wt_sequence[: a.position - 1] + a.mut_aa + wt_sequence[a.position :]
        candidates.append(
            {
                "mutation": a.mutation,
                "position": a.position,
                "wt_aa": a.wt_aa,
                "mut_aa": a.mut_aa,
                "mutant_sequence": mutant_sequence,
                "llr": a.llr,
                "composite_score": a.composite_score,
                "flags": a.flags,
                "rationale": a.rationale,
            }
        )

    pdb_path = Path(pdb_path) if pdb_path else None

    return {
        "wt_sequence": wt_sequence,
        "wt_fasta_path": str(fasta_path) if fasta_path else None,
        "reference_pdb_path": str(pdb_path) if pdb_path and pdb_path.exists() else None,
        "mutations": candidates,
    }


def write_ten_mutations_file(
    wt_sequence: str,
    top: list[AnnotatedMutation],
    output_path: str | Path,
    fasta_path: Optional[str | Path] = None,
    pdb_path: Optional[str | Path] = None,
) -> Path:
    """Build the payload above and write it to disk as JSON for petase_app.py
    to pick up (via its sidebar 'Load top candidate' uploader)."""
    payload = build_ten_mutations_payload(
        wt_sequence, top, fasta_path=fasta_path, pdb_path=pdb_path
    )
    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %d candidate mutant sequence(s) to %s", len(payload["mutations"]), output_path
    )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate ESM-2 mutation scores with PETase biology signals"
    )
    parser.add_argument("--scores", type=str, required=True,
                         help="Path to JSON output from esm2_scorer.py (scores.json)")
    parser.add_argument("--reference-dir", type=str, default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output", type=str, default="annotated_scores.json")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--fasta", type=str, default=None,
        help="Path to the WT FASTA that was scored (same file passed to "
             "esm2_scorer.py's --fasta). Required to also emit "
             "--ten-mutations-output, since that file needs full mutant "
             "sequences, not just positions.",
    )
    parser.add_argument(
        "--ten-mutations-output", type=str, default="ten_mutations.json",
        help="Where to write the top-N candidates (WT sequence + derived "
             "mutant sequence + reference structure path per candidate) for "
             "petase_app.py to load directly. Only written if --fasta is given.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    esm2_scores = json.loads(Path(args.scores).read_text())
    annotator = BiologyAnnotator(reference_dir=args.reference_dir)
    annotated = annotator.annotate(esm2_scores)

    Path(args.output).write_text(json.dumps(annotated_to_records(annotated), indent=2))
    logger.info("Wrote %d annotated mutations to %s", len(annotated), args.output)

    top = annotator.top_candidates(annotated, n=args.top_n)
    print(f"\nTop {len(top)} candidates (catalytic-triad hits excluded):")
    for a in top:
        print(f"  {a.mutation}\tscore={a.composite_score:+.2f}\tflags={a.flags}")
        print(f"    {a.rationale}")

    if args.fasta:
        wt_sequence = _load_fasta_sequence(args.fasta)
        write_ten_mutations_file(
            wt_sequence=wt_sequence,
            top=top,
            output_path=args.ten_mutations_output,
            fasta_path=args.fasta,
            pdb_path=Path(args.reference_dir) / "petase_structure.pdb",
        )
    else:
        logger.warning(
            "No --fasta given, so %s was NOT written. petase_app.py needs "
            "that file to load the top candidates' 3D models — rerun with "
            "--fasta <path to WT fasta> to produce it.",
            args.ten_mutations_output,
        )
