import io
import json
import re
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components

from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter("ignore", PDBConstructionWarning)

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_LENGTH = 400  # the free ESM Atlas API rejects longer sequences

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="PETase WT vs Mutant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 PETase Wild-Type vs Mutant Structural Analysis")
st.caption(
    "AI-guided structural visualization for predicted PETase mutations"
)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence):
    """
    Accepts either a raw protein sequence or FASTA text.
    Removes FASTA headers, whitespace and invalid characters.
    """
    if not sequence:
        return ""

    lines = sequence.strip().splitlines()

    if lines and lines[0].startswith(">"):
        lines = lines[1:]

    sequence = "".join(lines)
    sequence = re.sub(r"\s+", "", sequence).upper()

    invalid = set(sequence) - AMINO_ACIDS

    if invalid:
        raise ValueError(
            f"Invalid amino-acid characters found: {', '.join(sorted(invalid))}"
        )

    return sequence


def get_mutation(wt_sequence, mutant_sequence):
    """
    Finds single amino-acid substitutions.

    Example:
        WT:     AAAAAST...
        Mutant: AAAAAGT...

    Returns:
        S6G
    """

    wt = clean_sequence(wt_sequence)
    mutant = clean_sequence(mutant_sequence)

    if len(wt) != len(mutant):
        raise ValueError(
            "WT and mutant sequences must have the same length "
            "for a simple point-mutation comparison."
        )

    differences = []

    for i, (a, b) in enumerate(zip(wt, mutant), start=1):
        if a != b:
            differences.append((i, a, b))

    if not differences:
        return None, []

    mutation_names = [
        f"{wt_aa}{position}{mut_aa}"
        for position, wt_aa, mut_aa in differences
    ]

    return ", ".join(mutation_names), differences


# ---------------------------------------------------------
# ESMFOLD API
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def fold_sequence_esmfold(sequence):
    """
    Folds a single sequence using the free ESM Atlas API and returns
    PDB text. Cached by sequence, so re-folding the same sequence
    twice (e.g. after a rerun) reuses the previous result instead of
    calling the API again.

    Raises a RuntimeError with a human-readable message on any
    failure — the free API is known to be occasionally flaky/rate
    limited, so callers should show this to the user rather than
    letting it crash the app.
    """

    if len(sequence) > ESMFOLD_MAX_LENGTH:
        raise RuntimeError(
            f"Sequence is {len(sequence)} residues long, which exceeds "
            f"the free ESMFold API's limit of {ESMFOLD_MAX_LENGTH}. "
            "Fold it locally (HuggingFace facebook/esmfold_v1) or via "
            "esmatlas.com's web UI instead, then upload the PDB."
        )

    try:
        response = requests.post(
            ESMFOLD_API_URL,
            data=sequence,
            timeout=120
        )

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the ESMFold API: {e}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"ESMFold API returned status {response.status_code}. "
            "It may be temporarily down or rate-limited — try again "
            "in a moment, or fold locally / via esmatlas.com and "
            "upload the PDB instead."
        )

    pdb_text = response.text

    if not pdb_text.strip().startswith(("HEADER", "ATOM", "MODEL")):
        raise RuntimeError(
            "ESMFold API returned an unexpected response (not a PDB "
            "file). It may be temporarily down."
        )

    return pdb_text


# ---------------------------------------------------------
# PDB PARSING
# ---------------------------------------------------------

@st.cache_data
def parse_pdb(pdb_text):
    """
    Parse PDB text with BioPython.
    """

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "protein",
        io.StringIO(pdb_text)
    )

    return structure


def get_ca_atoms(structure):
    """
    Return CA atoms from the first model and first chain,
    in sequence order. This list's index (0-based) + 1 is the
    consistent "sequence position" used everywhere in this app —
    we never rely on PDB residue numbering (residue.id[1]), since
    that can be gapped, offset, or renumbered depending on the
    source of the PDB file.
    """

    model = next(structure.get_models())
    chain = next(model.get_chains())

    atoms = []

    for residue in chain.get_residues():
        if "CA" in residue:
            atoms.append(residue["CA"])

    return atoms


def calculate_rmsd(wt_structure, mutant_structure):
    """
    Align WT and mutant C-alpha atoms and calculate RMSD.
    """

    wt_atoms = get_ca_atoms(wt_structure)
    mutant_atoms = get_ca_atoms(mutant_structure)

    if len(wt_atoms) != len(mutant_atoms):
        st.warning(
            f"WT structure has {len(wt_atoms)} CA atoms but mutant has "
            f"{len(mutant_atoms)} — structures are not the same length. "
            "RMSD below is computed on a truncated, possibly misaligned "
            "subset and should be treated with caution."
        )

    n = min(len(wt_atoms), len(mutant_atoms))

    if n < 3:
        return None

    wt_atoms = wt_atoms[:n]
    mutant_atoms = mutant_atoms[:n]

    super_imposer = Superimposer()

    super_imposer.set_atoms(
        wt_atoms,
        mutant_atoms
    )

    return float(super_imposer.rms)


def get_ca_coordinates(structure):
    """
    Returns C-alpha coordinates.
    """

    atoms = get_ca_atoms(structure)

    return np.array([
        atom.get_coord()
        for atom in atoms
    ])


def calculate_local_rmsd(
    wt_structure,
    mutant_structure,
    mutation_position,
    window=5
):
    """
    Calculates RMSD around the mutation site.
    mutation_position is a 1-based sequence position (see get_ca_atoms).
    """

    if mutation_position is None:
        return None

    wt_atoms = get_ca_atoms(wt_structure)
    mutant_atoms = get_ca_atoms(mutant_structure)

    start = max(0, mutation_position - window - 1)
    end = min(
        len(wt_atoms),
        mutation_position + window
    )

    wt_local = wt_atoms[start:end]
    mutant_local = mutant_atoms[start:end]

    n = min(len(wt_local), len(mutant_local))

    if n < 3:
        return None

    super_imposer = Superimposer()

    super_imposer.set_atoms(
        wt_local[:n],
        mutant_local[:n]
    )

    return float(super_imposer.rms)


# ---------------------------------------------------------
# DISTANCE TO MUTATION
# ---------------------------------------------------------

def get_atom_at_seq_position(structure, position):
    """
    Gets the CA atom at a 1-based sequence position, using the same
    indexing scheme as get_ca_atoms / calculate_local_rmsd. This
    intentionally does NOT use residue.id[1] (PDB residue numbering),
    since that can disagree with sequence position across sources.
    """

    if position is None:
        return None

    atoms = get_ca_atoms(structure)
    idx = position - 1

    if 0 <= idx < len(atoms):
        return atoms[idx]

    return None


def calculate_mutation_displacement(
    wt_structure,
    mutant_structure,
    mutation_position
):
    """
    Calculates the displacement of the mutation residue's CA atom.
    """

    wt_atom = get_atom_at_seq_position(
        wt_structure,
        mutation_position
    )

    mutant_atom = get_atom_at_seq_position(
        mutant_structure,
        mutation_position
    )

    if wt_atom is None or mutant_atom is None:
        return None

    return float(
        np.linalg.norm(
            wt_atom.get_coord() -
            mutant_atom.get_coord()
        )
    )


# ---------------------------------------------------------
# STRUCTURE INFORMATION
# ---------------------------------------------------------

def get_structure_info(structure):

    atoms = list(structure.get_atoms())
    residues = list(structure.get_residues())

    return {
        "Residues": len(residues),
        "Atoms": len(atoms)
    }


def extract_plddt(pdb_text):
    """
    ESMFold stores confidence values in the B-factor column.
    Returns average value if available.
    """

    values = []

    for line in pdb_text.splitlines():

        if line.startswith("ATOM"):

            try:
                b_factor = float(line[60:66])
                values.append(b_factor)
            except ValueError:
                pass

    if not values:
        return None

    return float(np.mean(values))


# ---------------------------------------------------------
# ENVIRONMENTAL / EFFICIENCY HEURISTIC
# ---------------------------------------------------------

def _gaussian_factor(value, optimal, width):
    """Bell-curve falloff: 1.0 at the optimal value, decaying with
    distance from it. Used to model temperature/humidity/pH tolerance."""
    if width <= 0:
        return 1.0
    return float(np.exp(-((value - optimal) ** 2) / (2 * width ** 2)))


def estimate_relative_efficiency(
    wt_plddt,
    mutant_plddt,
    local_rmsd,
    mutation_displacement,
    temperature,
    humidity,
    ph,
    wt_optimal_temp,
    optimal_humidity,
    optimal_ph,
    temp_width=15.0,
    humidity_width=30.0,
    ph_width=2.0,
    thermal_shift_per_plddt_point=1.5,
    max_thermal_shift=15.0,
):
    """
    Produces a HEURISTIC, structure-derived "efficiency index" (0-100)
    for WT and mutant under the given conditions, and the percent
    difference between them.

    IMPORTANT: this is not a physical, kinetic, or thermodynamic
    simulation of enzyme activity. Real catalytic efficiency depends
    on transition-state energetics, active-site electrostatics, and
    dynamics that cannot be derived from a single static predicted
    structure. This function combines the structural signals this
    app actually has access to — pLDDT confidence, local backbone
    RMSD near the mutation, and mutation-site displacement — with a
    simple bell-curve model of environmental tolerance, to produce a
    directional, ranking-style score. Its purpose matches the
    hackathon brief's own framing: "guide and prioritize wet-lab
    validation" — a shortlist signal, not a validated efficiency
    prediction. Treat the output accordingly, and always calibrate
    the "assumed optimal" inputs against real literature/assay data
    where available.
    """

    if wt_plddt is None or mutant_plddt is None:
        return None

    plddt_delta = mutant_plddt - wt_plddt

    # Structural "intrinsic" penalty: how much the local geometry and
    # mutation-site environment changed. Larger local RMSD near the
    # mutation and large mutation-site displacement are treated as
    # riskier for the active site's geometry.
    geometry_penalty = 0.0

    if local_rmsd is not None:
        geometry_penalty += min(local_rmsd, 5.0) * 4.0  # up to 20 pts

    if mutation_displacement is not None:
        # mild displacement (~0-3 Å) isn't penalized; larger shifts are
        geometry_penalty += max(0.0, mutation_displacement - 3.0) * 3.0

    # Assumed thermal shift: a mutant predicted with higher confidence
    # (pLDDT) is assumed to tolerate a somewhat higher temperature
    # before unfolding — a common correlation in thermostability
    # engineering, but the magnitude here is an ADJUSTABLE ASSUMPTION,
    # not a measured value.
    thermal_shift = float(np.clip(
        plddt_delta * thermal_shift_per_plddt_point,
        -max_thermal_shift,
        max_thermal_shift,
    ))

    mutant_optimal_temp = wt_optimal_temp + thermal_shift

    # Humidity/pH tolerance applied identically to WT and mutant —
    # there's no structural signal here that differentiates them
    wt_env_factor = (
        _gaussian_factor(temperature, wt_optimal_temp, temp_width)
        * _gaussian_factor(humidity, optimal_humidity, humidity_width)
        * _gaussian_factor(ph, optimal_ph, ph_width)
    )

    mutant_env_factor = (
        _gaussian_factor(temperature, mutant_optimal_temp, temp_width)
        * _gaussian_factor(humidity, optimal_humidity, humidity_width)
        * _gaussian_factor(ph, optimal_ph, ph_width)
    )

    wt_index = max(0.0, min(100.0, 70.0 * wt_env_factor + 30.0))

    mutant_index = max(
        0.0,
        min(
            100.0,
            70.0 * mutant_env_factor + 30.0 - geometry_penalty + plddt_delta,
        ),
    )

    percent_change = (
        None if wt_index <= 0 else (mutant_index - wt_index) / wt_index * 100.0
    )

    return {
        "wt_index": wt_index,
        "mutant_index": mutant_index,
        "percent_change": percent_change,
        "thermal_shift": thermal_shift,
        "mutant_optimal_temp": mutant_optimal_temp,
    }


# ---------------------------------------------------------
# 3D VIEWER
# ---------------------------------------------------------

def display_structure(
    pdb_text,
    title,
    mutation_position=None,
    height=500
):

    st.markdown(f"### {title}")

    try:
        view = py3Dmol.view(
            width=700,
            height=height
        )

        view.addModel(
            pdb_text,
            "pdb"
        )

        # Protein cartoon
        view.setStyle({
            "cartoon": {
                "color": "spectrum"
            }
        })

        # Highlight mutation residue
        if mutation_position:

            view.setStyle(
                {
                    "resi": mutation_position
                },
                {
                    "cartoon": {
                        "color": "red"
                    },
                    "stick": {
                        "colorscheme": "greenCarbon",
                        "radius": 0.25
                    }
                }
            )

            view.addLabel(
                f"Mutation: residue {mutation_position}",
                {
                    "fontSize": 14,
                    "backgroundColor": "black",
                    "fontColor": "white"
                },
                {
                    "resi": mutation_position
                }
            )

        view.zoomTo()
        view.spin(False)

        html = view._make_html()

        components.html(
            html,
            height=height
        )

    except Exception as e:
        st.error(f"Could not render 3D structure for '{title}': {e}")


# ---------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------

st.sidebar.header("Demo")

st.sidebar.info(
    """
For the hackathon, you can upload predicted WT and mutant
PDB files.

This avoids repeatedly loading the large ESMFold model
during development.
"""
)

# ---------------------------------------------------------
# LOAD TOP CANDIDATE FROM biology_annotator.py
# ---------------------------------------------------------

# Fixed local path ten_mutations.json is written to by
# biology_annotator.py (run with --fasta). If it's there, load it
# automatically — no manual upload needed.
BASE_DIR = Path(__file__).resolve().parent.parent
NON_FINE_TUNED_DIR = BASE_DIR / "Models" / "Non_fine_tuned"
DEFAULT_TEN_MUTATIONS_PATH = NON_FINE_TUNED_DIR / "ten_mutations.json"
st.sidebar.header("Load top candidate")

ten_mutations_data = None
ten_mutations_source = None

if DEFAULT_TEN_MUTATIONS_PATH.exists():
    try:
        ten_mutations_data = json.loads(DEFAULT_TEN_MUTATIONS_PATH.read_text())
        ten_mutations_source = str(DEFAULT_TEN_MUTATIONS_PATH)
    except Exception as e:
        st.sidebar.error(
            f"Could not read {DEFAULT_TEN_MUTATIONS_PATH.name}: {e}"
        )
else:
    st.sidebar.caption(
        f"{DEFAULT_TEN_MUTATIONS_PATH.name} not found at the expected "
        "path — run biology_annotator.py with --fasta, or upload it "
        "manually below."
    )

ten_mutations_file = st.sidebar.file_uploader(
    "...or upload a different ten_mutations.json",
    type=["json"],
    key="ten_mutations_uploader",
)

if ten_mutations_file is not None:

    try:
        ten_mutations_data = json.loads(
            ten_mutations_file.read().decode("utf-8")
        )
        ten_mutations_source = ten_mutations_file.name
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        ten_mutations_data = None

if ten_mutations_data and ten_mutations_data.get("mutations"):

    st.sidebar.caption(f"Source: {ten_mutations_source}")

    candidate_labels = [
        f"{m['mutation']}  (score={m['composite_score']:+.2f})"
        for m in ten_mutations_data["mutations"]
    ]

    chosen_label = st.sidebar.selectbox(
        "Candidate mutation",
        candidate_labels,
        key="ten_mutations_choice",
    )

    chosen = ten_mutations_data["mutations"][
        candidate_labels.index(chosen_label)
    ]

    if chosen.get("rationale"):
        st.sidebar.caption(chosen["rationale"])

    if st.sidebar.button("⬇️🧬 Load & fold this candidate"):

        wt_sequence = ten_mutations_data["wt_sequence"]
        mutant_sequence = chosen["mutant_sequence"]

        st.session_state["wt_seq_input"] = wt_sequence
        st.session_state["mutant_seq_input"] = mutant_sequence

        reference_pdb_path_raw = ten_mutations_data.get("reference_pdb_path")
        wt_pdb_text = None

        if reference_pdb_path_raw:
            normalized = reference_pdb_path_raw.replace("\\", "/")
            reference_pdb_path = NON_FINE_TUNED_DIR / normalized

            if reference_pdb_path.exists():
            # Reuse the reference WT structure instead of re-folding it.
                wt_pdb_text = reference_pdb_path.read_text()

        try:
            if wt_pdb_text is None:
                with st.spinner(
                    f"Folding WT sequence for {chosen['mutation']} via ESMFold..."
                ):
                    wt_pdb_text = fold_sequence_esmfold(
                        clean_sequence(wt_sequence)
                    )

            with st.spinner(
                f"Folding mutant sequence {chosen['mutation']} via ESMFold..."
            ):
                mutant_pdb_text = fold_sequence_esmfold(
                    clean_sequence(mutant_sequence)
                )

            st.session_state["wt_pdb_text"] = wt_pdb_text
            st.session_state["mutant_pdb_text"] = mutant_pdb_text

            st.sidebar.success(
                f"Loaded and folded {chosen['mutation']} — scroll down and "
                "hit '🔬 Analyze WT vs Mutant' to compare."
            )

        except (ValueError, RuntimeError) as e:
            st.session_state["wt_pdb_text"] = wt_pdb_text
            st.session_state["mutant_pdb_text"] = None
            st.sidebar.error(f"Folding failed: {e}")

        st.rerun()

elif ten_mutations_data is not None:
    st.sidebar.warning("ten_mutations.json has no 'mutations' entries.")

# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------

st.header("1️⃣ Protein sequences")

col1, col2 = st.columns(2)

with col1:

    wt_seq = st.text_area(
        "Wild-Type PETase FASTA",
        height=180,
        placeholder="Paste WT FASTA sequence here...",
        key="wt_seq_input"
    )

with col2:

    mutant_seq = st.text_area(
        "Mutant PETase FASTA",
        height=180,
        placeholder="Paste mutant FASTA sequence here...",
        key="mutant_seq_input"
    )


# ---------------------------------------------------------
# PDB INPUT
# ---------------------------------------------------------

st.header("2️⃣ Structural models")

st.caption(
    "Either upload a PDB you already folded, or fold the pasted "
    "sequence above directly via the free ESMFold API. Folded "
    "structures are cached, so folding the same sequence twice "
    "(e.g. after tweaking something else) won't re-call the API."
)

if "wt_pdb_text" not in st.session_state:
    st.session_state["wt_pdb_text"] = None

if "mutant_pdb_text" not in st.session_state:
    st.session_state["mutant_pdb_text"] = None

pdb_col1, pdb_col2 = st.columns(2)

with pdb_col1:

    wt_file = st.file_uploader(
        "Upload Wild-Type PDB",
        type=["pdb"]
    )

    if st.button("🧬 Fold WT sequence via ESMFold"):

        try:
            wt_to_fold = clean_sequence(wt_seq)

            if not wt_to_fold:
                st.error("Paste a WT sequence above first.")
            else:
                with st.spinner("Folding WT sequence via ESMFold API..."):
                    st.session_state["wt_pdb_text"] = fold_sequence_esmfold(
                        wt_to_fold
                    )
                st.success("WT structure folded.")

        except (ValueError, RuntimeError) as e:
            st.error(str(e))

    if st.session_state["wt_pdb_text"]:
        st.caption("✅ WT structure available from ESMFold.")

with pdb_col2:

    mutant_file = st.file_uploader(
        "Upload Mutant PDB",
        type=["pdb"]
    )

    if st.button("🧬 Fold Mutant sequence via ESMFold"):

        try:
            mutant_to_fold = clean_sequence(mutant_seq)

            if not mutant_to_fold:
                st.error("Paste a mutant sequence above first.")
            else:
                with st.spinner("Folding mutant sequence via ESMFold API..."):
                    st.session_state["mutant_pdb_text"] = fold_sequence_esmfold(
                        mutant_to_fold
                    )
                st.success("Mutant structure folded.")

        except (ValueError, RuntimeError) as e:
            st.error(str(e))

    if st.session_state["mutant_pdb_text"]:
        st.caption("✅ Mutant structure available from ESMFold.")


# ---------------------------------------------------------
# ENVIRONMENTAL CONDITIONS
# ---------------------------------------------------------

st.header("3️⃣ Environmental conditions")

st.caption(
    "Set the conditions to evaluate under, and the WT's known or "
    "assumed optimum for each (ideally from literature or your own "
    "assay data). These feed a heuristic ranking score explained "
    "below the results — not a physical simulation of enzyme kinetics."
)

env_col1, env_col2, env_col3 = st.columns(3)

with env_col1:
    temperature = st.slider(
        "Temperature to evaluate at (°C)", 0, 100, 30
    )
    wt_optimal_temp = st.slider(
        "WT's known/assumed optimal temperature (°C)", 0, 100, 30
    )

with env_col2:
    humidity = st.slider(
        "Humidity to evaluate at (%)", 0, 100, 60
    )
    optimal_humidity = st.slider(
        "Assumed optimal humidity (%)", 0, 100, 60
    )

with env_col3:
    ph = st.slider(
        "pH to evaluate at", 1.0, 14.0, 8.0, step=0.1
    )
    optimal_ph = st.slider(
        "Assumed optimal pH", 1.0, 14.0, 8.0, step=0.1
    )


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

if st.button(
    "🔬 Analyze WT vs Mutant",
    type="primary"
):

    # Validate sequences
    try:

        wt_clean = clean_sequence(wt_seq)
        mutant_clean = clean_sequence(mutant_seq)

        if not wt_clean or not mutant_clean:
            st.error(
                "Please provide both WT and mutant sequences."
            )
            st.stop()

        mutation_string, mutations = get_mutation(
            wt_clean,
            mutant_clean
        )

        if not mutations:
            st.warning(
                "No sequence difference was detected."
            )
            st.stop()

    except Exception as e:

        st.error(f"Sequence error: {e}")
        st.stop()


    # -----------------------------------------------------
    # MUTATION
    # -----------------------------------------------------

    st.header("4️⃣ Mutation detected")

    st.success(
        f"Detected mutation: **{mutation_string}**"
    )

    if len(mutations) > 1:

        st.warning(
            "Multiple differences were detected. "
            "The structural metrics below use the first "
            "mutation position for mutation-site analysis."
        )

    mutation_position = mutations[0][0]

    wt_aa = mutations[0][1]
    mutant_aa = mutations[0][2]


    # -----------------------------------------------------
    # CHECK PDB FILES
    # -----------------------------------------------------

    wt_pdb = None
    mutant_pdb = None

    if wt_file is not None:
        wt_pdb = wt_file.read().decode("utf-8", errors="ignore")
    elif st.session_state["wt_pdb_text"]:
        wt_pdb = st.session_state["wt_pdb_text"]

    if mutant_file is not None:
        mutant_pdb = mutant_file.read().decode("utf-8", errors="ignore")
    elif st.session_state["mutant_pdb_text"]:
        mutant_pdb = st.session_state["mutant_pdb_text"]

    if wt_pdb is None or mutant_pdb is None:

        st.info(
            "Provide both WT and mutant structures — either upload "
            "PDB files above, or fold the pasted sequences via the "
            "ESMFold buttons — to run the structural comparison."
        )

        st.stop()


    # -----------------------------------------------------
    # PARSE
    # -----------------------------------------------------

    try:

        wt_structure = parse_pdb(wt_pdb)
        mutant_structure = parse_pdb(mutant_pdb)

    except Exception as e:

        st.error(
            f"Could not parse PDB files: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # SEQUENCE / STRUCTURE CONSISTENCY CHECK
    # -----------------------------------------------------

    wt_atom_count = len(get_ca_atoms(wt_structure))
    mutant_atom_count = len(get_ca_atoms(mutant_structure))

    if wt_atom_count != len(wt_clean):
        st.warning(
            f"WT PDB has {wt_atom_count} residues with CA atoms, but the "
            f"pasted WT sequence is {len(wt_clean)} residues long. "
            "The uploaded PDB may not match the pasted sequence — "
            "position-based metrics (mutation site, local RMSD, "
            "displacement) may be misaligned."
        )

    if mutant_atom_count != len(mutant_clean):
        st.warning(
            f"Mutant PDB has {mutant_atom_count} residues with CA atoms, "
            f"but the pasted mutant sequence is {len(mutant_clean)} "
            "residues long. The uploaded PDB may not match the pasted "
            "sequence — position-based metrics may be misaligned."
        )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    rmsd = calculate_rmsd(
        wt_structure,
        mutant_structure
    )

    local_rmsd = calculate_local_rmsd(
        wt_structure,
        mutant_structure,
        mutation_position
    )

    mutation_displacement = calculate_mutation_displacement(
        wt_structure,
        mutant_structure,
        mutation_position
    )

    wt_plddt = extract_plddt(wt_pdb)
    mutant_plddt = extract_plddt(mutant_pdb)


    # -----------------------------------------------------
    # METRIC DASHBOARD
    # -----------------------------------------------------

    st.header("5️⃣ Structural comparison")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        if rmsd is not None:

            st.metric(
                "Global RMSD",
                f"{rmsd:.2f} Å"
            )

        else:

            st.metric(
                "Global RMSD",
                "N/A"
            )


    with metric2:

        if local_rmsd is not None:

            st.metric(
                "Local RMSD",
                f"{local_rmsd:.2f} Å"
            )

        else:

            st.metric(
                "Local RMSD",
                "N/A"
            )


    with metric3:

        if mutation_displacement is not None:

            st.metric(
                "Mutation displacement",
                f"{mutation_displacement:.2f} Å"
            )

        else:

            st.metric(
                "Mutation displacement",
                "N/A"
            )


    with metric4:

        if wt_plddt is not None:

            st.metric(
                "WT mean pLDDT",
                f"{wt_plddt:.1f}"
            )

        else:

            st.metric(
                "WT mean pLDDT",
                "N/A"
            )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if mutant_plddt is not None:

        st.metric(
            "Mutant mean pLDDT",
            f"{mutant_plddt:.1f}"
        )


    # -----------------------------------------------------
    # EFFICIENCY ESTIMATE (HEURISTIC)
    # -----------------------------------------------------

    st.header("6️⃣ Estimated relative efficiency (heuristic)")

    st.warning(
        "⚠️ This is a STRUCTURAL HEURISTIC, not a measured or "
        "physically simulated catalytic efficiency. It combines "
        "pLDDT confidence, local RMSD near the mutation, and a "
        "simple bell-curve model of temperature/humidity/pH "
        "tolerance into a directional, ranking-style estimate — the "
        "kind of signal meant to help prioritize candidates for real "
        "wet-lab activity assays, not replace them. Treat the "
        "percentage as illustrative, not a validated prediction."
    )

    efficiency = estimate_relative_efficiency(
        wt_plddt,
        mutant_plddt,
        local_rmsd,
        mutation_displacement,
        temperature,
        humidity,
        ph,
        wt_optimal_temp,
        optimal_humidity,
        optimal_ph,
    )

    if efficiency is None:

        st.info(
            "Could not compute an efficiency estimate — pLDDT "
            "confidence values weren't found in one or both PDB "
            "files (this comes from the B-factor column in "
            "ESMFold-style output)."
        )

    else:

        eff_col1, eff_col2, eff_col3 = st.columns(3)

        with eff_col1:

            st.metric(
                "WT efficiency index",
                f"{efficiency['wt_index']:.1f} / 100"
            )

        with eff_col2:

            st.metric(
                "Mutant efficiency index",
                f"{efficiency['mutant_index']:.1f} / 100"
            )

        with eff_col3:

            pct = efficiency["percent_change"]

            if pct is None:

                st.metric("Relative change", "N/A")

            else:

                direction = "more" if pct >= 0 else "less"

                st.metric(
                    "Relative change",
                    f"{pct:+.1f}%",
                    delta=f"{abs(pct):.1f}% {direction} efficient than WT"
                )

        st.caption(
            f"Model assumes the mutant's effective optimal temperature "
            f"shifts by {efficiency['thermal_shift']:+.1f}°C relative "
            f"to WT (estimated optimum ≈ "
            f"{efficiency['mutant_optimal_temp']:.1f}°C), based on its "
            "structural confidence delta relative to WT. Adjust the "
            "'assumed optimal' sliders above to match real literature "
            "or assay values for a more meaningful comparison."
        )


    # -----------------------------------------------------
    # 3D VIEWERS
    # -----------------------------------------------------

    st.header("7️⃣ 3D structure visualization")

    view_tab, overlay_tab = st.tabs(["Side by side", "Overlay"])

    with view_tab:

        view_col1, view_col2 = st.columns(2)

        with view_col1:

            display_structure(
                wt_pdb,
                "Wild Type",
                mutation_position
            )

        with view_col2:

            display_structure(
                mutant_pdb,
                "Mutant",
                mutation_position
            )

    with overlay_tab:

        # -----------------------------------------------------
        # OVERLAY VIEW
        # -----------------------------------------------------

        st.header("WT vs Mutant overlay")

        try:
            overlay = py3Dmol.view(
                width=1000,
                height=600
            )

            overlay.addModel(
                wt_pdb,
                "pdb"
            )

            overlay.setStyle(
                {
                    "model": 0
                },
                {
                    "cartoon": {
                        "color": "blue"
                    }
                }
            )

            overlay.addModel(
                mutant_pdb,
                "pdb"
            )

            overlay.setStyle(
                {
                    "model": 1
                },
                {
                    "cartoon": {
                        "color": "orange"
                    }
                }
            )

            # Highlight mutation in both structures

            overlay.setStyle(
                {
                    "model": 0,
                    "resi": mutation_position
                },
                {
                    "stick": {
                        "colorscheme": "blueCarbon",
                        "radius": 0.3
                    }
                }
            )

            overlay.setStyle(
                {
                    "model": 1,
                    "resi": mutation_position
                },
                {
                    "stick": {
                        "colorscheme": "orangeCarbon",
                        "radius": 0.3
                    }
                }
            )

            overlay.zoomTo()

            components.html(
                overlay._make_html(),
                height=620
            )

        except Exception as e:
            st.error(f"Could not render overlay: {e}")


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.header("8️⃣ Mutation summary")

    if len(mutations) > 1:
        st.caption(
            f"⚠️ {len(mutations)} sequence differences detected "
            f"({mutation_string}). Structural metrics below reflect "
            f"only the first mutation ({mutation_string.split(',')[0].strip()})."
        )

    summary = {
        "Mutation": mutation_string,
        "WT amino acid": wt_aa,
        "Mutant amino acid": mutant_aa,
        "Position": mutation_position,
        "Global RMSD": (
            f"{rmsd:.2f} Å"
            if rmsd is not None
            else "N/A"
        ),
        "Local RMSD": (
            f"{local_rmsd:.2f} Å"
            if local_rmsd is not None
            else "N/A"
        ),
        "Mutation displacement": (
            f"{mutation_displacement:.2f} Å"
            if mutation_displacement is not None
            else "N/A"
        ),
        "WT mean pLDDT": (
            f"{wt_plddt:.1f}"
            if wt_plddt is not None
            else "N/A"
        ),
        "Mutant mean pLDDT": (
            f"{mutant_plddt:.1f}"
            if mutant_plddt is not None
            else "N/A"
        ),
        "Evaluated at": f"{temperature}°C, {humidity}% humidity, pH {ph}",
        "Relative efficiency (heuristic)": (
            f"{efficiency['percent_change']:+.1f}%"
            if efficiency is not None and efficiency["percent_change"] is not None
            else "N/A"
        )
    }

    st.table(summary)


    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    st.header("9️⃣ Structural interpretation")

    if rmsd is not None:

        if rmsd < 1.0:

            st.info(
                "The WT and mutant structures show a relatively "
                "small global structural deviation."
            )

        elif rmsd < 2.0:

            st.info(
                "The mutation produces a moderate structural "
                "difference between the predicted structures."
            )

        else:

            st.warning(
                "The predicted structures show a relatively "
                "large global structural deviation. This should "
                "be investigated before experimental validation."
            )

    st.caption(
        "Structural metrics are computational indicators and "
        "should not be interpreted as experimental proof of "
        "improved catalytic activity or stability."
    )
