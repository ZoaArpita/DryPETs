# petase_structure.pdb — not included

This sandbox has no network access to RCSB PDB, so a real coordinate file
can't be fetched or (obviously) fabricated here. `biology_annotator.py` is
written to degrade gracefully without it — it just skips the 3D-distance and
SASA signals and falls back to sequence-position lookups only.

To get full structural annotation working, download one of these to this
folder as `petase_structure.pdb`:

- **PDB 6QGC** — wild-type IsPETase, matches the exact numbering used in
  `petase_catalytic_residues.json` and `known_hotspots.json` (this is the
  structure your team's `structure_view.py` / ESMFold pipeline already
  references).
  https://www.rcsb.org/structure/6QGC
- **PDB 5XG0** — alternative wild-type IsPETase crystal structure, same
  numbering convention, also commonly cited in the catalytic-triad literature.
  https://www.rcsb.org/structure/5XG0

Either works. Once downloaded:

```bash
curl -o petase_structure.pdb https://files.rcsb.org/download/6QGC.pdb
```

Then `BiologyAnnotator` will automatically pick it up (see
`StructuralAnnotator` in `biology_annotator.py`) and compute:
- Cα distance from each candidate mutation position to the nearest catalytic
  triad / oxyanion-hole residue
- Per-residue solvent-accessible surface area (SASA), cached to
  `sasa_cache.json` in this folder after the first run so it isn't
  recomputed every pass
