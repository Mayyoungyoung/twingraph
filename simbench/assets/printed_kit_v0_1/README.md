Original user-provided print geometry, imported from
`F:\RAL\3D\TwinGraph_print_kit_v0_1\TwinGraph_print_kit_v0_1\stl\htzp`.

The seven STL files are byte-identical copies. The `_2` pin and holder are byte-identical
duplicates instantiated twice in the scene. `manifest.json` pins their SHA256 hashes.
STL units are mm, with pin head facing down for printing; the simulator applies
0.001 scale and explicitly reverses that print orientation.

`provenance/build_models.py`, `parameters.json`, and `geometry_checks.json` were copied
from the same supplied print package. Running the CAD source is unnecessary for simulation.

Full STL meshes render the original shape. Collision uses the original CAD union of
boxes and cylinders, plus individually convex ring sectors made from original STL vertices.
This preserves all concave channels and holes; full-STL convex collision is disabled.

Mass is integrated over the source solid with an explicitly assumed effective density
of 700 kg/m3. This is not measured material/infill calibration. The wipe pad is a separate
36 x 24 x 8 mm foam accessory specified by the print-kit source, assumed to weigh 8 g.
