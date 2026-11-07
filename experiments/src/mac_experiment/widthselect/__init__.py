"""MAC bit-width selection experiment (W_align / W_acc).

Independent schema, run IDs and result roots.  See
``docs/mac_width_selection.md`` for the full semantics, the verified W_align
definition and the W_acc custom-format proposal (pending user confirmation).

Modules import lazily so offline analysis never imports the execution engine or
the sampler.
"""

WIDTHSELECT_PACKAGE = "mac_experiment.widthselect"
