native/pprop_rs (Rust extension)
================================

Heisenberg propagation (:meth:`~pprop.propagator.Propagator.propagate`) runs
in a Rust extension, ``pprop_rs``, built from ``native/pprop_rs/`` with
`maturin <https://www.maturin.rs/>`_. It is a separate package from
``pprop`` itself, published to PyPI as ``pprop-rs`` and installed
automatically as a dependency. In a development checkout, build it by
running ``maturin develop --release`` from ``native/pprop_rs/``.

Since it is a compiled extension with no Python source for Sphinx to
introspect, it is not covered by autodoc here. Users never call it directly:
:class:`~pprop.propagator.Propagator` does. It has two engines:

- **tape** (the default): :meth:`~pprop.propagator.Propagator.propagate`
  calls ``pprop_rs.compile_tapes()``, which compiles each observable once
  into a ``TapeProgram``, a list of steps on numbered boxes, one box per
  Pauli word. ``prop(params)`` and ``prop.eval_and_grad(params)`` run those
  steps (``TapeProgram.eval`` and ``TapeProgram.eval_and_grad``).
- **terms** (the original engine): ``pprop_rs.propagate_batch()`` keeps every
  Pauli path as its own term, a coefficient times a product of sines and
  cosines. It is only used to build ``prop.exprs``, the readable expression,
  on request.

``native/pprop_rs/README.md`` maps the crate's modules, says which file runs
for each Python call, and lists the places to change when adding a gate.
