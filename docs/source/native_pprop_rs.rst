native/pprop_rs (Rust extension)
================================

Heisenberg propagation (:meth:`~pprop.propagator.Propagator.propagate`) runs
in a Rust extension, ``pprop_rs``, built from ``native/pprop_rs/`` with
`maturin <https://www.maturin.rs/>`_. It is a separate package from
``pprop`` itself and is not built by a plain ``pip install -e .``; build it
by running ``maturin develop --release`` from ``native/pprop_rs/``.

Since it is a compiled extension with no Python source for Sphinx to
introspect, it is not covered by autodoc here. Its public interface is one
function, ``pprop_rs.propagate_batch()``, called internally by
:meth:`~pprop.propagator.Propagator.propagate`; most users will never call
it directly.
