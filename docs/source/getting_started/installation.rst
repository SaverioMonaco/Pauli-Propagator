Installation
------------

Propagation runs in a Rust extension, so installation is two steps: the
Python package, then building that extension.

.. code-block:: bash

    pip install .

    pip install maturin
    cd native/pprop_rs
    maturin develop --release

The last command needs a Rust toolchain (`rustup <https://rustup.rs/>`_).
``maturin develop`` builds ``pprop_rs`` and installs it straight into the
active virtual environment; re-run it after pulling changes to
``native/pprop_rs``.

Running the tutorial notebooks requires the ``notebooks`` extra:

.. code-block:: bash

    pip install -e ".[notebooks]"
