***********
Quick Start
***********

``PyADTS`` provides a "full-stack" framework to build time-series anomaly detection workflows.

=================
Loading Datasets
=================

.. code-block:: python

    from pyadts.datasets import SMD

    data = SMD(root='data/smd', download=True)


==============
Preprocessing
==============

.. code-block:: python



=================
Model Definition
=================

=========
Training
=========

===========
Evaluation
===========

====================
Ensemble (optional)
====================

=============================
Score Calibration (optional)
=============================
