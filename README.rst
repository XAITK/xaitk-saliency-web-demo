===================
XAITK Saliency Demo
===================

Web application demonstrating XAITK Saliency functionality

* Free software: BSD License


Installing
----------

It is recommended to use conda to properly install the various ML packages.

macOS conda setup
^^^^^^^^^^^^^^^^^

.. code-block:: console

    brew install miniforge
    conda init zsh

venv creation for AI
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    conda create --name xaitk python=3.9
    conda activate xaitk
    conda install "pytorch==1.9.1" "torchvision==0.10.1" -c pytorch
    conda install scipy "scikit-learn==0.24.2" "scikit-image==0.18.3" -c conda-forge
    pip install -e .


Run the application

.. code-block:: console

    conda activate xaitk
    xaitk_saliency_demo


|image_1| |image_2| |image_3| |image_4|

.. |image_1| image:: gallery/xaitk-classification-rise-4.jpg
  :width: 20%
.. |image_2| image:: gallery/xaitk-classification-sliding-window.jpg
  :width: 20%
.. |image_3| image:: gallery/xaitk-detection-retina.jpg
  :width: 20%
.. |image_4| image:: gallery/xaitk-similarity-1.jpg
  :width: 20%
