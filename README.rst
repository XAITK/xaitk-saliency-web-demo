===================
XAITK Saliency Demo
===================

Web application demonstrating XAITK Saliency functionality

* Free software: BSD License
* Created using template from `trame-cookiecutter <https://github.com/Kitware/trame-cookiecutter>`_


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

    conda create --name xaitk python=3.9 -y
    conda activate xaitk

    # For development when inside repo
    pip install -e .

    # For testing (no need to clone repo)
    pip install xaitk-saliency-demo

Run the application

.. code-block:: console

    conda activate xaitk
    xaitk-saliency-demo


|image_1| |image_2| |image_3| |image_4|

.. |image_1| image:: gallery/xaitk-classification-rise-4.jpg
  :width: 20%
.. |image_2| image:: gallery/xaitk-classification-sliding-window.jpg
  :width: 20%
.. |image_3| image:: gallery/xaitk-detection-retina.jpg
  :width: 20%
.. |image_4| image:: gallery/xaitk-similarity-1.jpg
  :width: 20%


Contribute
----------

Commit messages needs to follow `semantic-release <https://github.com/semantic-release/semantic-release>`_ expectation.

- **fix(scope): summary** will trigger a +0.0.1 version
- **feat(scope): summary** will trigger a +0.1.0 version bump
- **ci/chore/docs** will not trigger a release

Then to move to +1.0.0 you need to add a **BREAKING CHANGE: xyz** after the body of the commit message.  

Docker image
------------

For each commit to master the CI will push a new **kitware/trame:xaitk** image to dockerhub.
Such image can be ran locally using the following command line.

.. code-block:: console

    docker run -it --rm --gpus all -p 8080:80 kitware/trame:xaitk
