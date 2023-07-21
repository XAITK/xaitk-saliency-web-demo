===================
XAITK Saliency Demo
===================

XAITK Saliency functionality demonstration via an interactive Web UI that can run locally, in jupyter or in the cloud.
A docker image is available for cloud deployment or local testing (``kitware/trame:xaitk``).
This application has been mainly tested on Linux but should work everywhere assuming all the dependencies manage to install on your system.

|image_1| |image_2| |image_3| |image_4|

.. |image_1| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/master/gallery/xaitk-classification-rise-4.jpg
  :width: 20%
.. |image_2| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/master/gallery/xaitk-classification-sliding-window.jpg
  :width: 20%
.. |image_3| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/master/gallery/xaitk-detection-retina.jpg
  :width: 20%
.. |image_4| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/master/gallery/xaitk-similarity-1.jpg
  :width: 20%


License
-------

This application is provided under the BSD Open Source License.


Installing
----------

On Linux, ``pip`` is enough to install this Python library.

.. code-block:: console

    python3.9 -m venv .venv
    source .venv/bin/activate
    pip install -U pip xaitk-saliency-demo

Then from that virtual-environment you can run the application like below

.. code-block:: console

    # Executable
    xaitk-saliency-demo

    # Module approach
    python -m xaitk_saliency_demo.app

Within Jupyter you can do the following

.. code-block:: console

    from xaitk_saliency_demo.app.jupyter import show
    show()




Docker image
------------

For each commit to master the CI will push a new **kitware/trame:xaitk** image to dockerhub.
Such image can be ran locally using the following command line.

.. code-block:: console

    docker run -it --rm --gpus all -p 8080:80 kitware/trame:xaitk
