Publishing to PyPI
-------------------

This assume you have **twine** available within your python environment and updated the package version inside **setup.cfg**

.. code-block:: console

    rm -rf dist build

    python setup.py sdist bdist_wheel
    twine check dist/*
    twine upload dist/*
