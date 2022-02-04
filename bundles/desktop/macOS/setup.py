from setuptools import setup

ENTRY_POINT = ['xaitk-saliency-demo']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'strip': True,
    'iconfile': 'xaitk.icns',
    'includes': ['WebKit', 'Foundation', 'setuptools']
}

setup(
    app=ENTRY_POINT,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)