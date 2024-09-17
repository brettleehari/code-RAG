from setuptools import setup, find_packages

setup(
    name='rag_app',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'llama-index==0.11.8'
    ],
)