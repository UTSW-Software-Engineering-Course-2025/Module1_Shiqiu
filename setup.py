from setuptools import setup, find_packages

setup(
    name='Module1_shiqiu',
    version='1.0.0',
    description='Your project description here',
    author='Shiqiu Yu',
    author_email='shiqiu.yu2@utsouthwestern.com',
    packages=find_packages(where="cookie"),
    package_dir={"": "cookie"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "panda"
    ],
    python_requires='>=3.7',
)
