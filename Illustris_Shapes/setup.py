from setuptools import setup

setup(
    name='Illustris_Shapes',
    version='1.0.0',
    packages=["Illustris_Shapes"],
    install_requires=["numpy", "astropy", "scipy", "halotools", "tqdm", "illustris_python", "inertia_tensors"],
    tests_require=["nose","coverage"],
)
