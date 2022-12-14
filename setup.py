from setuptools import setup

requirements = [
    "numpy>=1.22.4",
    "gpflow>=2.6.3",
    "jax>=0.3.13",
    "pandas>=1.5.1",
    "plotly>=5.11.0",
    "matplotlib>=3.5.2",
    "scipy>=1.8.1",
    "setuptools>=59.6.0",
]

setup(name='searchep',
      version='0.1',
      description='A module to search for exceptional points',  # url='http://ictshore.com/',
      author='Patrick Egenlauf',
      author_email='p.egenlauf@gmx.de',  # license='MIT',
      packages=['searchep'],
      install_requires=requirements,
      zip_safe=False)
