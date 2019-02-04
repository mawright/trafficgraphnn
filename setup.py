from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='trafficgraphnn',
    version='0.1',
    packages=['trafficgraphnn'],
    install_requires=requirements,
    url='',
    license='',
    author='Matthew A. Wright',
    author_email='mwright@berkeley.edu',
    description=[
        'Research code for road traffic network learning with neural '
        'networks for graph-structured data']
)
