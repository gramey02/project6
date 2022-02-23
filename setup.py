from setuptools import find_packages
from setuptools import setup

setup(
    name= 'regression',
    version= '0.0.1',
    author= 'Grace Ramey',
    author_email= 'Grace.Ramey@ucsf.edu',
    packages= find_packages(),
    description= 'Trains a logistic regression model complete with gradient descent',
	install_requires= ['pandas', 'numpy', 'matplotlib', 'scikit-learn']
)