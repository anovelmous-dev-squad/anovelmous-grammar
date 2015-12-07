from setuptools import setup

setup(
    name='anovelmous-grammar',
    version='0.2.0',
    packages=['grammar'],
    url='https://github.com/anovelmous-dev-squad/anovelmous-grammar',
    license='MIT',
    author='Greg Ziegan',
    author_email='greg.ziegan@gmail.com',
    description='Grammar module for the anovelmous collaborative writing app.',
    install_requires=[
        'nltk>=3.0',
        'numpy>=1.9',
        'scipy>=0.16'
    ]
)
