from setuptools import setup

setup(
    name='anovelmous-grammar',
    version='0.1.1',
    packages=['grammar'],
    url='',
    license='MIT',
    author='Greg Ziegan',
    author_email='greg.ziegan@gmail.com',
    description='Grammar module for the anovelmous collaborative writing app.',
    install_requires=[
        'nltk>=3.0',
        'numpy>=1.9'
    ]
)
