from setuptools import setup, find_packages

setup(
    name='HydroToolkit',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'seaborn',
        'requests',
        'pymannkendall',
        'os',
        'typing',
        'random',
        're',
        'functools',
        'wqchartpy',
        'sklearn'
    ],

    author='Paul Töchterle',
    author_email='paul.m.teochterle@gmail.com',
    description='A package for hydrogeological timeseries data analysis and plotting',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/paultoechterle/HydroToolkit.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='MIT',
    keywords='hydrogeology timeseries data analysis plotting',
    include_package_data=True,  # Include package data as specified in MANIFEST.in
)