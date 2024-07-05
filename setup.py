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
        'wqchartpy',
        'scikit-learn'
    ],
    include_package_data=True,
    package_data={
        'HydroToolkit': ['data/*.csv', 'data/*.xlsx', 'style/*.mplstyle'],
    },
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