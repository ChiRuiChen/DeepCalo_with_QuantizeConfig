import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='deepcalo',
    version='0.2.2',
    author='Frederik Faye',
    author_email='frederikfaye@gmail.com',
    description='Package for doing deep supervised learning on ATLAS data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/ffaye/deepcalo',
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.15.4',
                      'keras>=2.2.4',
                      'h5py>=2.8.0',
                      'joblib>=0.13.0',
                      'keras-drop-block>=0.4.0'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
