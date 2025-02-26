import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='mlyzed',  
     version='0.2',
     py_modules = ["mlyzed"],
     install_requires = [
                         "numpy",
                         "scipy",
                         "ase",
                         ],
     author="Artem Dembitskiy",
     author_email="art.dembitskiy@gmail.com",
     description="Molecular dynamics post-processing toolbox",
     key_words = ['MD', 'trajectory', 'diffusion', 'conductivity', 'unwrapping'],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dembart/mlyzed",
     package_data={"mlyzed": ["*.txt", "*.rst", '*.md'], 
                    #'tests':['*'], 
                    },
     classifiers=[
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "Programming Language :: Python :: 3.11",
         "Programming Language :: Python :: 3.12",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    include_package_data=True,
    #package_dir={'': 'mlyzed'},
    packages=setuptools.find_packages(),
 )


