from setuptools import setup, find_packages

setup(
    name="carag",  
    version="1.0.7",  
    author="Mohamed Rizwan",
    author_email="rizdelhi@gmail.com",
    description="An efficient python library for building AI applications using the Retrieval-Augmented Generation (RAG) pipeline.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rizwandel/Build-Standard-RAG-with-Qdrant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version
    install_requires=[
        "PyMuPDF==1.25.3",
        "fastembed==0.6.0",
        "qdrant-client==1.10.0", 
        "mistralai==1.6.0",
        "python-dotenv==1.1.0",
        "ipywidgets==8.1.6"
    ]
)
