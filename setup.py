from setuptools import setup, find_packages

setup(
    name="carag",  # Replace with your package name
    version="1.0.6",  # Initial version
    author="Mohamed Rizwan",
    author_email="rizdelhi@gmail.com",
    description="A Python Package for RAG",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rizwandel/New-RAG",
    packages=find_packages(),  # Automatically finds Python modules
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version
    install_requires=[
        "PyMuPDF==1.25.3", # PDF Document loader
        "fastembed==0.6.0",
        "qdrant-client==1.10.0", # Qdrant vector database
        "mistralai==1.6.0",
        "python-dotenv==1.1.0"
    ],  # List dependencies (if any)
)