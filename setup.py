import setuptools

with open("README.org", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="article-recommender-bmansurov",
    version="0.0.1",
    author="Bahodir Mansurov",
    author_email="bmansurov@wikimedia.org",
    description="article normalized scores generator",
    long_description=long_description,
    long_description_content_type="text/x-org",
    url="https://gerrit.wikimedia.org/r/#/admin/projects/research/article-recommender",
    packages=['article-recommender'],
    package_dir={'article-recommender': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
)
