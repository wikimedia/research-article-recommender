import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="article_recommender-bmansurov",
    version="0.0.2.1",
    author="Bahodir Mansurov",
    author_email="bmansurov@wikimedia.org",
    description="article normalized scores generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gerrit.wikimedia.org/r/#/admin/projects/research/article-recommender",
    packages=['article_recommender'],
    package_dir={'article_recommender': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
)
