from setuptools import setup

setup(
    name="article-recommender",
    version="0.0.2",
    author="Bahodir Mansurov",
    author_email="bmansurov@wikimedia.org",
    description="Recommend Wikipedia articles for creation",
    url="https://gerrit.wikimedia.org/r/#/admin/projects/research/article-recommender",
    packages=['article_recommender'],
    package_data={
        '': ['data/*'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
)
