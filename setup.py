import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hhrnet-uyoung",
    version="0.1",
    author="Uyoung Jeong",
    author_email="jeong.uyoung@unist.ac.kr",
    description="custom modification of higherhrnet",
    url="https://github.com/uyoung-jeong/HigherHRNet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
