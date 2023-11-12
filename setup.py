from setuptools import find_packages, setup

setup(name="vexpr",
      version="0.1.0",
      description="Represent code expressions as data structures, then transform them",
      author="Marcus Lewis",
      url="https://vexpr.org/",
      packages=find_packages(),
      package_data={'vexpr': ['vexpr/package_data/*',]},
      include_package_data=True,
      zip_safe=False,
)
