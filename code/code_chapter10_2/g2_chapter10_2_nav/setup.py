from glob import glob
import os
from setuptools import find_packages, setup

package_name = "g2_chapter10_2_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="G2 Robot Course",
    maintainer_email="student@example.com",
    description="G2 Chapter 10-2 Nav2 teaching package",
    license="Apache-2.0",
    entry_points={"console_scripts": ["send_goal = g2_chapter10_2_nav.send_goal:main"]},
)
