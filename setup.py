from setuptools import setup

try:
    from pyqt_distutils.build_ui import build_ui
    cmdclass = {"build_ui": build_ui}
except:
    cmdclass = {}

setup(
    name = "rfmatch_tool",
    author = "Maarten Lont",
    author_email = "maartenlont@gmail.com",
    license="GNU General Public License (GPL)",
    version = "0.1",
    packages = ["rfmatch_tool"],
    cmdclass=cmdclass,
)
