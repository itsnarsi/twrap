# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-19T11:54:03-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   cibitaw1
# @Last modified time: 2018-09-22T17:55:51-05:00
from setuptools import setup
from setuptools import find_packages
setup(name='twrap',
      version='0.1.0',
      description='A pyTorch Wrapper.',
      url='https://github.com/itsnarsi/twrap',
      author='Narsi Reddy',
      author_email='sainarsireddy@outlook.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['terminaltables',
      'torch>=0.4.0'],
      zip_safe=False)
