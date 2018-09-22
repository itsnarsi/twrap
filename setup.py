# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-19T11:54:03-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   cibitaw1
# @Last modified time: 2018-09-22T17:27:21-05:00
from setuptools import setup

setup(name='twrap',
      version='0.0.1',
      description='A pyTorch Wrapper.',
      url='https://github.com/itsnarsi/twrap',
      author='Narsi Reddy',
      author_email='sainarsireddy@outlook.com',
      license='MIT',
      packages=['twrap'],
      install_requires=['terminaltables',
      'torch'],
      zip_safe=False)
