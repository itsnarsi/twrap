# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-19T11:53:44-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   cibitaw1
# @Last modified time: 2018-09-19T11:59:06-05:00
import torch
print(torch.__version__)
if '0.4.' not in torch.__version__:
    raise Exception('Only works currently with PyTorch ver0.4.x')
