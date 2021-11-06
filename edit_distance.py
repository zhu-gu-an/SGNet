#-*- coding:utf-8 -*-
# author:zhuxuechao

'''
这个函数是为评价预测的准确率做准备，采用的编辑距离计算预测值和真实值之间的距离
'''

import difflib

def get_edit_distance(str1 , str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += j2 - j1
        elif tag == 'delete':
            leven_cost += i2 - i1
    return leven_cost


if __name__ == '__main__':
    print(get_edit_distance('ABD', 'DFG'))
"""
s=difflib.SequenceMatcher(isjunk=None,a,b,autojunk=True)：构造函数，主要创建任何类型序列的比较对象。
isjunk是关键字参数，主要设置过滤函数，如想丢掉a和b比较序列里特定的字符，就可以设置相应的函数

s.get_opcodes()函数每执行一次返回5个元素的元组，元组描述了a和b比较序列的相同不同处。5个元素的元组表示为(tag, i1, i2, j1, j2)，其中tag表示动作，i1表示序列a的开始位置，i2表示序列a的结束位置，j1表示序列b的开始位置，j2表示序列b的结束位置。
tag表示的字符串为：
replace表示a[i1:i2]将要被b[j1:j2]替换。
delete表示a[i1:i2]将要被删除。
insert表示b[j1:j2]将被插入到a[i1:i1]地方。
equal表示a[i1:i2]==b[j1:j2]相同。

string.maketrans(instr,outstr)返回一个翻译表,instr中的字符是需要被outstr中的字符替换，而且instr和outstr的长度必须相等
str.maketrans(intab, outtab)方法用于创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。两个字符串的长度必须相同，为一一对应的关系。

"""