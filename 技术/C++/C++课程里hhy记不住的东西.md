# 关于常量
```c++
#include <iostream>
using namespace std;

int main() {
	cout << typeid(2.0f).name() << endl;
	cout << typeid(2.0).name() << endl;
	cout << typeid(2l).name() << endl;
	cout << typeid(2).name() << endl; 
  return 0;
}

/* output:
float
double
long
int
*/
```

注意
关于单引号和双引号的问题
单引号是字符，双引号是字符串
```c++
#include <iostream>
using namespace std;

int main() {
    char a = 'l';
    cout << a << endl;
    string b = "hello";
    cout << b << endl;
  return 0;
}

output:
l
hello
```
但是下面是不行的：
```c++
#include <iostream>
using namespace std;

int main() {
    string b = 'hello'; ->这个单引号很坏
    cout << b << endl;
  return 0;
}

output:
“初始化”: 无法从“int”转换为“std::basic_string<char,std::char_traits<char>,std::allocator<char>>”
```

比较特殊的数字写法
```c++
#include <iostream>
using namespace std;

int main() {
	cout << .145 << endl;
	cout << 2.145E-1 << endl;
	cout << 2.145E2 << endl;
  return 0;
}

output:
0.145
0.2145
214.5
```

## 关于自增运算
```c++
#include <iostream>
using namespace std;

int main() {
	int a = 1;
	int b = (a++) * 2; -> 先自增，后乘法

	int c = 1;
	int d = (++c) * 2; -> 先乘法，后自增

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
	cout << "c = " << c << endl;
	cout << "d = " << d << endl;
  return 0;
}

output:
a = 2
b = 2

c = 2
d = 4
```
注意区别，
- 如果++在后面，那么会先做自增运算，然后做别的运算（也就是先自增，然后用自增之后的数字×2）
- 反之先做别的运算然后自增

## 关于关系运算
1. 以下写法在C++中是被允许的