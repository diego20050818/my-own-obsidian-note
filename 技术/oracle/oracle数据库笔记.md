---
tags:
  - 基础知识
  - oracle
  - 专业课
---
# SQL语句

## 数据库定义语言 DDL
```SQL
-- 创建表
CREATE TABLE 表名 (
    列名1 数据类型 [约束],
    列名2 数据类型 [约束],
    ...
);

-- 创建视图
CREATE VIEW 视图名 AS
SELECT 语句;

-- 创建索引
CREATE INDEX 索引名 ON 表名(列名);

-- 删除表/视图/索引
DROP TABLE 表名;
DROP VIEW 视图名;
DROP INDEX 索引名;

-- 修改表结构
ALTER TABLE 表名
ADD 列名 数据类型;
ALTER TABLE 表名
MODIFY 列名 新数据类型;
ALTER TABLE 表名
DROP COLUMN 列名;

```

## 数据库查询语言 DQL
```SQL
-- 基本查询结构
SELECT 列1, 列2, ...
FROM 表名
WHERE 条件
GROUP BY 分组列
HAVING 分组条件
ORDER BY 排序列;
```

## 数据库操作语言 DML

```SQL
-- 插入数据
INSERT INTO 表名 (列1, 列2, ...)
VALUES (值1, 值2, ...);

-- 更新数据
UPDATE 表名
SET 列1 = 值1, 列2 = 值2, ...
WHERE 条件;

-- 删除数据
DELETE FROM 表名
WHERE 条件;
```

---

# PLSQL
是在oracle在标准sql语言上的扩展，
可以嵌入sql语言，还可以定义变量和常量
允许使用条件语句和循环语句，允许使用例外来处理各种错误

plsql将sql语句有机结合起来，变成一个完整的功能

## 块编程

```SQL
declare -> 可选
	变量定义
	光标定义

begin -> 一定要有
	程序体
	exception
	例外处理
end;
一般的变量定义为 v_开头
```
