import os

from devtoolbox.markdown.converter import MarkdownConverter

def test_markdown_to_word():
    """
    测试将 Markdown 转换为 Word 文档的示例程序
    """
    # 创建示例 Markdown 内容
    markdown_content = """# 测试文档

## 第一部分
这是一个测试文档，用于演示 MarkdownHandler 的功能。

## 第二部分
* 项目1
* 项目2
* 项目3

### 子标题
这是一些示例文本。
"""

    try:
        # 使用内容字符串创建 MarkdownConverter 实例
        converter = MarkdownConverter(content=markdown_content)

        # 设置输出文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, 'output.docx')

        # 转换为 Word 文档
        converter.to_docx(output_path)
        print(f"成功将 Markdown 转换为 Word 文档！\n输出文件: {output_path}")

    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")

if __name__ == "__main__":
    test_markdown_to_word()