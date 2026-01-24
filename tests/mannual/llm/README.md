# Gemini Provider 测试脚本

## 测试脚本说明

### 1. 基础功能测试
**文件**: `test_gemini_provider.py`

测试 Gemini provider 的基础功能，包括：
- 基础聊天
- JSON 模式
- Complete 方法
- List models
- Embed 方法（验证 NotImplementedError）

**运行方式**:
```bash
export GOOGLE_API_KEY="your-api-key"
python tests/mannual/llm/test_gemini_provider.py
```

### 2. 使用示例数据测试
**文件**: `test_gemini_with_samples.py`

使用项目中的示例数据测试 Gemini provider，包括：
- 使用 `sample_data/llm/prompts/chat.txt` 进行聊天测试
- 使用 `sample_data/llm/prompts/chain.txt` 进行任务分解测试
- 使用 `sample_data/text/read_aloud_test.txt` 进行文本处理测试
- 使用 `sample_data/markdown/basic.md` 进行 Markdown 处理测试
- JSON 模式测试
- 多轮对话测试

**运行方式**:
```bash
export GOOGLE_API_KEY="your-api-key"
python tests/mannual/llm/test_gemini_with_samples.py
```

## 环境变量

必需的环境变量：
- `GOOGLE_API_KEY`: Google Gemini API 密钥

可选的环境变量（有默认值）：
- `GEMINI_MODEL`: 模型名称（默认: gemini-2.5-flash-lite）
- `GEMINI_TEMPERATURE`: 温度参数（默认: 0.7）
- `GEMINI_MAX_TOKENS`: 最大 token 数（默认: 80000）
- `GEMINI_TOP_P`: Top-P 采样（默认: 1.0）
- `GEMINI_TOP_K`: Top-K 采样（默认: 40）

## 示例数据文件

测试脚本会使用以下示例数据：
- `sample_data/llm/prompts/chat.txt` - 聊天系统提示词
- `sample_data/llm/prompts/chain.txt` - 任务分解提示词
- `sample_data/text/read_aloud_test.txt` - 中英文混合文本
- `sample_data/markdown/basic.md` - Markdown 示例文件

## 注意事项

1. 这些测试会调用真实的 Google Gemini API，会产生费用
2. 确保设置了正确的 `GOOGLE_API_KEY`
3. 如果示例文件不存在，相关测试会被跳过
4. 测试结果会显示在控制台，包括通过/失败/跳过的统计
