# devtoolbox

开发中使用的工具集，包含AI，JIRA，Markdown处理和其他使用的开发工具库，不断扩展中。

## 关于本项目的来源

在两年前的 2023 年 12 月，ChatGPT 横空出世，一下子闯入了我的视野。起初，我以为这只是现有 AI 技术的优化升级版，但在亲身体验后，我立刻意识到——这是一款颠覆性的产品。甚至，我认为这或许是我们 80 后最后一次能够抓住的创业机会。

那一天的情景我依然记忆犹新。那是一个周五，恰好也是我第一次感染新冠的日子。上午，我已经感觉到身体有些异样，但仍然坚持完成了工作。中午吃了退烧药小憩片刻后，下午 3 点参加了公司的周例会。就在这次例会上，我第一次向全体同事介绍了 ChatGPT。从那一刻起，我开始全身心投入 ChatGPT 应用的研究。

过去近 10 年，我长期从事基础架构层的软件开发，渐渐感到倦怠。而当 ChatGPT 出现后，我毫不犹豫地选择从应用层切入。接下来的两年里，我不断尝试利用 ChatGPT 进行各种应用探索——从最早的视频翻译与配音，到自动化多模态新闻，再到后来的易经算命小程序，直至如今在公司内部推行 RAG（检索增强生成），开发了大量与 AI 应用相关的代码。

在这一过程中，我也逐渐在多个平台上积累了一定的影响力。我的 **抖音和今日头条粉丝突破 1 万**，**CSDN 关注人数超过 6000**，在 AI 技术应用、自动化内容生成等领域的探索吸引了不少的关注。这些成果不仅验证了 AI 在应用层的可行性，也让我更加坚定地认为，AI 时代的关键在于如何真正落地和变现。

然而，随着项目的不断增多，我发现自己尚未对这些经验进行系统化的整理和抽象。因此，我希望开发一个全新的 AI 应用开发基础库，沉淀自己的技术积累，为后续的 AI 研发提供更高效的支持。

## 功能

### AI 模块
- **LLM（大型语言模型）**
  - OpenAI API 集成
  - Azure OpenAI 集成
  - DeepSeek API 集成
  - 多个 LLM 提供者的统一接口
  - 可配置的速率限制和重试机制

- **语音处理**
  - Azure 语音服务集成
  - VolcEngine 语音服务集成
  - Whisper 集成
  - 文本转语音（TTS）功能
  - 语音转文本（STT）功能，包含详细的元数据输出

### 开发工具
- **Jira 集成**
  - 用于问题管理的 Jira 客户端
  - 自定义字段处理
  - 批量操作支持

- **Markdown 处理**
  - Markdown 转 HTML 转换
  - 图像下载和处理
  - 自定义格式选项
  - 目录生成

- **搜索引擎**
  - 自定义搜索引擎实现
  - 索引和查询功能

- **文本处理**
  - 文本操作工具
  - 格式转换工具

- **图像处理**
  - 图像操作工具
  - 格式转换工具

### 实用模块
- **存储**
  - 灵活的存储后端支持
  - 文件系统和云存储选项

- **命令行工具**
  - 常见操作的 CLI 接口
  - 脚本自动化支持

## 使用

### 作为依赖

1. **安装**
   ```bash
   # 安装所有功能
   pip install devtoolbox[all]

   # 安装特定功能
   pip install devtoolbox[llm]      # 仅 LLM 功能
   pip install devtoolbox[speech]   # 仅语音功能
   pip install devtoolbox[jira]     # 仅 Jira 功能
   ```

2. **基本使用示例**

   **LLM 模块**
   ```python
   from devtoolbox.llm import LLMService
   from devtoolbox.llm import OpenAIConfig

   # 使用 OpenAI 初始化
   config = OpenAIConfig(api_key="your-api-key")
   service = LLMService(config)

   # 生成文本
   response = service.generate("你好，你怎么样？")
   ```

      **语音模块**
   ```python
   from devtoolbox.speech import SpeechService
   from devtoolbox.speech import AzureConfig

   # 使用 Azure 初始化
   config = AzureConfig(
       subscription_key="your-key",
       region="your-region"
   )
   service = SpeechService(config)

   # 将文本转换为语音
   audio = service.text_to_speech("你好，世界！")

   # 将语音转换为文本（同时生成转录文本和元数据）
   service.speech_to_text(
       "input.wav",
       "output.txt",
       output_format="txt"
   )
   # 同时会创建 output.txt.metadata.json 文件，包含详细的处理信息
   ```

   **Jira 集成**
   ```python
   from devtoolbox import JiraClient

   client = JiraClient(
       server="your-jira-server",
       username="your-username",
       api_token="your-token"
   )

   # 创建一个问题
   issue = client.create_issue(
       project="PROJ",
       summary="新问题",
       description="问题描述"
   )
   ```

3. **配置**
   - 所有模块支持通过环境变量进行配置
   - 请参见 `DEVELOPER.md` 以获取详细的配置选项
   - 使用 `.env` 文件进行本地开发

4. **语音转文本元数据输出**

   语音转文本功能现在会生成详细的元数据文件（.metadata.json），包含：

   - **音频文件信息**：总时长、采样率、声道数、位深度等
   - **处理统计**：音频分块数量、总文本长度、输出文件路径等
   - **存储优化信息**：压缩比例、节省空间大小、格式转换详情等
   - **分块详细信息**：每个音频块的时间戳、文本内容、文件大小等

   这些元数据可用于：
   - 音频处理质量分析
   - 存储优化效果评估
   - 进一步的数据处理和分析

### 开发指南

1. **设置开发环境**
   ```bash
   # 克隆代码库
   git clone https://github.com/cloud2ai/devtoolbox.git
   cd devtoolbox

   # 创建并激活虚拟环境
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上: venv\Scripts\activate

   # 安装开发依赖
   pip install -e ".[dev]"
   ```

2. **项目结构**
   ```
   devtoolbox/
   ├── llm/              # 大型语言模型实现
   ├── speech/           # 语音处理实现
   ├── jira/            # Jira 集成
   ├── markdown/        # Markdown 处理
   ├── search_engine/   # 搜索引擎实现
   ├── text/            # 文本处理工具
   ├── images/          # 图像处理工具
   ├── utils/           # 通用工具
   └── cmd/             # 命令行工具
   ```

3. **添加新功能**
   - 遵循新提供者的驱动模式
   - 为新模块实现基类
   - 为新提供者添加配置类
   - 更新 `DEVELOPER.md` 中的文档

4. **测试**
   ```bash
   # 运行所有测试
   pytest

   # 运行特定模块测试
   pytest tests/llm
   pytest tests/speech
   ```

5. **文档**
   - 更新 `README.md` 以反映用户可见的更改
   - 更新 `DEVELOPER.md` 以反映开发指南
   - 为所有新代码添加文档字符串
   - 遵循现有的代码风格和模式