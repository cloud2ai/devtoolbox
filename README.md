# devtoolbox

A Python toolbox for AI, Jira, Markdown, and other development utilities.

## About This Project

Two years ago, in December 2023, ChatGPT burst onto the scene and immediately caught my attention. At first, I thought it was just an improved version of existing AI technology. But after using it, I quickly realized that this was a groundbreaking product. I even believed that this might be the last big entrepreneurial opportunity for those of us born in the 1980s.

I still remember that day very clearly. It was a Friday, and coincidentally, it was also the day I caught COVID for the first time. In the morning, I started feeling unwell but still pushed through my work. At noon, I took some fever medicine and rested for a bit. By 3 PM, I joined our weekly company meeting. It was during this meeting that I introduced ChatGPT to my colleagues for the first time. From that moment on, I fully committed myself to exploring ChatGPT applications.

For nearly 10 years, I had been working in software development at the infrastructure level, and over time, I started feeling burnt out. But when ChatGPT emerged, I made a firm decision to shift my focus to application development. Over the next two years, I experimented with various ChatGPT-powered applications—from video translation and voiceovers to automated multimodal news generation, an I Ching fortune-telling mini-program, and eventually implementing RAG (Retrieval-Augmented Generation) within my company. Along the way, I developed a significant amount of AI-related code.

During this journey, I also gained a growing presence on multiple platforms. **I surpassed 10,000 followers on Douyin and Toutiao, and over 6,000 followers on CSDN.** My explorations in AI applications and automated content generation attracted considerable attention. These achievements reinforced my belief that in the AI era, success depends on finding real-world applications and monetization strategies.

However, as I worked on more and more projects, I realized that I hadn't properly organized or abstracted my learnings. That's why I now want to develop a new foundational AI application library—one that consolidates my technical experience and provides a more efficient foundation for future AI development.

## Features

### AI Modules
- **LLM (Large Language Models)**
  - OpenAI API integration
  - Azure OpenAI integration
  - DeepSeek API integration
  - Unified interface for multiple LLM providers
  - Configurable rate limiting and retry mechanisms

- **Speech Processing**
  - Azure Speech Services integration
  - VolcEngine Speech Services integration
  - Whisper integration
  - Text-to-Speech (TTS) capabilities
  - Speech-to-Text (STT) capabilities

### Development Tools
- **Jira Integration**
  - Jira client for issue management
  - Custom field handling
  - Bulk operations support

- **Markdown Processing**
  - Markdown to HTML conversion
  - Image downloading and processing
  - Custom formatting options
  - Table of contents generation

- **Search Engine**
  - Custom search engine implementation
  - Indexing and query capabilities

- **Text Processing**
  - Text manipulation utilities
  - Format conversion tools

- **Image Processing**
  - Image manipulation utilities
  - Format conversion tools

### Utility Modules
- **Storage**
  - Flexible storage backend support
  - File system and cloud storage options

- **Command Line Tools**
  - CLI interface for common operations
  - Script automation support

## Usage

### As a Dependency

1. **Installation**
   ```bash
   # Install all features
   pip install devtoolbox[all]
   
   # Install specific features
   pip install devtoolbox[llm]      # LLM features only
   pip install devtoolbox[speech]   # Speech features only
   pip install devtoolbox[jira]     # Jira features only
   ```

2. **Basic Usage Examples**

   **LLM Module**
   ```python
   from devtoolbox.llm import LLMService
   from devtoolbox.llm import OpenAIConfig
   
   # Initialize with OpenAI
   config = OpenAIConfig(api_key="your-api-key")
   service = LLMService(config)
   
   # Generate text
   response = service.generate("Hello, how are you?")
   ```

   **Speech Module**
   ```python
   from devtoolbox.speech import SpeechService
   from devtoolbox.speech import AzureConfig
   
   # Initialize with Azure
   config = AzureConfig(
       subscription_key="your-key",
       region="your-region"
   )
   service = SpeechService(config)
   
   # Convert text to speech
   audio = service.text_to_speech("Hello, world!")
   ```

   **Jira Integration**
   ```python
   from devtoolbox import JiraClient
   
   client = JiraClient(
       server="your-jira-server",
       username="your-username",
       api_token="your-token"
   )
   
   # Create an issue
   issue = client.create_issue(
       project="PROJ",
       summary="New issue",
       description="Issue description"
   )
   ```

3. **Configuration**
   - All modules support configuration through environment variables
   - See `DEVELOPER.md` for detailed configuration options
   - Use `.env` files for local development

### Development Guide

1. **Setup Development Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/cloud2ai/devtoolbox.git
   cd devtoolbox
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

2. **Project Structure**
   ```
   devtoolbox/
   ├── llm/              # Large Language Model implementations
   ├── speech/           # Speech processing implementations
   ├── jira/            # Jira integration
   ├── markdown/        # Markdown processing
   ├── search_engine/   # Search engine implementation
   ├── text/            # Text processing utilities
   ├── images/          # Image processing utilities
   ├── utils/           # Common utilities
   └── cmd/             # Command line tools
   ```

3. **Adding New Features**
   - Follow the driver pattern for new providers
   - Implement base classes for new modules
   - Add configuration classes for new providers
   - Update documentation in `DEVELOPER.md`

4. **Testing**
   ```bash
   # Run all tests
   pytest
   
   # Run specific module tests
   pytest tests/llm
   pytest tests/speech
   ```

5. **Documentation**
   - Update `README.md` for user-facing changes
   - Update `DEVELOPER.md` for development guidelines
   - Add docstrings to all new code
   - Follow existing code style and patterns