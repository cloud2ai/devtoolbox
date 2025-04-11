"""Whisper Speech Recognition Basic Usage Example

This example demonstrates the basic usage of Whisper for speech recognition.
"""

from devtoolbox.speech.whisper_provider import WhisperConfig
from devtoolbox.speech.provider import SpeechProvider

# 1. 配置 Whisper
config = WhisperConfig(
    model_name="base",  # 可选: tiny, base, small, medium, large
    language="zh",      # 可选: zh, en, ja 等
    task="transcribe"   # 可选: transcribe, translate
)

# 2. 创建语音服务提供者
provider = SpeechProvider(config)

# 3. 语音转文本示例
audio_path = "input.wav"  # 确保音频文件存在
output_path = "transcription.txt"
provider.speech_to_text(
    speech_path=audio_path,
    output_path=output_path
)

print(f"Transcription saved to: {output_path}")

# 提示：
# 1. 确保已安装 whisper 包: pip install openai-whisper
# 2. 首次运行时会自动下载模型
# 3. 模型大小越大，识别效果越好，但需要更多计算资源