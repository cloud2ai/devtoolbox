"""Azure Speech Service Basic Usage Example

This example demonstrates the basic usage of Azure Speech Service for:
1. Text-to-speech (TTS)
2. Speech-to-text (STT)
"""

import os
from devtoolbox.speech.azure_provider import AzureConfig
from devtoolbox.speech.service import SpeechService

# 1. 配置 Azure Speech Service
config = AzureConfig()

# 2. 创建语音服务提供者
provider = SpeechService(config)

# 3. 文本转语音示例
text = "你好，这是 Azure 语音服务的测试。"
output_path = "output.wav"
provider.text_to_speech(
    text=text,
    output_path=output_path,
    speaker="zh-CN-XiaoxiaoNeural"  # 使用中文女声
)

# 4. 语音转文本示例
audio_path = "input.wav"  # 确保音频文件存在
transcription_path = "transcription.txt"
provider.speech_to_text(
    speech_path=audio_path,
    output_path=transcription_path
)

print(f"Text-to-speech output saved to: {output_path}")
print(f"Speech-to-text output saved to: {transcription_path}")

# 提示：使用前请设置环境变量
# export AZURE_SUBSCRIPTION_KEY='your-subscription-key'
# export AZURE_REGION='your-region'