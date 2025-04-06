import logging

from devtoolbox.text.splitter import TokenSplitter


def main():
    """Example usage of TokenSplitter."""
    # Sample text for demonstration in English
    text_en = """
The transformer architecture has become a cornerstone of deep learning models,
especially in the field of natural language processing (NLP).
It is the foundation for models like BERT, GPT, and many others.
Transformer models use self-attention mechanisms that allow them to process
words in parallel, significantly improving training efficiency and
performance on large datasets.  BERT (Bidirectional Encoder Representations
from Transformers) is one such model that has achieved state-of-the-art
performance on various NLP tasks. Another model, GPT (Generative Pre-trained
Transformer), is optimized for generative tasks like text generation and
language modeling.
"""

    # Sample text for demonstration in Chinese
    text_zh = """
变压器架构已成为深度学习模型的基石，特别是在自然语言
处理（NLP）领域。它是像BERT、GPT和许多其他模型的基础。变压器模型使用自
注意力机制，使它们能够并行处理单词，显著提高训练效率和在大型数据集上的性能。 BERT
（双向编码器表示来自变压器）是一个这样的模型，在各种NLP任务上取得了
最先进的性能。 另一个模型，GPT（生成预训练变压器），针对文本生成和语言建模
等生成任务进行了优化。
"""

    # Initialize the splitter with default GPT-4 model
    splitter_gpt4 = TokenSplitter(
        max_tokens=100,  # Set a small token limit to demonstrate splitting
        overlap_tokens=0,
        model_name="gpt-4"
    )

    # Split the English text into paragraphs
    paragraphs_gpt4_en = splitter_gpt4.split_text(text_en)

    # Print each paragraph and its token count for English text
    print("=== Using GPT-4 tokenizer for English text ===")
    for i, para in enumerate(paragraphs_gpt4_en, 1):
        print(f"\nParagraph {i}:")
        print(f"Content: {para.content}")
        print(f"Token count: {para.token_count}")
        print(f"Sentences count: {len(para.sentences)}")

    # Split the Chinese text into paragraphs
    paragraphs_gpt4_zh = splitter_gpt4.split_text(text_zh)

    # Print each paragraph and its token count for Chinese text
    print("=== Using GPT-4 tokenizer for Chinese text ===")
    for i, para in enumerate(paragraphs_gpt4_zh, 1):
        print(f"\nParagraph {i}:")
        print(f"Content: {para.content}")
        print(f"Token count: {para.token_count}")
        print(f"Sentences count: {len(para.sentences)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
