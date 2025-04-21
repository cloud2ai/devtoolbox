import logging
from devtoolbox.text_splitter.paragraph_splitter import ParagraphSplitter


# Set logging level to INFO for process visibility
logging.basicConfig(level=logging.INFO)


def process_text(text: str, title: str):
    """Process text and print results
    
    Args:
        text: Text to be processed
        title: Example title
    """
    print(f"\n{'='*50}")
    print(f"Example: {title}")
    print(f"{'='*50}")
    
    # Create ParagraphSplitter instance
    splitter = ParagraphSplitter(text)

    # Execute splitting
    paragraphs = splitter.split()

    # Print results
    print(f"\nTotal paragraphs split: {len(paragraphs)}\n")
    
    for i, para in enumerate(paragraphs, 1):
        print(f"Paragraph {i}:")
        print(f"Text length: {para.length}")
        print(f"Number of sentences: {len(para.sentences)}")
        print("Sentence list:")
        for j, sentence in enumerate(para.sentences, 1):
            print(f"  {j}. {sentence}")
        print()

    # Get keywords (returns list of tuples: [(keyword, frequency), ...])
    keywords = splitter.get_keywords(top_k=20, min_length=2)

    # Print keywords and their frequencies
    print("\nKeywords:")
    for keyword, frequency in keywords:
        print(f"{keyword}: {frequency}")


def main():
    # Chinese example text
    chinese_text = """
    变压器架构已成为深度学习模型的基石，特别是在自然语言处理（NLP）领域。它是像BERT、GPT和许多其他模型的基础。变压器模型使用自注意力机制，使它们能够并行处理单词，显著提高训练效率和在大型数据集上的性能。

    BERT（双向编码器表示来自变压器）是一个这样的模型，在各种NLP任务上取得了最先进的性能。另一个模型，GPT（生成预训练变压器），针对文本生成和语言建模等生成任务进行了优化。

    这些模型的出现彻底改变了自然语言处理领域。它们不仅提高了各种任务的性能，还使得模型能够更好地理解语言的上下文和语义。
    """

    # English example text
    english_text = """
    The transformer architecture has become a cornerstone of deep learning models, particularly in the field of Natural Language Processing (NLP). It serves as the foundation for models like BERT, GPT, and many others. Transformers use self-attention mechanisms, enabling them to process words in parallel, significantly improving training efficiency and performance on large datasets.

    BERT (Bidirectional Encoder Representations from Transformers) is one such model that has achieved state-of-the-art performance on various NLP tasks. Another model, GPT (Generative Pre-trained Transformer), is optimized for generative tasks like text generation and language modeling.

    The emergence of these models has revolutionized the field of natural language processing. They have not only improved performance across various tasks but also enabled models to better understand language context and semantics.
    """

    # Process Chinese text
    process_text(chinese_text, "Chinese Text Processing Example")

    # Process English text
    process_text(english_text, "English Text Processing Example")


if __name__ == "__main__":
    main() 