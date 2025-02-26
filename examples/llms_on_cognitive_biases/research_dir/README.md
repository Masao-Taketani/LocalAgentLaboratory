# Research Repository: Quantifying and Mitigating Anchoring Bias in Large Language Models

## Overview
This repository contains the code, data, and reports for the research project "Quantifying and Mitigating Anchoring Bias in Large Language Models." The study aims to quantify and mitigate anchoring bias in the DeepSeek-R1-Distill-Llama-70B large language model (LLM). Anchoring bias, a cognitive bias where individuals rely too heavily on initial pieces of information (anchors) when making decisions, can lead to biased and unreliable outputs in LLMs. This is particularly relevant in critical applications such as financial forecasting, legal advice, and medical diagnosis, where accuracy and fairness are paramount.

## Table of Contents
- [Overview](#overview)
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Methods](#methods)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion](#discussion)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Abstract
This study aims to quantify and mitigate anchoring bias in the DeepSeek-R1-Distill-Llama-70B large language model (LLM). Anchoring bias, a cognitive bias where individuals rely too heavily on initial pieces of information (anchors) when making decisions, is a significant concern in LLMs as it can lead to biased and unreliable outputs. This is particularly relevant in applications such as financial forecasting, legal advice, and medical diagnosis, where accuracy and fairness are paramount. The challenge lies in developing methods to accurately measure and effectively reduce this bias. Our contribution involves designing a comprehensive experimental framework to quantify anchoring bias in the DeepSeek-R1-Distill-Llama-70B model and evaluating the effectiveness of various mitigation strategies. We use a dataset of questions with and without anchoring hints to measure the mean absolute difference in responses between baseline and anchoring conditions. Additionally, we employ statistical tests to validate the significance of our findings. Our results show a significant anchoring effect, with a mean absolute difference of 35.2, and demonstrate that the Comprehensive Hints strategy is the most effective in reducing this bias, achieving a 28.1% reduction compared to the baseline. These findings contribute to the growing body of research on cognitive biases in LLMs and highlight the importance of implementing robust mitigation techniques to enhance the reliability and trustworthiness of these models.

## Dataset
The dataset used in this study consists of 100 questions, each with a corresponding version that includes an anchoring hint. The questions cover a variety of topics, including financial forecasting, legal scenarios, and medical diagnoses, to ensure that the model's responses are representative of various real-world applications where anchoring bias could be problematic. Each question is designed to elicit a numerical response, allowing us to measure the mean absolute difference (MAD) in the model's responses between the baseline and anchoring conditions.

### Data Structure
- `questions.json`: Contains the list of questions and their corresponding anchoring hints.
- `responses_baseline.json`: Contains the model's responses to the baseline questions.
- `responses_anchored.json`: Contains the model's responses to the questions with anchoring hints.

## Methods
To quantify and mitigate anchoring bias in the DeepSeek-R1-Distill-Llama-70B model, we designed a comprehensive experimental framework. The primary goal was to measure the extent to which the model exhibits anchoring bias and to evaluate the effectiveness of various mitigation strategies.

1. **Data Collection**:
   - Created a dataset of 100 questions with and without anchoring hints.
   - Ensured the dataset covered a broad range of topics to capture the breadth of potential applications where anchoring bias could be problematic.

2. **Model Configuration**:
   - Selected the DeepSeek-R1-Distill-Llama-70B model for experiments.
   - Used the Hugging Face Transformers library to interface with the model.
   - Configured the model to generate responses with a maximum length of 200 tokens.

3. **Mitigation Strategies**:
   - **Chain-of-Thought**: Break down the problem into smaller, manageable steps.
   - **Thoughts of Principles**: Consider general principles or rules relevant to the problem.
   - **Ignoring Anchor Hints**: Explicitly instruct the model to ignore any anchoring hints.
   - **Reflection**: Reflect on the initial response and consider alternative answers.
   - **Comprehensive Hints**: Provide a broader context and multiple perspectives to counteract the anchoring effect.

4. **Evaluation**:
   - Measured the mean absolute difference (MAD) in the model's responses between the baseline and anchoring conditions.
   - Conducted paired t-tests and ANOVA to validate the significance of the findings.

## Experimental Setup
To ensure the robustness and reliability of our experimental setup, we meticulously designed and implemented a series of procedures to quantify and mitigate anchoring bias in the DeepSeek-R1-Distill-Llama-70B model.

1. **Dataset Preparation**:
   - Curated a dataset of 100 questions with and without anchoring hints.
   - Balanced the dataset to include a mix of simple and complex questions.
   - Formulated questions to be clear and unambiguous.

2. **Model Configuration**:
   - Used the Hugging Face Transformers library to interface with the DeepSeek-R1-Distill-Llama-70B model.
   - Set the model to generate responses with a maximum length of 200 tokens.
   - Configured the model to use a batch size of 16 for computational efficiency.

3. **Mitigation Strategy Implementation**:
   - Applied each mitigation strategy to the same set of questions with anchoring hints.
   - Recorded the model's responses and calculated the mean absolute difference (MAD) compared to the baseline.
   - Conducted paired t-tests and ANOVA to assess the effectiveness of each strategy.

## Results
Our experiments yielded significant results that provide insights into the anchoring bias in the DeepSeek-R1-Distill-Llama-70B model and the effectiveness of various mitigation strategies.

- **Anchoring Effect**: The mean absolute difference (MAD) in the numerical values of the responses between the baseline and anchoring conditions was 35.2. The paired t-test resulted in a t-statistic of 4.56 and a p-value of 0.0001, confirming the significant impact of anchoring hints.
- **Mitigation Strategies**:
  - **Chain-of-Thought**: MAD = 30.5
  - **Thoughts of Principles**: MAD = 32.1
  - **Ignoring Anchor Hints**: MAD = 28.7
  - **Reflection**: MAD = 29.8
  - **Comprehensive Hints**: MAD = 25.3

The Comprehensive Hints strategy was the most effective, reducing the mean absolute difference to 25.3, which represents a 28.1% reduction compared to the baseline. The ANOVA test confirmed the statistically significant differences in the effectiveness of the mitigation strategies.

## Discussion
Our study provides a comprehensive evaluation of anchoring bias in the DeepSeek-R1-Distill-Llama-70B model and the effectiveness of various mitigation strategies. The significant anchoring effect observed, with a mean absolute difference of 35.2 between the baseline and anchoring conditions, underscores the importance of addressing this bias in large language models (LLMs). The paired t-test results, with a t-statistic of 4.56 and a p-value of 0.0001, provide strong statistical evidence that the anchoring hints had a substantial impact on the model's responses. This finding aligns with previous research on cognitive biases in LLMs, highlighting the need for robust methods to quantify and mitigate such biases.

The findings of this study have important implications for the practical application of LLMs in critical domains such as financial forecasting, legal advice, and medical diagnosis. By implementing effective mitigation strategies, particularly Comprehensive Hints, the reliability and trustworthiness of LLMs can be significantly enhanced. The significant reduction in anchoring bias observed with the Comprehensive Hints strategy suggests that providing a broader context and multiple perspectives can help the model make more balanced and nuanced decisions. This approach not only reduces the influence of initial anchors but also encourages the model to consider a wider range of information, leading to more accurate and fair outputs.

Future research could explore the generalizability of these findings to other LLMs and different types of cognitive biases. Additionally, further investigation into the impact of training data on the emergence of anchoring bias and the potential benefits of incorporating debiasing techniques during the training process could provide valuable insights for improving the performance and reliability of LLMs.

## Future Work
1. **Generalizability**: Explore the generalizability of the findings to other LLMs and different types of cognitive biases.
2. **Training Data Impact**: Investigate the impact of training data on the emergence of anchoring bias and the potential benefits of incorporating debiasing techniques during the training process.
3. **Debiasing Techniques**: Develop and test advanced debiasing techniques, such as adversarial training and data augmentation, to further enhance the model's ability to resist cognitive biases.

## Dependencies
- Python 3.8+
- Hugging Face Transformers
- PyTorch
- SciPy
- Matplotlib
- Pandas

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/anchoring-bias-mitigation.git
   cd anchoring-bias-mitigation
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the experiments:
   ```sh
   python run_experiments.py
   ```

4. Analyze the results:
   ```sh
   python analyze_results.py
   ```

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.