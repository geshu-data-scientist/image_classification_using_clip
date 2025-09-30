📄 Document vs. Non-Document Classifier
A simple web application built with Streamlit that classifies uploaded images as either a "Document" or a "Non-Document" using OpenAI's powerful CLIP model.

✨ Features
Zero-Shot Classification: The app can identify documents without being explicitly trained on a document-specific dataset.

Powered by OpenAI CLIP: Leverages the state-of-the-art clip-vit-base-patch32 model to understand the content of an image in relation to text descriptions.

Simple Interface: An easy-to-use drag-and-drop uploader for quick image analysis.

Real-time Results: Get an instant classification and a confidence score directly in the browser.

🧠 How It Works
This application is a great example of zero-shot learning with a multi-modal model. The core logic is straightforward:

Text Prompts: Two lists of text descriptions (prompts) are defined: one describing various types of documents (e.g., "a photo of an Aadhaar card," "a photo of a printed document") and another for non-documents (e.g., "a photo of a damaged car part," "a photo of a car accident").

Image-Text Comparison: When you upload an image, the CLIP model simultaneously analyzes the visual content of the image and the semantic meaning of all text prompts.

Scoring: CLIP calculates a similarity score between the image and each text prompt.

Classification: The application groups the scores. It finds the highest score among the "document" prompts and the highest score among the "non-document" prompts. The final classification is given to the group with the overall highest score.
